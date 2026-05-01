"""Tool-Output Compiler tests — Phase 5.

Validates that tool outputs and structured content are handled correctly:
1. ContentProfiler detects specialized content types (logs, diffs, traces, etc.)
2. ToolOutputFilter applies schema-aware projection and summarization
3. FormatConverter never corrupts JSON or code blocks
4. OutputCleanup preserves fenced code blocks
5. MessageDeduplicator preserves tool messages
6. Cross-transform pipelines are safe on structured content

Every optimization needs:
- a baseline
- a toggle
- a test
- a measurable signal
"""

from __future__ import annotations

import json

import pytest

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.result import unwrap
from lattice.core.transport import Message, Request, Response
from lattice.transforms.content_profiler import ContentProfile, ContentProfiler
from lattice.transforms.format_conv import FormatConverter
from lattice.transforms.message_dedup import MessageDeduplicator
from lattice.transforms.output_cleanup import OutputCleanup
from lattice.transforms.reference_sub import ReferenceSubstitution
from lattice.transforms.tool_filter import ToolOutputFilter

# =============================================================================
# ContentProfiler — new profile detection
# =============================================================================

class TestContentProfilerNewProfiles:
    """ContentProfiler correctly classifies specialized content types."""

    def _make_request(self, content: str, role: str = "user") -> Request:
        return Request(messages=[Message(role=role, content=content)])

    def test_log_output_detection(self) -> None:
        """Timestamped log lines are classified as LOG_OUTPUT."""
        logs = (
            "2024-01-15T10:23:45 INFO  Starting service v1.2.3 on port 8080\n"
            "2024-01-15T10:23:46 DEBUG Connecting to database at postgres://localhost\n"
            "2024-01-15T10:23:47 ERROR Connection timeout after 30 seconds\n"
            "2024-01-15T10:23:48 WARN  Retrying in 5 seconds (attempt 1 of 3)\n"
            "2024-01-15T10:23:49 INFO  Retry successful\n"
            "2024-01-15T10:23:50 DEBUG Processing request id 12345\n"
            "2024-01-15T10:23:51 INFO  Request completed in 12ms\n"
            "2024-01-15T10:23:52 WARN  High memory usage detected: 85%\n"
            "2024-01-15T10:23:53 ERROR Failed to write to disk: no space left\n"
            "2024-01-15T10:23:54 CRITICAL Service shutting down\n"
        )
        profiler = ContentProfiler(short_threshold_tokens=5)
        profile = profiler._classify(self._make_request(logs))
        assert profile == ContentProfile.LOG_OUTPUT

    def test_diff_output_detection(self) -> None:
        """Unified diff is classified as DIFF_OUTPUT."""
        diff = (
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -10,5 +10,5 @@\n"
            " def foo():\n"
            "-    return 1\n"
            "+    return 2\n"
            " \n"
            " def bar():\n"
            "-    pass\n"
            "+    return None\n"
            " \n"
            " def baz():\n"
            "-    x = 1\n"
            "+    x = 2\n"
            " \n"
            " def qux():\n"
            "-    y = 3\n"
            "+    y = 4\n"
        )
        profiler = ContentProfiler(short_threshold_tokens=5)
        profile = profiler._classify(self._make_request(diff))
        assert profile == ContentProfile.DIFF_OUTPUT

    def test_stack_trace_detection(self) -> None:
        """Python stack trace is classified as STACK_TRACE."""
        trace = (
            "Traceback (most recent call last):\n"
            '  File "/app/server.py", line 42, in handle_request\n'
            "    result = process(data)\n"
            '  File "/app/worker.py", line 88, in process\n'
            "    intermediate = transform(data)\n"
            '  File "/app/transform.py", line 15, in transform\n'
            "    validated = validate(data)\n"
            '  File "/app/validate.py", line 22, in validate\n'
            "    raise ValueError('bad input provided by user')\n"
            "ValueError: bad input provided by user\n"
        )
        profiler = ContentProfiler(short_threshold_tokens=5)
        profile = profiler._classify(self._make_request(trace))
        assert profile == ContentProfile.STACK_TRACE

    def test_grep_output_detection(self) -> None:
        """Grep results are classified as GREP_OUTPUT."""
        # Generate many grep lines to overwhelm any incidental narrative score
        # Use _ instead of . in filenames to avoid triggering sentence split on dots
        grep_lines = [
            f"src/a_py:{i}:def func_{i}():"
            for i in range(1, 51)
        ]
        grep = "\n".join(grep_lines)
        profiler = ContentProfiler(short_threshold_tokens=5)
        profile = profiler._classify(self._make_request(grep))
        assert profile == ContentProfile.GREP_OUTPUT

    def test_file_tree_detection(self) -> None:
        """Directory tree listings are classified as FILE_TREE."""
        tree = (
            "src/\n"
            "├── main.py\n"
            "├── worker.py\n"
            "├── api/\n"
            "│   ├── routes.py\n"
            "│   └── handlers.py\n"
            "└── utils/\n"
            "    ├── __init__.py\n"
            "    ├── helpers.py\n"
            "    └── parsers.py\n"
        )
        profiler = ContentProfiler(short_threshold_tokens=5)
        profile = profiler._classify(self._make_request(tree))
        assert profile == ContentProfile.FILE_TREE

    def test_mcp_output_detection(self) -> None:
        """MCP tool results with is_error are classified as MCP_OUTPUT."""
        mcp = json.dumps({
            "tool_call_id": "call_123",
            "is_error": False,
            "content": "File contents here with some additional text to make it longer"
        })
        profiler = ContentProfiler(short_threshold_tokens=5)
        profile = profiler._classify(self._make_request(mcp))
        assert profile == ContentProfile.MCP_OUTPUT

    def test_strategy_for_log_output(self) -> None:
        """LOG_OUTPUT strategy disables semantic_compress and enables dedup."""
        profiler = ContentProfiler()
        strategy = profiler._select_strategy(ContentProfile.LOG_OUTPUT, Request(messages=[]))
        assert strategy["semantic_compress"] is False
        assert strategy["message_dedup"] is True
        assert strategy["reference_sub"] is True

    def test_strategy_for_stack_trace(self) -> None:
        """STACK_TRACE strategy disables cleanup and format_conversion."""
        profiler = ContentProfiler()
        strategy = profiler._select_strategy(ContentProfile.STACK_TRACE, Request(messages=[]))
        assert strategy["output_cleanup"] is False
        assert strategy["semantic_compress"] is False
        assert strategy["format_conversion"] is False

    def test_strategy_for_file_tree(self) -> None:
        """FILE_TREE strategy disables cleanup and tool_filter."""
        profiler = ContentProfiler()
        strategy = profiler._select_strategy(ContentProfile.FILE_TREE, Request(messages=[]))
        assert strategy["output_cleanup"] is False
        assert strategy["tool_filter"] is False
        assert strategy["reference_sub"] is True

    def test_short_content_bypass(self) -> None:
        """Very short content is classified SHORT regardless of signals."""
        profiler = ContentProfiler(short_threshold_tokens=500)
        req = Request(messages=[Message(role="user", content="hi")])
        profile = profiler._classify(req)
        assert profile == ContentProfile.SHORT


# =============================================================================
# ToolOutputFilter — schema projection and summarization
# =============================================================================

class TestToolOutputFilterSchemaAware:
    """ToolOutputFilter keeps schema-referenced fields and identity fields."""

    def test_schema_aware_projection(self) -> None:
        """Only schema fields + identity fields are kept."""
        tools = [{
            "type": "function",
            "function": {
                "name": "get_user",
                "parameters": {
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                    }
                }
            }
        }]
        request = Request(
            messages=[Message(
                role="tool",
                content=json.dumps({
                    "id": "123",
                    "name": "Alice",
                    "created_at": "2024-01-01",
                    "internal_secret": "abc",
                }),
            )],
            tools=tools,
        )
        filt = ToolOutputFilter()
        result = unwrap(filt.process(request, TransformContext()))
        parsed = json.loads(result.messages[0].content)
        assert "id" in parsed
        assert "name" in parsed
        assert "created_at" not in parsed
        assert "internal_secret" not in parsed

    def test_identity_fields_always_preserved(self) -> None:
        """id, name, type, status, error, result are always preserved."""
        request = Request(
            messages=[Message(
                role="tool",
                content=json.dumps({
                    "id": "123",
                    "status": "ok",
                    "error": None,
                    "result": 42,
                    "metadata": {"extra": "data"},
                }),
            )],
            tools=[],
        )
        filt = ToolOutputFilter()
        result = unwrap(filt.process(request, TransformContext()))
        parsed = json.loads(result.messages[0].content)
        assert parsed["id"] == "123"
        assert parsed["status"] == "ok"
        assert "metadata" not in parsed

    def test_summarize_large_list(self) -> None:
        """Very large outputs are summarized with statistics."""
        rows = [{"id": i, "value": i * 10} for i in range(200)]
        request = Request(
            messages=[Message(
                role="tool",
                content=json.dumps(rows),
            )],
        )
        filt = ToolOutputFilter(summarize_threshold_tokens=10)
        result = unwrap(filt.process(request, TransformContext()))
        parsed = json.loads(result.messages[0].content)
        assert parsed.get("_summary") is True
        assert parsed.get("total_count") == 200
        assert "sample" in parsed
        assert "averages" in parsed

    def test_non_json_unchanged(self) -> None:
        """Non-JSON tool output is left untouched."""
        request = Request(
            messages=[Message(role="tool", content="This is plain text output.")],
        )
        filt = ToolOutputFilter()
        result = unwrap(filt.process(request, TransformContext()))
        assert result.messages[0].content == "This is plain text output."

    def test_tool_role_required(self) -> None:
        """User messages without tool_call_id or is_tool_output are skipped."""
        request = Request(
            messages=[Message(
                role="user",
                content=json.dumps({"id": "123", "noise": "xxx"}),
            )],
        )
        filt = ToolOutputFilter()
        result = unwrap(filt.process(request, TransformContext()))
        # Should remain unchanged
        assert "noise" in result.messages[0].content

    def test_no_savings_skips_mutation(self) -> None:
        """ToolOutputFilter leaves content unchanged when filtering would expand it."""
        content = json.dumps({
            "id": "123",
            "name": "Alice",
            "created_at": "2024-01-01",
            "internal_secret": "abc",
        })
        request = Request(
            messages=[Message(role="tool", content=content)],
        )
        filt = ToolOutputFilter(min_savings_chars=1000)
        result = unwrap(filt.process(request, TransformContext()))
        assert result.messages[0].content == content


# =============================================================================
# FormatConverter — structured content safety
# =============================================================================

class TestFormatConverterSafety:
    """FormatConverter never corrupts JSON or code blocks."""

    def test_json_array_of_objects_converts_to_csv(self) -> None:
        """Uniform JSON array becomes CSV."""
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        request = Request(messages=[Message(role="user", content=json.dumps(data))])
        conv = FormatConverter()
        result = unwrap(conv.process(request, TransformContext()))
        assert "a,b" in result.messages[0].content or "a\tb" in result.messages[0].content

    def test_mixed_keys_not_converted(self) -> None:
        """Irregular JSON is not converted."""
        data = [{"a": 1}, {"b": 2}]
        request = Request(messages=[Message(role="user", content=json.dumps(data))])
        conv = FormatConverter()
        result = unwrap(conv.process(request, TransformContext()))
        # Should remain as JSON since keys are mixed
        assert result.messages[0].content == json.dumps(data)

    def test_nested_dict_converts_to_yaml(self) -> None:
        """Nested dict becomes YAML."""
        data = {"server": {"host": "0.0.0.0", "port": 8080}}
        request = Request(messages=[Message(role="user", content=json.dumps(data))])
        conv = FormatConverter()
        result = unwrap(conv.process(request, TransformContext()))
        content = result.messages[0].content
        assert "server:" in content
        assert "host:" in content

    def test_markdown_table_to_csv(self) -> None:
        """Markdown table is converted to CSV."""
        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |\n"
        request = Request(messages=[Message(role="user", content=md)])
        conv = FormatConverter()
        result = unwrap(conv.process(request, TransformContext()))
        lines = result.messages[0].content.strip().splitlines()
        assert lines[0].replace("\t", ",") == "Name,Age"

    def test_tool_calls_message_skipped(self) -> None:
        """Messages with tool_calls are never modified."""
        request = Request(
            messages=[Message(
                role="assistant",
                content="Using tool...",
                tool_calls=[{"id": "1", "function": {"name": "foo", "arguments": "{}"}}],
            )],
        )
        conv = FormatConverter()
        result = unwrap(conv.process(request, TransformContext()))
        assert result.messages[0].content == "Using tool..."


# =============================================================================
# OutputCleanup — code block preservation
# =============================================================================

class TestOutputCleanupCodeBlockSafety:
    """OutputCleanup never modifies content inside fenced code blocks."""

    def test_code_block_preserved(self) -> None:
        """Patterns inside ``` are not cleaned."""
        content = (
            "Sure! Here is the code:\n\n"
            "```python\n"
            "# Let me know if you need anything else\n"
            "print('hello')\n"
            "```\n\n"
            "Let me know if you have any questions."
        )
        response = Response(content=content)
        cleanup = OutputCleanup()
        result = cleanup.reverse(response, TransformContext())
        # Code block comment should remain
        assert "# Let me know if you need anything else" in result.content
        # But trailing fluff outside block should be removed
        assert "Let me know if you have any questions." not in result.content

    def test_tool_calls_skipped(self) -> None:
        """Responses with tool_calls are never cleaned."""
        response = Response(
            content="Let me know if you need anything else.",
            tool_calls=[{"id": "1", "function": {"name": "foo", "arguments": "{}"}}],
        )
        cleanup = OutputCleanup()
        result = cleanup.reverse(response, TransformContext())
        assert result.content == "Let me know if you need anything else."

    def test_min_savings_threshold(self) -> None:
        """Cleanup only applies if savings exceed threshold."""
        content = "Hello world."
        response = Response(content=content)
        cleanup = OutputCleanup(min_savings_chars=100)
        result = cleanup.reverse(response, TransformContext())
        assert result.content == "Hello world."


# =============================================================================
# MessageDeduplicator — tool message preservation
# =============================================================================

class TestMessageDeduplicatorToolSafety:
    """MessageDeduplicator never removes tool messages."""

    def test_tool_messages_preserved(self) -> None:
        """Messages with role 'tool' are never deduplicated."""
        request = Request(
            messages=[
                Message(role="user", content="Run test"),
                Message(role="tool", content="ok"),
                Message(role="tool", content="ok"),
                Message(role="assistant", content="Done"),
            ],
        )
        dedup = MessageDeduplicator()
        result = unwrap(dedup.process(request, TransformContext()))
        tool_count = sum(1 for m in result.messages if m.role == "tool")
        assert tool_count == 2

    def test_exact_duplicates_removed(self) -> None:
        """Exact duplicate user messages are removed."""
        request = Request(
            messages=[
                Message(role="user", content="Hello world this is a longer message"),
                Message(role="user", content="Hello world this is a longer message"),
                Message(role="user", content="Another distinct message content here"),
                Message(role="user", content="Another distinct message content here"),
                Message(role="user", content="Final unique message at the end"),
            ],
        )
        dedup = MessageDeduplicator(preserve_last_n=1, min_message_length=5)
        result = unwrap(dedup.process(request, TransformContext()))
        assert len(result.messages) == 3  # Hello, Another, Final

    def test_last_n_preserved(self) -> None:
        """Last N messages are always preserved."""
        request = Request(
            messages=[
                Message(role="user", content="A"),
                Message(role="user", content="B"),
                Message(role="user", content="C"),
                Message(role="user", content="D"),
            ],
        )
        dedup = MessageDeduplicator(preserve_last_n=2)
        result = unwrap(dedup.process(request, TransformContext()))
        # Last 2 preserved, first 2 deduped if duplicate (they are not)
        assert len(result.messages) == 4


# =============================================================================
# Cross-transform safety pipeline
# =============================================================================

class TestCrossTransformSafety:
    """Full pipeline never corrupts structured content."""

    @pytest.mark.asyncio
    async def test_json_tool_output_survives_pipeline(self) -> None:
        """Tool output JSON is preserved through the full pipeline."""
        config = LatticeConfig(compression_mode="safe")
        pipeline = CompressorPipeline(config=config)
        pipeline.register(ContentProfiler())
        pipeline.register(ToolOutputFilter())
        pipeline.register(FormatConverter())
        pipeline.register(OutputCleanup())

        original_json = {"id": "123", "status": "ok", "result": [1, 2, 3]}
        request = Request(
            messages=[Message(role="tool", content=json.dumps(original_json))],
        )
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        # Should still be valid JSON
        parsed = json.loads(modified.messages[0].content)
        assert parsed["id"] == "123"
        assert parsed["status"] == "ok"

    @pytest.mark.asyncio
    async def test_code_block_survives_pipeline(self) -> None:
        """Code blocks inside user messages survive pipeline."""
        config = LatticeConfig(compression_mode="safe")
        pipeline = CompressorPipeline(config=config)
        pipeline.register(ContentProfiler())
        pipeline.register(ReferenceSubstitution())
        pipeline.register(OutputCleanup())

        request = Request(
            messages=[Message(
                role="user",
                content=(
                    "```python\n"
                    "x = '550e8400-e29b-41d4-a716-446655440000'\n"
                    "print(x)\n"
                    "```"
                ),
            )],
        )
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        # Code block should remain intact
        assert "```python" in modified.messages[0].content
        assert "550e8400-e29b-41d4-a716-446655440000" in modified.messages[0].content

    @pytest.mark.asyncio
    async def test_metrics_signal_present(self) -> None:
        """Pipeline records measurable signals for every transform."""
        config = LatticeConfig(
            compression_mode="safe",
            transform_content_profiler=True,
            transform_tool_filter=True,
            transform_message_dedup=True,
        )
        pipeline = CompressorPipeline(config=config)
        pipeline.register(ContentProfiler())
        pipeline.register(ToolOutputFilter())
        pipeline.register(MessageDeduplicator())

        request = Request(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="tool", content=json.dumps({"id": "1", "value": "x"})),
            ],
        )
        context = TransformContext()
        result = await pipeline.process(request, context)
        unwrap(result)

        assert "content_profiler" in context.metrics["transforms"]
        assert "tool_filter" in context.metrics["transforms"] or "message_dedup" in context.metrics["transforms"]


# =============================================================================
# ToolOutputFilter — structure-aware projection (Phase 5)
# =============================================================================

class TestToolOutputFilterStructureAware:
    """ToolOutputFilter applies structure-aware compression."""

    def test_log_output_projection(self) -> None:
        """Log with repeated errors gets collapsed."""
        log = (
            "2024-01-15T10:00:01 ERROR Connection failed to db1\n"
            "2024-01-15T10:00:02 ERROR Connection failed to db1\n"
            "2024-01-15T10:00:03 ERROR Connection failed to db1\n"
            "2024-01-15T10:00:04 INFO  Health check passed\n"
            "2024-01-15T10:00:05 INFO  Health check passed\n"
            "2024-01-15T10:00:06 INFO  Health check passed\n"
            "2024-01-15T10:00:07 INFO  Health check passed\n"
            "2024-01-15T10:00:08 WARN  High latency 120ms\n"
        )
        filt = ToolOutputFilter()
        request = Request(
            messages=[Message(role="tool", content=log)],
            metadata={"_lattice_profile": "log_output"},
        )
        result = unwrap(filt.process(request, TransformContext()))
        content = result.messages[0].content
        # Repeated ERROR should be deduplicated to one line
        assert content.count("ERROR Connection failed to db1") == 1
        # INFO should be collapsed with ellipsis (more than 3 lines)
        assert "..." in content
        # WARN should be preserved
        assert "WARN  High latency" in content

    def test_diff_output_projection(self) -> None:
        """Diff keeps only changed hunks; context is collapsed."""
        diff = (
            "diff --git a/src/main.py b/src/main.py\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,10 +1,10 @@\n"
            " unchanged line 1\n"
            " unchanged line 2\n"
            " unchanged line 3\n"
            " unchanged line 4\n"
            " unchanged line 5\n"
            " unchanged line 6\n"
            " unchanged line 7\n"
            " unchanged line 8\n"
            " unchanged line 9\n"
            "-old value\n"
            "+new value\n"
            " unchanged line 10\n"
        )
        filt = ToolOutputFilter()
        request = Request(
            messages=[Message(role="tool", content=diff)],
            metadata={"_lattice_profile": "diff_output"},
        )
        result = unwrap(filt.process(request, TransformContext()))
        content = result.messages[0].content
        # Changed lines preserved
        assert "-old value" in content
        assert "+new value" in content
        # Hunk header preserved
        assert "@@" in content

    def test_stack_trace_projection(self) -> None:
        """Stack trace deduplicates repeated frames."""
        trace = (
            "Traceback (most recent call last):\n"
            '  File "/app/server.py", line 42, in handle_request\n'
            "    result = process(data)\n"
            '  File "/app/worker.py", line 88, in process\n'
            "    intermediate = transform(data)\n"
            '  File "/app/server.py", line 42, in handle_request\n'
            "    result = process(data)\n"
            '  File "/app/worker.py", line 88, in process\n'
            "    intermediate = transform(data)\n"
            "ValueError: bad input\n"
        )
        filt = ToolOutputFilter()
        request = Request(
            messages=[Message(role="tool", content=trace)],
            metadata={"_lattice_profile": "stack_trace"},
        )
        result = unwrap(filt.process(request, TransformContext()))
        content = result.messages[0].content
        # Unique frames should appear once each
        assert content.count("server.py") == 1
        assert content.count("worker.py") == 1
        assert "ValueError: bad input" in content

    def test_grep_output_projection(self) -> None:
        """Grep output deduplicates repeated filenames."""
        grep = (
            "src/main.py:10:def foo():\n"
            "src/main.py:20:def bar():\n"
            "src/main.py:30:def baz():\n"
            "src/main.py:40:def qux():\n"
            "src/utils.py:5:import os\n"
            "src/utils.py:6:import sys\n"
        )
        filt = ToolOutputFilter()
        request = Request(
            messages=[Message(role="tool", content=grep)],
            metadata={"_lattice_profile": "grep_output"},
        )
        result = unwrap(filt.process(request, TransformContext()))
        content = result.messages[0].content
        # Should preserve first and last for main.py
        assert "src/main.py:10" in content
        assert "src/main.py:40" in content
        # utils.py should be preserved
        assert "src/utils.py" in content

    def test_mcp_output_projection(self) -> None:
        """MCP tool result filters safely."""
        mcp = json.dumps({
            "tool_call_id": "call_123",
            "is_error": False,
            "content": "File contents here",
            "type": "text",
            "tool": "read_file",
            "internal_blob": "should_be_removed",
            "_private": "also_removed",
        })
        filt = ToolOutputFilter()
        request = Request(
            messages=[Message(role="tool", content=mcp)],
            metadata={"_lattice_profile": "mcp_output"},
        )
        result = unwrap(filt.process(request, TransformContext()))
        parsed = json.loads(result.messages[0].content)
        assert parsed["tool_call_id"] == "call_123"
        assert parsed["content"] == "File contents here"
        assert "internal_blob" not in parsed
        assert "_private" not in parsed


# =============================================================================
# FormatConverter — new content type handling (Phase 5)
# =============================================================================

class TestFormatConverterNewTypes:
    """FormatConverter handles diffs and logs."""

    def test_wide_table_handling(self) -> None:
        """Wide tables auto-switch to TSV."""
        data = [{f"col_{i}": f"value_{i}" for i in range(12)} for _ in range(3)]
        request = Request(messages=[Message(role="user", content=json.dumps(data))])
        conv = FormatConverter()
        result = unwrap(conv.process(request, TransformContext()))
        content = result.messages[0].content
        # Should use tabs as delimiter due to >8 columns
        assert "\t" in content

    def test_nested_json_arrays(self) -> None:
        """Nested arrays inside JSON objects are preserved (not corrupted)."""
        data = {"items": [[1, 2], [3, 4]], "meta": "info"}
        original = json.dumps(data)
        request = Request(messages=[Message(role="user", content=original)])
        conv = FormatConverter()
        result = unwrap(conv.process(request, TransformContext()))
        # Should remain valid JSON (not corrupted)
        content = result.messages[0].content
        parsed = json.loads(content)
        assert parsed["items"] == [[1, 2], [3, 4]]
        assert parsed["meta"] == "info"


# =============================================================================
# ToolOutputFilter — code block safety inside tool output (Phase 5)
# =============================================================================

class TestToolOutputFilterCodeBlockSafety:
    """Fenced code blocks inside tool output stay intact."""

    def test_code_blocks_inside_tool_output(self) -> None:
        """JSON tool output that contains code strings is not corrupted."""
        data = {
            "id": "123",
            "code": "```python\nprint('hello')\n```",
            "result": "ok",
        }
        request = Request(
            messages=[Message(role="tool", content=json.dumps(data))],
            metadata={"_lattice_profile": "tool_output"},
        )
        filt = ToolOutputFilter()
        result = unwrap(filt.process(request, TransformContext()))
        parsed = json.loads(result.messages[0].content)
        # Code block string should be preserved
        assert "```python" in parsed["code"]
        assert "print('hello')" in parsed["code"]
