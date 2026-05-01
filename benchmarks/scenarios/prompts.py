"""Real-world benchmark scenarios for LATTICE SDK end-to-end evaluation.

Each scenario represents a realistic use case that benefits from
LATTICE compression. Scenarios cover:
- Code-heavy conversations
- Tool output processing
- Table/data analysis
- Multi-turn sessions
- Long-context repetition
- Mixed content types
"""

from __future__ import annotations

import json
from typing import Any

# =============================================================================
# Scenario definitions
# =============================================================================

class BenchmarkScenario:
    """A single benchmark scenario with prompt and expected behavior."""

    def __init__(
        self,
        name: str,
        category: str,
        description: str,
        messages: list[dict[str, Any]],
        expect_json: bool = False,
        json_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        complexity: str = "medium",
        expected_tier: str | None = None,
        target_features: list[str] | None = None,
        proof: str = "",
        safe_transforms: list[str] | None = None,
        risky_transforms: list[str] | None = None,
        forbidden_transforms: list[str] | None = None,
        required_answer_properties: list[str] | None = None,
        judge_rubric: str = "",
    ) -> None:
        self.name = name
        self.category = category
        self.description = description
        self.messages = messages
        self.expect_json = expect_json
        self.json_schema = json_schema
        self.tools = tools
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.complexity = complexity
        self.expected_tier = expected_tier
        self.target_features = list(target_features or [])
        self.proof = proof
        self.safe_transforms = list(safe_transforms or [])
        self.risky_transforms = list(risky_transforms or [])
        self.forbidden_transforms = list(forbidden_transforms or [])
        self.required_answer_properties = list(required_answer_properties or [])
        self.judge_rubric = judge_rubric

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "messages": self.messages,
            "expect_json": self.expect_json,
            "tools": self.tools is not None,
            "max_tokens": self.max_tokens,
            "complexity": self.complexity,
            "expected_tier": self.expected_tier,
            "target_features": self.target_features,
            "proof": self.proof,
            "safe_transforms": self.safe_transforms,
            "risky_transforms": self.risky_transforms,
            "forbidden_transforms": self.forbidden_transforms,
            "required_answer_properties": self.required_answer_properties,
            "judge_rubric": self.judge_rubric,
        }


# =============================================================================
# Scenario generators
# =============================================================================

def _gen_uuid_list(count: int = 20) -> str:
    """Generate a list of UUIDs with some duplicates."""
    uuids = [
        "550e8400-e29b-41d4-a716-446655440000",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "6ba7b810-9dad-11d1-80b4-00c04fd430c8",  # dup
        "550e8400-e29b-41d4-a716-446655440000",  # dup
    ]
    # Extend with unique-ish UUIDs
    for i in range(count - len(uuids)):
        uuids.append(f"{i:08x}-1234-5678-9abc-def012345678")
    return "\n".join(f"  [{i}] {u}" for i, u in enumerate(uuids))


def _gen_build_errors(count: int = 60) -> str:
    """Generate realistic build error JSON."""
    errors = []
    error_types = [
        ("Module not found", "error"),
        ("Syntax error", "error"),
        ("Type mismatch", "warning"),
    ]
    for i in range(count):
        msg, severity = error_types[i % len(error_types)]
        errors.append({
            "_internal": {"stack": ["file.py", "line 42"], "timestamp": f"2024-01-{i+1:02d}T00:00:00Z"},
            "message": msg,
            "severity": severity,
            "module": f"module_{i}",
        })
    return json.dumps({"errors": errors}, indent=2)


def _gen_employee_table(count: int = 100) -> str:
    """Generate a large employee table in markdown."""
    rows = [
        "| ID | Name | Department | Salary | Status |",
        "|----|------|------------|--------|--------|",
    ]
    for i in range(count):
        rows.append(f"| {i} | Emp_{i} | Engineering | ${100000+i*1000} | active |")
    return "\n".join(rows)


def _gen_code_review_context() -> str:
    """Generate a realistic code review with repeated patterns."""
    chunks = []
    for i in range(10):
        chunks.append(f"""
### File: src/module_{i}.py
```python
def process_data_{i}(data: list[dict]) -> list[dict]:
    # TODO: optimize this loop
    result = []
    for item in data:
        if item.get("status") == "active":
            result.append({{"id": item["id"], "value": item["value"] * 2}})
    return result
```
**Issue**: Line 4-6 has repeated dictionary access pattern. Consider list comprehension.
""")
    return "\n".join(chunks)


def _gen_multi_turn_session() -> list[dict[str, Any]]:
    """Generate a multi-turn conversation with repetition."""
    system = (
        "You are a senior software engineer with 20 years of experience in Python, "
        "Rust, Go, and systems design. You write clean, maintainable code and provide "
        "concise, accurate answers. You avoid unnecessary verbosity and always consider "
        "performance implications. You are familiar with async/await, type hints, "
        "dataclasses, and modern Python best practices."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Explain list comprehensions in Python."},
        {"role": "assistant", "content": "List comprehensions provide a concise way to create lists..."},
        {"role": "user", "content": "Now show me dict comprehensions."},
        {"role": "assistant", "content": "Dict comprehensions use {k: v for k, v in iterable}..."},
        {"role": "user", "content": "What about set comprehensions?"},
        {"role": "assistant", "content": "Set comprehensions use {x for x in iterable}..."},
        {"role": "user", "content": "Can you explain generator expressions too?"},
        {"role": "assistant", "content": "Generator expressions use (x for x in iterable)..."},
        {"role": "user", "content": "When should I use each?"},
    ]
    return messages


def _gen_api_docs_summary() -> str:
    """Generate API documentation with repeated structure."""
    endpoints = []
    for i in range(20):
        endpoints.append(f"""
## GET /api/v1/resource_{i}

**Parameters:**
- `id` (string, required): Unique identifier
- `limit` (integer, optional): Max results (default: 50)
- `offset` (integer, optional): Pagination offset

**Response:**
```json
{{"id": "{i}", "name": "Resource {i}", "status": "active"}}
```

**Errors:**
- 404: Resource not found
- 429: Rate limit exceeded
- 500: Internal server error
""")
    return "\n".join(endpoints)


def _gen_tool_call_conversation() -> list[dict[str, Any]]:
    """Generate a conversation with tool calls."""
    return [
        {"role": "system", "content": "You have access to weather and calculator tools."},
        {"role": "user", "content": "What's the weather in NYC and SF?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'}},
            {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "SF"}'}},
        ]},
        {"role": "tool", "content": "72°F, sunny", "tool_call_id": "call_1"},
        {"role": "tool", "content": "65°F, foggy", "tool_call_id": "call_2"},
        {"role": "user", "content": "What's the average temperature?"},
    ]


def _gen_cache_prefix_session() -> list[dict[str, Any]]:
    """Generate a stable-prefix session for cache-arbitrage evaluation."""
    system = (
        "You are LATTICE's production evaluator. Preserve stable context, "
        "prefer exact-prefix reuse, and summarize only the deltas."
    )
    docs = "\n".join(
        f"- Design note {i}: keep manifest ordering stable across turns."
        for i in range(12)
    )
    return [
        {"role": "system", "content": system},
        {"role": "assistant", "content": f"Project context:\n{docs}", "metadata": {"is_static_doc": True}},
        {"role": "user", "content": "Given the repeated context, what should the next delta contain?"},
    ]


def _gen_dictionary_repetition() -> list[dict[str, Any]]:
    """Generate text with repeated phrases for dictionary compression."""
    repeated = "The session manifest should remain stable and the session manifest should remain stable. "
    repeated += "The session manifest should remain stable and the session manifest should remain stable."
    return [
        {"role": "system", "content": "Compress repeated language losslessly."},
        {"role": "user", "content": repeated * 4},
    ]


def _gen_context_budget_prompt() -> list[dict[str, Any]]:
    """Generate a prompt that exceeds a token budget and should trigger selection."""
    docs = []
    for i in range(18):
        docs.append(
            f"Document {i}: The subsystem {i} stores request id {i:04d}, trace code {i:04d}-{i+1:04d}, "
            f"and rollout note for module_{i}. The question is which subset is most relevant to the error path."
        )
    return [
        {"role": "system", "content": "Choose only the most relevant documents."},
        {"role": "assistant", "content": "\n".join(docs), "metadata": {"is_static_doc": True}},
        {"role": "user", "content": "Which documents mention the error path and request ids?"},
    ]


def _gen_runtime_pressure_prompt() -> list[dict[str, Any]]:
    """Generate a prompt that should push the runtime classifier upward."""
    reasoning_chain = (
        "Prove the optimization contract is correct. "
        "State the theorem, derive the lemma, explain the deduction, and "
        "summarize the proof obligations before proposing the final mitigation."
    )
    return [
        {"role": "system", "content": "You are debugging a multi-stage production outage with latency regressions."},
        {"role": "user", "content": (
            "Analyze the following stack trace, retry policy, streaming stall, and cache divergence. "
            "Return a concise plan with failure modes, mitigation steps, and a ranked triage list."
        )},
        {"role": "tool", "content": _gen_build_errors(24)},
        {"role": "assistant", "content": _gen_code_review_context()},
        {"role": "assistant", "content": reasoning_chain * 12},
        {"role": "user", "content": reasoning_chain * 8},
    ]


def _gen_grammar_json_table_prompt() -> list[dict[str, Any]]:
    """Generate structured content for grammar compression evaluation."""
    return [
        {"role": "system", "content": "Summarize structured data without losing keys."},
        {"role": "user", "content": (
            f"JSON:\n{json.dumps({'services': [{'name': f'svc_{i}', 'status': 'ok', 'latency_ms': 120 + i} for i in range(20)]}, indent=2)}\n\n"
            f"Table:\n{_gen_employee_table(24)}"
        )},
    ]


def _gen_cleanup_noise_prompt() -> list[dict[str, Any]]:
    """Generate noisy text that should be normalized and trimmed."""
    return [
        {"role": "system", "content": "Clean up the response without changing meaning."},
        {"role": "user", "content": "  Please   summarize this.  \n\n\nKeep the meaning.   Remove   extra   spaces.   "},
        {"role": "assistant", "content": "Sure.   I will    clean it up.  "},
    ]


def _gen_message_dedup_prompt() -> list[dict[str, Any]]:
    """Generate repeated turns for message deduplication."""
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Summarize the incident report."},
        {"role": "assistant", "content": "The incident report shows a retry loop."},
        {"role": "user", "content": "Summarize the incident report."},
        {"role": "assistant", "content": "The incident report shows a retry loop."},
        {"role": "user", "content": "Now explain the root cause once."},
    ]


def _gen_semantic_compress_prompt() -> list[dict[str, Any]]:
    """Generate a long-form narrative that should compress semantically."""
    narrative = (
        "The platform experienced a cascading slowdown after a cache invalidation storm, "
        "which caused repeated recomputation, increased queueing delay, and higher tail latency. "
        "Operators observed that the request path remained logically correct, but the runtime "
        "behavior degraded because the same expensive work was performed multiple times. "
        "The next step is to retain the root cause, the user-visible impact, and the mitigation plan "
        "while trimming incidental repetition, boilerplate, and unsupported side notes."
    )
    return [
        {"role": "system", "content": "Summarize the incident clearly and keep the root cause."},
        {"role": "user", "content": narrative * 6},
    ]


# =============================================================================
# Scenario registry
# =============================================================================

ALL_SCENARIOS: list[BenchmarkScenario] = [
    BenchmarkScenario(
        name="uuid_deduplication",
        category="reference_substitution",
        description="Repeated UUIDs in error logs should be aliased",
        complexity="medium",
        expected_tier="SIMPLE",
        target_features=["reference_sub"],
        proof="UUID-heavy logs should collapse to aliases without changing the failure explanation.",
        messages=[
            {"role": "system", "content": "Debug transaction failures."},
            {"role": "user", "content": f"Transactions failed:\n{_gen_uuid_list(30)}\n\nWhy did the duplicate UUIDs fail?"},
        ],
    ),

    BenchmarkScenario(
        name="tool_output_filtering",
        category="tool_filter",
        description="Strip internal fields from large JSON tool outputs",
        complexity="complex",
        expected_tier="COMPLEX",
        target_features=["tool_filter"],
        proof="Tool JSON should retain schema fields while removing internal-only payload.",
        messages=[
            {"role": "system", "content": "Summarize build errors."},
            {"role": "user", "content": "Why did the build fail?"},
            {"role": "tool", "content": _gen_build_errors(60)},
        ],
        tools=[{"type": "function", "function": {"name": "get_build_errors", "description": "Get build errors"}}],
    ),

    BenchmarkScenario(
        name="table_compression",
        category="format_conversion",
        description="Convert markdown tables to compact CSV/TSV",
        complexity="complex",
        expected_tier="MEDIUM",
        target_features=["format_conversion"],
        proof="Wide markdown tables should convert to a more compact transport format.",
        messages=[
            {"role": "system", "content": "Analyze employee data and summarize salary trends."},
            {"role": "user", "content": f"Here is the employee data:\n{_gen_employee_table(100)}\n\nWhat are the trends?"},
        ],
    ),

    BenchmarkScenario(
        name="prefix_optimization",
        category="prefix_opt",
        description="Deduplicate repeated system prompt across turns",
        complexity="medium",
        expected_tier="MEDIUM",
        target_features=["prefix_opt"],
        proof="Stable prefix material should not be re-sent on every turn.",
        messages=_gen_multi_turn_session(),
    ),

    BenchmarkScenario(
        name="code_review_patterns",
        category="structural_fingerprint",
        description="Detect repeated code review comment patterns",
        complexity="complex",
        expected_tier="MEDIUM",
        target_features=["structural_fingerprint", "reference_sub"],
        proof="Repeated code review structure should be compressed while preserving code blocks.",
        messages=[
            {"role": "system", "content": "Review the following code changes."},
            {"role": "user", "content": f"Please review:\n{_gen_code_review_context()}\n\nSummarize the common issues."},
        ],
    ),

    BenchmarkScenario(
        name="api_docs_summarization",
        category="dictionary_compress",
        description="Remove low-information boilerplate from API docs",
        complexity="medium",
        expected_tier="MEDIUM",
        target_features=["dictionary_compress"],
        proof="Boilerplate docs should collapse through lossless phrase compression.",
        messages=[
            {"role": "system", "content": "Summarize API endpoints."},
            {"role": "user", "content": f"Document these endpoints concisely:\n{_gen_api_docs_summary()}"},
        ],
    ),

    BenchmarkScenario(
        name="tool_call_preservation",
        category="integration",
        description="Ensure tool calls survive compression roundtrip",
        complexity="complex",
        expected_tier="SIMPLE",
        target_features=["tool_filter"],
        proof="Tool-call conversations must preserve call IDs and arguments.",
        messages=_gen_tool_call_conversation(),
        tools=[
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "calculate", "description": "Calculate", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}}},
        ],
    ),

    BenchmarkScenario(
        name="json_response_format",
        category="format_conversion",
        description="Preserve JSON structure while compressing",
        complexity="medium",
        expected_tier="MEDIUM",
        target_features=["format_conversion", "grammar_compress"],
        proof="JSON outputs should remain valid after transport optimizations.",
        messages=[
            {"role": "system", "content": "Return JSON with analysis results."},
            {"role": "user", "content": f"Analyze this data:\n{json.dumps({'users': [{'id': i, 'name': f'User_{i}', 'score': i * 10} for i in range(50)]})}\n\nReturn top 5 users as JSON."},
        ],
        expect_json=True,
        json_schema={"required": ["users"]},
    ),

    BenchmarkScenario(
        name="cache_arbitrage_prefix",
        category="cache_arbitrage",
        description="Stable prefix should be separated from variable query content",
        complexity="medium",
        expected_tier="SIMPLE",
        target_features=["cache_arbitrage"],
        proof="Stable prefix segments should be identified for cache reuse.",
        messages=_gen_cache_prefix_session(),
    ),

    BenchmarkScenario(
        name="dictionary_repetition",
        category="dictionary_compress",
        description="Repeated phrases should trigger dictionary compression",
        complexity="medium",
        expected_tier="SIMPLE",
        target_features=["dictionary_compress"],
        proof="Repeated phrases should compress to a learned dictionary reference.",
        messages=_gen_dictionary_repetition(),
    ),

    BenchmarkScenario(
        name="cleanup_noise",
        category="output_cleanup",
        description="Whitespace and punctuation noise should be cleaned",
        complexity="simple",
        expected_tier="SIMPLE",
        target_features=["output_cleanup"],
        proof="Whitespace cleanup should normalize the prompt without semantic loss.",
        messages=_gen_cleanup_noise_prompt(),
    ),

    BenchmarkScenario(
        name="message_dedup_turns",
        category="message_dedup",
        description="Repeated turns should be deduplicated conservatively",
        complexity="medium",
        expected_tier="SIMPLE",
        target_features=["message_dedup"],
        proof="Duplicate turns should be removed only when the content is exact or near-exact.",
        messages=_gen_message_dedup_prompt(),
    ),

    BenchmarkScenario(
        name="semantic_compress_longform",
        category="semantic_compress",
        description="Long narrative should compress while keeping the core incident",
        complexity="complex",
        expected_tier="MEDIUM",
        target_features=["rate_distortion"],
        proof="Narrative summaries should preserve the cause, impact, and mitigation.",
        messages=_gen_semantic_compress_prompt(),
    ),

    BenchmarkScenario(
        name="context_budget_pressure",
        category="context_selector",
        description="Long context should trigger token-budgeted document selection",
        complexity="complex",
        expected_tier="MEDIUM",
        target_features=["context_selector", "strategy_selector"],
        proof="Budget pressure should force document selection rather than full retention.",
        messages=_gen_context_budget_prompt(),
    ),

    BenchmarkScenario(
        name="runtime_contract_pressure",
        category="runtime_contract",
        description="Heavy reasoning and tool output should score higher on runtime contract",
        complexity="reasoning",
        expected_tier="REASONING",
        target_features=["runtime_contract"],
        proof="High-complexity requests should retain fidelity and a wider transform budget.",
        messages=_gen_runtime_pressure_prompt(),
        tools=[
            {"type": "function", "function": {"name": "get_trace", "description": "Get trace"}},
            {"type": "function", "function": {"name": "get_policy", "description": "Get policy"}},
            {"type": "function", "function": {"name": "get_budget", "description": "Get budget"}},
            {"type": "function", "function": {"name": "get_mitigation", "description": "Get mitigation"}},
        ],
    ),

    BenchmarkScenario(
        name="grammar_json_table",
        category="grammar_compress",
        description="Structured JSON and table rows should trigger grammar compression",
        complexity="complex",
        expected_tier="MEDIUM",
        target_features=["grammar_compress"],
        proof="Structured data should shrink without breaking syntax or tabular semantics.",
        messages=_gen_grammar_json_table_prompt(),
    ),

    BenchmarkScenario(
        name="simple_baseline",
        category="baseline",
        description="Simple prompt with minimal compression expected",
        complexity="simple",
        expected_tier="SIMPLE",
        target_features=[],
        proof="Simple requests should avoid unnecessary optimization work.",
        messages=[
            {"role": "user", "content": "Hello, how are you?"},
        ],
    ),

    BenchmarkScenario(
        name="mixed_realworld",
        category="mixed",
        description="Complex multi-turn with UUIDs, JSON, code, and errors",
        complexity="reasoning",
        expected_tier="SIMPLE",
        target_features=["reference_sub", "format_conversion", "tool_filter"],
        proof="Mixed real-world prompts should show the safe transforms cooperating on one request.",
        messages=[
            {"role": "system", "content": "You are debugging a production issue."},
            {"role": "user", "content": (
                f"Error from transaction 550e8400-e29b-41d4-a716-446655440000:\n"
                f"```json\n{json.dumps({'trace': [{'id': i, 'fn': f'foo_{i}', '_internal': {'ts': 1700000000 + i}} for i in range(30)], 'message': 'Null pointer'})}\n```\n"
                f"Investigate."
            )},
        ],
    ),
]


def get_scenarios(names: list[str] | None = None) -> list[BenchmarkScenario]:
    """Get benchmark scenarios, optionally filtered by name."""
    if names is None:
        return list(ALL_SCENARIOS)
    return [s for s in ALL_SCENARIOS if s.name in names]


def list_scenario_names() -> list[str]:
    """List all available scenario names."""
    return [s.name for s in ALL_SCENARIOS]


# =============================================================================
# Safety expectations per scenario
# =============================================================================

_SCENARIO_SAFETY: dict[str, dict[str, Any]] = {
    "uuid_deduplication": {
        "safe_transforms": ["reference_sub", "prefix_optimizer", "output_cleanup"],
        "risky_transforms": [],
        "forbidden_transforms": ["semantic_compress", "dictionary_compress", "structural_fingerprint"],
        "required_answer_properties": ["explains duplicate UUID failure", "preserves error context"],
        "judge_rubric": "Answer must explain the duplicate UUID failure. UUID references may be compressed but must remain identifiable.",
    },
    "tool_output_filtering": {
        "safe_transforms": ["tool_filter", "format_conversion"],
        "risky_transforms": ["reference_sub"],
        "forbidden_transforms": ["semantic_compress", "structural_fingerprint"],
        "required_answer_properties": ["correct error counts", "removes _internal fields", "preserves severity labels"],
        "judge_rubric": "Answer must correctly count errors (Module not found 20x, Syntax error 20x, Type mismatch 20x). Internal fields stripped.",
    },
    "table_compression": {
        "safe_transforms": ["format_conversion", "prefix_optimizer"],
        "risky_transforms": ["reference_sub"],
        "forbidden_transforms": ["semantic_compress", "dictionary_compress", "structural_fingerprint"],
        "required_answer_properties": ["salary trend analysis", "department homogeneity", "salary range correct"],
        "judge_rubric": "Answer must identify linear salary trend ($1k increments), Engineering-only department, and salary range.",
    },
    "prefix_optimization": {
        "safe_transforms": ["prefix_optimizer"],
        "risky_transforms": ["reference_sub"],
        "forbidden_transforms": ["semantic_compress"],
        "required_answer_properties": ["explains comprehension types correctly"],
        "judge_rubric": "Answer must correctly explain list/dict/set comprehensions. No meaning drift from compressed prompts allowed.",
    },
    "code_review_patterns": {
        "safe_transforms": ["reference_sub", "output_cleanup", "prefix_optimizer"],
        "risky_transforms": ["structural_fingerprint", "dictionary_compress"],
        "forbidden_transforms": ["semantic_compress"],
        "required_answer_properties": ["identifies repeated patterns", "code blocks preserved"],
        "judge_rubric": "Answer must identify repeated code review patterns. Code blocks must remain intact. No placeholder substitution in code.",
    },
    "api_docs_summarization": {
        "safe_transforms": ["reference_sub", "output_cleanup"],
        "risky_transforms": ["dictionary_compress", "grammar_compress"],
        "forbidden_transforms": ["semantic_compress", "structural_fingerprint"],
        "required_answer_properties": ["endpoint count correct", "parameter structure preserved"],
        "judge_rubric": "Boilerplate may be collapsed but endpoint names, parameters, and response structure must remain identifiable.",
    },
    "tool_call_preservation": {
        "safe_transforms": [],
        "risky_transforms": [],
        "forbidden_transforms": ["reference_sub", "semantic_compress", "dictionary_compress", "structural_fingerprint"],
        "required_answer_properties": ["tool calls preserved", "call IDs intact", "weather data correct"],
        "judge_rubric": "Tool calls must survive compression roundtrip. Call IDs and arguments must be preserved exactly.",
    },
    "json_response_format": {
        "safe_transforms": ["format_conversion", "output_cleanup"],
        "risky_transforms": ["reference_sub"],
        "forbidden_transforms": ["semantic_compress", "structural_fingerprint"],
        "required_answer_properties": ["valid JSON", "top 5 users identified", "scores preserved"],
        "judge_rubric": "Output must be valid JSON with correct top 5 users. Scores must match original values.",
    },
    "cache_arbitrage_prefix": {
        "safe_transforms": ["cache_arbitrage", "prefix_optimizer"],
        "risky_transforms": [],
        "forbidden_transforms": ["semantic_compress", "structural_fingerprint"],
        "required_answer_properties": ["stable prefix identified", "preserves context"],
        "judge_rubric": "Stable prefix segments must be correctly identified. Cache-able content should be marked.",
    },
    "dictionary_repetition": {
        "safe_transforms": ["reference_sub", "output_cleanup"],
        "risky_transforms": ["dictionary_compress"],
        "forbidden_transforms": ["semantic_compress", "structural_fingerprint"],
        "required_answer_properties": ["repeated phrase compressed", "semantic meaning preserved"],
        "judge_rubric": "Repeated phrases may be compressed. The core message must remain the same.",
    },
    "cleanup_noise": {
        "safe_transforms": ["output_cleanup", "prefix_optimizer"],
        "risky_transforms": [],
        "forbidden_transforms": ["semantic_compress", "reference_sub", "structural_fingerprint"],
        "required_answer_properties": ["meaning unchanged", "whitespace normalized"],
        "judge_rubric": "Whitespace should be normalized but semantics must not change. The message must mean the same thing.",
    },
    "message_dedup_turns": {
        "safe_transforms": ["message_dedup", "output_cleanup"],
        "risky_transforms": ["reference_sub"],
        "forbidden_transforms": ["semantic_compress", "structural_fingerprint"],
        "required_answer_properties": ["duplicate turns removed", "last message preserved", "root cause explanation intact"],
        "judge_rubric": "Duplicate turns should be removed. The final unique turn (root cause) must be preserved.",
    },
    "semantic_compress_longform": {
        "safe_transforms": ["prefix_optimizer", "output_cleanup"],
        "risky_transforms": ["semantic_compress", "rate_distortion"],
        "forbidden_transforms": ["structural_fingerprint"],
        "required_answer_properties": ["root cause preserved", "impact preserved", "mitigation plan preserved"],
        "judge_rubric": "Narrative may be compressed but must preserve: root cause, user-visible impact, and mitigation plan. Incidental repetition may be trimmed.",
    },
    "context_budget_pressure": {
        "safe_transforms": ["context_selector", "strategy_selector", "prefix_optimizer"],
        "risky_transforms": ["reference_sub"],
        "forbidden_transforms": ["semantic_compress", "structural_fingerprint"],
        "required_answer_properties": ["most relevant documents selected", "error path documents prioritized"],
        "judge_rubric": "Must select documents about error path and request IDs. Full retention of all 18 documents is a failure.",
    },
    "runtime_contract_pressure": {
        "safe_transforms": ["runtime_contract", "reference_sub"],
        "risky_transforms": ["semantic_compress"],
        "forbidden_transforms": ["structural_fingerprint", "dictionary_compress"],
        "required_answer_properties": ["failure modes identified", "mitigation steps listed", "triage ranked"],
        "judge_rubric": "High-complexity reasoning request. Must preserve reasoning chain, failure modes, mitigation, and triage ranking.",
    },
    "grammar_json_table": {
        "safe_transforms": ["format_conversion", "output_cleanup"],
        "risky_transforms": ["grammar_compress", "reference_sub"],
        "forbidden_transforms": ["semantic_compress", "structural_fingerprint"],
        "required_answer_properties": ["JSON keys preserved", "table structure preserved", "values accurate"],
        "judge_rubric": "Structured data may be compressed but JSON keys and table structure must remain. No placeholder substitution in JSON values.",
    },
    "simple_baseline": {
        "safe_transforms": ["output_cleanup"],
        "risky_transforms": [],
        "forbidden_transforms": ["reference_sub", "semantic_compress", "dictionary_compress", "structural_fingerprint", "message_dedup"],
        "required_answer_properties": ["response is conversational"],
        "judge_rubric": "Simple greeting. No compression should be applied beyond basic cleanup. Output must be conversational.",
    },
    "mixed_realworld": {
        "safe_transforms": ["reference_sub", "format_conversion", "tool_filter"],
        "risky_transforms": [],
        "forbidden_transforms": ["semantic_compress", "structural_fingerprint"],
        "required_answer_properties": ["debugging logic correct", "Null pointer identified", "JSON structure preserved"],
        "judge_rubric": "Mixed real-world prompt. Must preserve debugging logic, identify Null pointer issue, and reference the correct transaction UUID.",
    },
}


# Apply safety expectations to scenarios
for scenario in ALL_SCENARIOS:
    safety = _SCENARIO_SAFETY.get(scenario.name, {})
    scenario.safe_transforms = list(safety.get("safe_transforms", []))
    scenario.risky_transforms = list(safety.get("risky_transforms", []))
    scenario.forbidden_transforms = list(safety.get("forbidden_transforms", []))
    scenario.required_answer_properties = list(safety.get("required_answer_properties", []))
    scenario.judge_rubric = str(safety.get("judge_rubric", ""))
