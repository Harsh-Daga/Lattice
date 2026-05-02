"""Content Profiler / Adaptive Strategy Selector.

Profiles the content of a request and assigns a compression strategy.
This is a **meta-transform** that runs first and configures the behavior
of downstream transforms by setting metadata in the TransformContext.

It does not modify the Request content directly.

**Research basis:**
- Different content types benefit from different compression strategies:
  - Code-heavy → reference substitution + whitespace optimization
  - Table-heavy → format conversion (CSV/TSV)
  - Narrative-long → semantic compression
  - Tool-output-heavy → tool filtering + summarization
  - Mixed → balanced approach
- Adaptive strategy selection can improve compression by 15-25%
  compared to fixed pipelines (informed by LLMLingua-2's segment-based
  compression approach).

**Reversible:** No-op on forward and reverse.

**Performance:** Single-pass content scanning. Target: <0.1ms.

Priority: 1 (runs before all other transforms)
"""

from __future__ import annotations

import enum
import re
from typing import Any

from lattice.core.context import (
    METADATA_KEY_PROTECTED_SPANS,
    METADATA_KEY_RISK_SCORE,
    METADATA_KEY_SCHEDULE,
    METADATA_KEY_SIG,
    METADATA_KEY_SIG_SUMMARY,
    METADATA_KEY_TASK_CLASSIFICATION,
    TransformContext,
)
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.scheduler import decide_schedule
from lattice.core.semantic_graph import SemanticImportanceGraph, SemanticSpan
from lattice.core.task_classifier import classify_task
from lattice.core.transport import Request, Response
from lattice.utils.validation import SemanticRiskScore, compute_risk_score

# =============================================================================
# Content profiles
# =============================================================================


class ContentProfile(enum.Enum):
    """Classification of request content type."""

    CODE_HEAVY = "code_heavy"  # Lots of code blocks, identifiers
    TABLE_HEAVY = "table_heavy"  # JSON arrays, markdown tables
    NARRATIVE_LONG = "narrative_long"  # Long natural language text
    TOOL_OUTPUT = "tool_output"  # Tool/API response JSON
    LOG_OUTPUT = "log_output"  # Timestamped log lines
    DIFF_OUTPUT = "diff_output"  # Unified diff / patch
    STACK_TRACE = "stack_trace"  # Exception stack traces
    GREP_OUTPUT = "grep_output"  # Grep / search results
    FILE_TREE = "file_tree"  # Directory tree listings
    MCP_OUTPUT = "mcp_output"  # MCP tool result structures
    MIXED = "mixed"  # Balanced mix
    SHORT = "short"  # Too short to benefit from compression


# =============================================================================
# ContentProfiler
# =============================================================================


class ContentProfiler(ReversibleSyncTransform):
    """Profile request content and recommend compression strategy.

    Algorithm:
    1. Analyze all messages for content type signals
    2. Compute profile scores for each category
    3. Select dominant profile
    4. Write strategy metadata to TransformContext

    Downstream transforms can read `context.session_state["content_profile"]`
    to adjust their behavior.

    Configuration
    -------------
    - enable_adaptive: Enable profiling. Default: True.
    - short_threshold_tokens: Requests below this are "short" profile.
      Default: 50.
    - code_block_weight: Score weight for code blocks. Default: 3.
    - table_row_weight: Score weight per table-like row. Default: 2.
    - narrative_length_weight: Score weight per 100 tokens of narrative.
      Default: 1.
    """

    name = "content_profiler"
    priority = 1  # Run FIRST, before all other transforms

    def __init__(
        self,
        enable_adaptive: bool = True,
        short_threshold_tokens: int = 50,
        code_block_weight: float = 3.0,
        table_row_weight: float = 2.0,
        narrative_length_weight: float = 1.0,
    ) -> None:
        self.enable_adaptive = enable_adaptive
        self.short_threshold_tokens = short_threshold_tokens
        self.code_block_weight = code_block_weight
        self.table_row_weight = table_row_weight
        self.narrative_length_weight = narrative_length_weight

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Profile request content, compute risk, build SIG, and set strategy."""
        if not self.enable_adaptive:
            return Ok(request)

        profile = self._classify(request)
        strategy = self._select_strategy(profile, request)
        risk_score = self._compute_risk(request)

        # Build SIG — Semantic Importance Graph
        sig = _build_importance_graph(request)
        task = classify_task(request)

        # Build scheduler decision from SIG + RATS + risk
        transform_names = [
            t for t in strategy if isinstance(strategy.get(t), bool) and strategy.get(t) is True
        ]
        schedule = decide_schedule(
            transform_names=list(transform_names),
            task=task,
            risk=risk_score,
            protected_span_count=sig.protected_count,
            total_budget_ms=task.budget_ms,
        )

        # Store in context for downstream transforms
        state = context.get_transform_state(self.name)
        state["profile"] = profile.value
        state["strategy"] = strategy
        state["risk_score"] = risk_score.to_dict()
        state["task_class"] = task.to_dict()
        state["protected_spans"] = sig.protected_span_ids

        context.record_metric(self.name, "profile", profile.value)
        context.record_metric(self.name, "total_tokens", request.token_estimate)
        context.record_metric(self.name, "risk_score", risk_score.total)
        context.record_metric(self.name, "risk_level", risk_score.level)
        context.record_metric(self.name, "sig_total_spans", sig.total_spans)
        context.record_metric(self.name, "sig_protected", sig.protected_count)
        context.record_metric(self.name, "task_class", task.task_class.value)

        # Set canonical metadata keys
        request.metadata["_lattice_profile"] = profile.value
        request.metadata["_lattice_strategy"] = strategy
        request.metadata[METADATA_KEY_RISK_SCORE] = risk_score.to_dict()
        request.metadata[METADATA_KEY_SIG] = sig.to_dict()
        request.metadata[METADATA_KEY_SIG_SUMMARY] = sig.summary()
        request.metadata[METADATA_KEY_PROTECTED_SPANS] = sig.protected_span_ids
        request.metadata[METADATA_KEY_TASK_CLASSIFICATION] = task.to_dict()
        request.metadata[METADATA_KEY_SCHEDULE] = schedule.to_dict()

        return Ok(request)

        profile = self._classify(request)
        strategy = self._select_strategy(profile, request)
        risk_score = self._compute_risk(request)

        # Store in context for downstream transforms
        state = context.get_transform_state(self.name)
        state["profile"] = profile.value
        state["strategy"] = strategy
        state["risk_score"] = risk_score.to_dict()

        context.record_metric(self.name, "profile", profile.value)
        context.record_metric(self.name, "total_tokens", request.token_estimate)
        context.record_metric(self.name, "risk_score", risk_score.total)
        context.record_metric(self.name, "risk_level", risk_score.level)

        # Set per-transform hints in request metadata
        request.metadata["_lattice_profile"] = profile.value
        request.metadata["_lattice_strategy"] = strategy
        request.metadata["_lattice_risk_score"] = risk_score.to_dict()

        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """No-op."""
        return response

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(self, request: Request) -> ContentProfile:
        """Classify the request content type."""
        total_tokens = request.token_estimate

        if total_tokens < self.short_threshold_tokens:
            return ContentProfile.SHORT

        scores: dict[ContentProfile, float] = {
            ContentProfile.CODE_HEAVY: 0.0,
            ContentProfile.TABLE_HEAVY: 0.0,
            ContentProfile.NARRATIVE_LONG: 0.0,
            ContentProfile.TOOL_OUTPUT: 0.0,
            ContentProfile.LOG_OUTPUT: 0.0,
            ContentProfile.DIFF_OUTPUT: 0.0,
            ContentProfile.STACK_TRACE: 0.0,
            ContentProfile.GREP_OUTPUT: 0.0,
            ContentProfile.FILE_TREE: 0.0,
            ContentProfile.MCP_OUTPUT: 0.0,
        }

        all_text = "\n".join(m.content for m in request.messages)

        # Code signals
        code_blocks = len(re.findall(r"```[\w]*\n", all_text))
        inline_code = len(re.findall(r"`[^`]+`", all_text))
        scores[ContentProfile.CODE_HEAVY] += (
            code_blocks * self.code_block_weight + inline_code * 0.5
        )

        # Table signals
        json_arrays = len(re.findall(r"\[\s*\{", all_text))
        md_tables = len(re.findall(r"^\s*\|.*\|\s*$", all_text, re.MULTILINE))
        scores[ContentProfile.TABLE_HEAVY] += (
            json_arrays * self.table_row_weight + md_tables * self.table_row_weight
        )

        # Narrative signals
        non_code_text = re.sub(r"```.*?```", "", all_text, flags=re.DOTALL)
        sentences = len(re.split(r"[.!?]+", non_code_text))
        scores[ContentProfile.NARRATIVE_LONG] += sentences * self.narrative_length_weight

        # Tool output signals
        tool_msgs = sum(1 for m in request.messages if m.role in ("tool", "function"))
        scores[ContentProfile.TOOL_OUTPUT] += tool_msgs * 5.0

        # Log output signals: timestamped lines, severity levels
        log_timestamps = len(re.findall(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", all_text))
        log_levels = len(
            re.findall(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b", all_text)
        )
        scores[ContentProfile.LOG_OUTPUT] += log_timestamps * 2.0 + log_levels * 1.5

        # Diff output signals: unified diff headers, +/- lines, @@ hunks, diff --git
        diff_headers = len(re.findall(r"^(---|\+\+\+) ", all_text, re.MULTILINE))
        diff_lines = len(re.findall(r"^[\+\-]", all_text, re.MULTILINE))
        diff_hunks = len(re.findall(r"^@@ [-+\d,\s]+ @@", all_text, re.MULTILINE))
        diff_git = len(re.findall(r"^diff --git ", all_text, re.MULTILINE))
        scores[ContentProfile.DIFF_OUTPUT] += (
            diff_headers * 3.0 + diff_lines * 0.5 + diff_hunks * 2.0 + diff_git * 3.0
        )

        # Stack trace signals: exception patterns, file:line references, Java-style traces
        trace_exceptions = len(re.findall(r"\b(Exception|Error|Traceback)\b", all_text))
        trace_file_lines = len(re.findall(r"File \".+?\", line \d+", all_text))
        trace_java_style = len(re.findall(r"\bat\s+\S+\s*\([^)]+:\d+\)", all_text))
        scores[ContentProfile.STACK_TRACE] += (
            trace_exceptions * 2.0 + trace_file_lines * 1.5 + trace_java_style * 1.5
        )

        # Grep output signals: filename:line:match or filename:line-column:match pattern
        grep_matches = len(re.findall(r"^.+?:\d+?:.+$", all_text, re.MULTILINE))
        grep_with_column = len(re.findall(r"^.+?:\d+:\d+:.+$", all_text, re.MULTILINE))
        scores[ContentProfile.GREP_OUTPUT] += grep_matches * 1.0 + grep_with_column * 0.5

        # File tree signals: directory indentation, tree branch characters, tree command output
        tree_lines = len(re.findall(r"^[\s│├└├──]*[├└]── ", all_text, re.MULTILINE))
        tree_cmd = len(
            re.findall(r"^[\s│]*\d+\s+directories,\s+\d+\s+files", all_text, re.MULTILINE)
        )
        tree_indent = len(re.findall(r"^\s+[^\s/]+(?:\.\w+)?/?$", all_text, re.MULTILINE))
        scores[ContentProfile.FILE_TREE] += tree_lines * 1.5 + tree_cmd * 2.0 + tree_indent * 0.5

        # MCP output signals: tool result structures with is_error, content, type, tool fields
        mcp_results = len(re.findall(r'"is_error"\s*:\s*(true|false)', all_text))
        mcp_fields = len(re.findall(r'"(content|type|tool)"\s*:\s*"', all_text))
        scores[ContentProfile.MCP_OUTPUT] += mcp_results * 2.0 + mcp_fields * 0.5

        # Normalize by total tokens to avoid bias toward long requests
        if total_tokens > 0:
            for profile in scores:
                scores[profile] /= total_tokens / 100.0

        # Determine dominant profile. If no profile clearly dominates,
        # treat the request as mixed so downstream transforms stay conservative.
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        max_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if max_score < 0.5 or (second_score > 0 and max_score < second_score * 1.35):
            return ContentProfile.MIXED

        dominant = ranked[0][0]
        return dominant

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def _compute_risk(self, request: Request) -> SemanticRiskScore:
        """Compute semantic risk score for the request."""
        return compute_risk_score(request)

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_strategy(self, profile: ContentProfile, _request: Request) -> dict[str, Any]:
        """Select compression parameters based on profile."""
        base: dict[str, Any] = {
            "reference_sub": True,
            "tool_filter": True,
            "prefix_opt": True,
            "output_cleanup": True,
            "format_conversion": True,
            "message_dedup": True,
            "semantic_compress": False,
            "structure_type": profile.value,
        }

        if profile == ContentProfile.SHORT:
            return {
                **base,
                "reference_sub": False,
                "tool_filter": False,
                "format_conversion": False,
                "message_dedup": False,
            }

        if profile == ContentProfile.CODE_HEAVY:
            return {
                **base,
                "semantic_compress": False,
                "format_conversion": False,
                "reference_sub": True,
            }

        if profile == ContentProfile.TABLE_HEAVY:
            return {**base, "format_conversion": True, "semantic_compress": False}

        if profile == ContentProfile.NARRATIVE_LONG:
            return {
                **base,
                "semantic_compress": True,
                "compression_ratio": 0.6,
                "format_conversion": False,
            }

        if profile == ContentProfile.TOOL_OUTPUT:
            return {
                **base,
                "tool_filter": True,
                "semantic_compress": False,
                "format_conversion": True,
            }

        if profile == ContentProfile.LOG_OUTPUT:
            # Logs: dedup repeated lines, keep recent, no semantic compress
            return {
                **base,
                "tool_filter": True,
                "semantic_compress": False,
                "format_conversion": False,
                "message_dedup": True,
                "reference_sub": True,
            }

        if profile == ContentProfile.DIFF_OUTPUT:
            # Diffs: reference substitution for repeated paths, no cleanup
            return {
                **base,
                "reference_sub": True,
                "output_cleanup": False,
                "semantic_compress": False,
                "format_conversion": False,
            }

        if profile == ContentProfile.STACK_TRACE:
            # Stack traces: reference substitution for paths, keep structure
            return {
                **base,
                "reference_sub": True,
                "output_cleanup": False,
                "semantic_compress": False,
                "format_conversion": False,
                "tool_filter": False,
            }

        if profile == ContentProfile.GREP_OUTPUT:
            # Grep: format conversion if tabular, reference sub for paths
            return {
                **base,
                "format_conversion": True,
                "reference_sub": True,
                "semantic_compress": False,
                "output_cleanup": False,
            }

        if profile == ContentProfile.FILE_TREE:
            # File trees: heavy reference substitution, no cleanup
            return {
                **base,
                "reference_sub": True,
                "output_cleanup": False,
                "semantic_compress": False,
                "format_conversion": False,
                "tool_filter": False,
            }

        if profile == ContentProfile.MCP_OUTPUT:
            # MCP: schema-aware tool filter, preserve structure
            return {
                **base,
                "tool_filter": True,
                "semantic_compress": False,
                "format_conversion": True,
                "output_cleanup": False,
            }

        # MIXED
        return base


# =============================================================================
# SIG: Semantic Importance Graph builder
# =============================================================================


def _build_importance_graph(request: Request) -> SemanticImportanceGraph:
    """Build a semantic importance graph from a request.

    Steps:
    1. Segment text into spans by structure boundaries
    2. Extract features per span (frequency, entities, position, etc.)
    3. Compute importance scores
    4. Derive protected spans
    """
    text = "\n".join(msg.content or "" for msg in request.messages)
    if not text.strip():
        return SemanticImportanceGraph(total_spans=0)

    spans = _segment_spans(text)
    _extract_features(spans, text, request)
    _compute_importance(spans)
    _derive_protected(spans)

    importance_values = [s.importance for s in spans]
    avg_importance = sum(importance_values) / len(importance_values) if importance_values else 0.0

    return SemanticImportanceGraph(
        spans=spans,
        total_spans=len(spans),
        protected_count=sum(1 for s in spans if s.protected),
        average_importance=round(avg_importance, 2),
    )


def _segment_spans(text: str) -> list[SemanticSpan]:
    """Split text into spans by structure boundaries.

    Boundaries: code fences, JSON blocks, markdown tables, diff headers,
    log lines, and sentences.
    """
    spans: list[SemanticSpan] = []
    pos = 0

    # Split by code fences first
    parts = re.split(r"(`{3}[\s\S]*?`{3})", text)
    for part in parts:
        if part.startswith("```"):
            spans.append(
                SemanticSpan(
                    span_id=len(spans),
                    text=part,
                    start_char=pos,
                    end_char=pos + len(part),
                    structure_type="code",
                )
            )
        elif part.strip():
            # Split non-code by major boundaries
            sub_spans = _segment_structured(part, pos)
            spans.extend(sub_spans)
        pos += len(part)

    # Deduplicate span IDs
    for i, s in enumerate(spans):
        s.span_id = i

    return spans


def _segment_structured(text: str, offset: int) -> list[SemanticSpan]:
    """Segment non-code text into structured spans."""
    spans: list[SemanticSpan] = []
    pos = 0

    # Split by JSON blocks
    parts = re.split(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", text)
    for part in parts:
        stripped = part.strip()
        if not stripped:
            pos += len(part)
            continue
        if stripped.startswith("{") or stripped.startswith("["):
            spans.append(
                SemanticSpan(
                    span_id=0,
                    text=part,
                    start_char=offset + pos,
                    end_char=offset + pos + len(part),
                    structure_type="json",
                )
            )
        elif stripped.startswith("|"):
            spans.append(
                SemanticSpan(
                    span_id=0,
                    text=part,
                    start_char=offset + pos,
                    end_char=offset + pos + len(part),
                    structure_type="table",
                )
            )
        else:
            # Split by sentence
            sentences = re.split(r"((?<=[.!?])\s+)", stripped)
            for sent in sentences:
                if sent.strip():
                    spans.append(
                        SemanticSpan(
                            span_id=0,
                            text=sent,
                            start_char=offset + pos,
                            end_char=offset + pos + len(sent),
                            structure_type="narrative",
                        )
                    )
                    pos += len(sent)
        pos += len(part)

    return spans


def _extract_features(
    spans: list[SemanticSpan],
    full_text: str,
    request: Request,
) -> None:
    """Extract per-span features: frequency, entities, position, signals."""
    total_spans = len(spans)
    if total_spans == 0:
        return

    for i, span in enumerate(spans):
        # Frequency: count occurrences of this span's text
        span.frequency = max(1.0, full_text.count(span.text) or 1.0)

        # Position weight: first and last spans get a bonus
        if i == 0:
            span.position_weight = 1.0
        elif i == total_spans - 1:
            span.position_weight = 0.8
        elif i < total_spans * 0.2:
            span.position_weight = 0.6
        else:
            span.position_weight = 0.4

        # Entity density: count UUIDs, numbers, URLs in the span
        span_text = span.text
        uuids = len(
            re.findall(
                r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                span_text,
                re.IGNORECASE,
            )
        )
        numbers = len(re.findall(r"\b\d+(?:\.\d+)?\b", span_text))
        urls = len(re.findall(r"https?://[^\s)]+", span_text, re.IGNORECASE))
        span.entity_density = min(
            (uuids * 3 + numbers * 0.5 + urls * 2) / max(len(span_text.split()), 1), 1.0
        )

        # Dependency score: check if this span is referenced later
        if i < total_spans - 1:
            later_text = " ".join(s.text for s in spans[i + 1 :])
            # Simple check: do words from this span appear in later spans
            span_words = set(span.text.lower().split()) - {
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "to",
                "of",
                "in",
                "for",
                "and",
                "or",
            }
            if span_words:
                appearing = sum(1 for w in span_words if w in later_text.lower())
                span.dependency_score = min(appearing / len(span_words), 1.0)

        # Task relevance: does the span contain task-carrying content
        task_indicators = [
            "error",
            "failure",
            "root cause",
            "mitigation",
            "debug",
            "fix",
            "investigate",
            "analyze",
            "compare",
            "trend",
            "conclusion",
        ]
        span.task_relevance = min(
            sum(0.15 for ti in task_indicators if ti in span_text.lower()), 1.0
        )

        # Reasoning signal: explicit reasoning markers
        reasoning_markers = [
            "because",
            "therefore",
            "thus",
            "hence",
            "since",
            "if",
            "then",
            "else",
            "consequently",
        ]
        span.reasoning_signal = any(m in span_text.lower() for m in reasoning_markers)


def _compute_importance(spans: list[SemanticSpan]) -> None:
    """Compute importance score per span.

    Formula: 0.25*frequency + 0.20*dependency + 0.20*entity_density
           + 0.20*task_relevance + 0.15*position
    """
    max_freq = max((s.frequency for s in spans), default=1.0)
    for span in spans:
        freq_norm = span.frequency / max_freq if max_freq > 0 else 0.0
        score = (
            0.25 * freq_norm
            + 0.20 * span.dependency_score
            + 0.20 * span.entity_density
            + 0.20 * span.task_relevance
            + 0.15 * span.position_weight
        )
        # Boost for reasoning signals
        if span.reasoning_signal:
            score *= 1.5
        span.importance = min(round(score * 100, 1), 100.0)


def _derive_protected(spans: list[SemanticSpan]) -> None:
    """Contrastive protection: top-k by importance + hard force-protect rules.

    Instead of an absolute threshold that over-protects, uses:
    1. Top 20% by importance score (minimum 1 span protected)
    2. Force-protect rules for counts, errors, root cause, IDs, reasoning
    3. Compressible flag for boilerplate that passed under the threshold
    """
    if not spans:
        return

    # Step 1: Rank by importance
    ranked = sorted(spans, key=lambda s: s.importance, reverse=True)
    protected_count = max(1, int(0.20 * len(spans)))

    # Step 2: Protect top-k
    for span in ranked[:protected_count]:
        span.protected = True

    # Step 3: Force-protect rules — always protect content containing:
    for span in spans:
        text_lower = span.text.lower()

        # Exact numbers, dates, IDs
        if re.search(r"\b\d+(?:\.\d+)?\b", span.text) and len(re.findall(r"\b\d+(?:\.\d+)?\b", span.text)) >= 2:
            span.protected = True

        # Root cause statements
        if re.search(r"\broot cause\b|\bcaused by\b|\bdue to\b", text_lower):
            span.protected = True

        # Error messages
        if re.search(r"\b(error|exception|failure|crash|timeout|refused|denied)\b", text_lower):
            span.protected = True

        # Counts and distributions
        if re.search(r"\b\d+\s+(errors|failures|warnings|requests|timeouts|attempts)\b", text_lower):
            span.protected = True

        # Stack traces
        if re.search(r"\bat\s+\S+\s*\([^)]+:\d+\)|\bFile\s+\".+?\",\s+line\s+\d+", span.text):
            span.protected = True

        # User's final question (last user message)
        if span.reasoning_signal and span.position_weight >= 0.8:
            span.protected = True

        # Tool calls and IDs
        if re.search(r'"(call_id|tool_call_id|tool_use_id)"|"id"\s*:\s*"call_', span.text):
            span.protected = True

        # Structured output instructions
        if re.search(r"\breturn json\b|\boutput format\b|\btable format\b", text_lower):
            span.protected = True

    # Step 4: Mark compressible spans (boilerplate, metadata, non-diagnostic repetition)
    for span in spans:
        span.compressible = (
            not span.protected
            and span.structure_type in ("narrative", "log_line")
            and not span.reasoning_signal
        )
