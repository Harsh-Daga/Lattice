"""Content profile classification and compression strategy selection."""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any

from lattice.planner.task_classifier import TaskClassification
from lattice.transport.types import Request


class ContentProfile(enum.Enum):
    """Classification of request content type."""

    CODE_HEAVY = "code_heavy"
    TABLE_HEAVY = "table_heavy"
    NARRATIVE_LONG = "narrative_long"
    TOOL_OUTPUT = "tool_output"
    LOG_OUTPUT = "log_output"
    DIFF_OUTPUT = "diff_output"
    STACK_TRACE = "stack_trace"
    GREP_OUTPUT = "grep_output"
    FILE_TREE = "file_tree"
    MCP_OUTPUT = "mcp_output"
    MIXED = "mixed"
    SHORT = "short"


@dataclass(frozen=True)
class ClassifierConfig:
    """Weights and thresholds for content signal scoring."""

    short_threshold_tokens: int = 50
    code_block_weight: float = 3.0
    table_row_weight: float = 2.0
    narrative_length_weight: float = 1.0


def classify_by_signals(
    request: Request,
    config: ClassifierConfig | None = None,
) -> ContentProfile:
    """Classify the request content type from structural signals."""
    cfg = config or ClassifierConfig()
    total_tokens = request.token_estimate

    if total_tokens < cfg.short_threshold_tokens:
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

    code_blocks = len(re.findall(r"```[\w]*\n", all_text))
    inline_code = len(re.findall(r"`[^`]+`", all_text))
    scores[ContentProfile.CODE_HEAVY] += code_blocks * cfg.code_block_weight + inline_code * 0.5

    json_arrays = len(re.findall(r"\[\s*\{", all_text))
    md_tables = len(re.findall(r"^\s*\|.*\|\s*$", all_text, re.MULTILINE))
    scores[ContentProfile.TABLE_HEAVY] += (
        json_arrays * cfg.table_row_weight + md_tables * cfg.table_row_weight
    )

    non_code_text = re.sub(r"```.*?```", "", all_text, flags=re.DOTALL)
    sentences = len(re.split(r"[.!?]+", non_code_text))
    scores[ContentProfile.NARRATIVE_LONG] += sentences * cfg.narrative_length_weight

    tool_msgs = sum(1 for m in request.messages if m.role in ("tool", "function"))
    scores[ContentProfile.TOOL_OUTPUT] += tool_msgs * 5.0

    log_timestamps = len(re.findall(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", all_text))
    log_levels = len(re.findall(r"\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b", all_text))
    scores[ContentProfile.LOG_OUTPUT] += log_timestamps * 2.0 + log_levels * 1.5

    diff_headers = len(re.findall(r"^(---|\+\+\+) ", all_text, re.MULTILINE))
    diff_lines = len(re.findall(r"^[\+\-]", all_text, re.MULTILINE))
    diff_hunks = len(re.findall(r"^@@ [-+\d,\s]+ @@", all_text, re.MULTILINE))
    diff_git = len(re.findall(r"^diff --git ", all_text, re.MULTILINE))
    scores[ContentProfile.DIFF_OUTPUT] += (
        diff_headers * 3.0 + diff_lines * 0.5 + diff_hunks * 2.0 + diff_git * 3.0
    )

    trace_exceptions = len(re.findall(r"\b(Exception|Error|Traceback)\b", all_text))
    trace_file_lines = len(re.findall(r"File \".+?\", line \d+", all_text))
    trace_java_style = len(re.findall(r"\bat\s+\S+\s*\([^)]+:\d+\)", all_text))
    scores[ContentProfile.STACK_TRACE] += (
        trace_exceptions * 2.0 + trace_file_lines * 1.5 + trace_java_style * 1.5
    )

    grep_matches = len(re.findall(r"^.+?:\d+?:.+$", all_text, re.MULTILINE))
    grep_with_column = len(re.findall(r"^.+?:\d+:\d+:.+$", all_text, re.MULTILINE))
    scores[ContentProfile.GREP_OUTPUT] += grep_matches * 1.0 + grep_with_column * 0.5

    tree_lines = len(re.findall(r"^[\s│├└├──]*[├└]── ", all_text, re.MULTILINE))
    tree_cmd = len(re.findall(r"^[\s│]*\d+\s+directories,\s+\d+\s+files", all_text, re.MULTILINE))
    tree_indent = len(re.findall(r"^\s+[^\s/]+(?:\.\w+)?/?$", all_text, re.MULTILINE))
    scores[ContentProfile.FILE_TREE] += tree_lines * 1.5 + tree_cmd * 2.0 + tree_indent * 0.5

    mcp_results = len(re.findall(r'"is_error"\s*:\s*(true|false)', all_text))
    mcp_fields = len(re.findall(r'"(content|type|tool)"\s*:\s*"', all_text))
    scores[ContentProfile.MCP_OUTPUT] += mcp_results * 2.0 + mcp_fields * 0.5

    if total_tokens > 0:
        for profile in scores:
            scores[profile] /= total_tokens / 100.0

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    max_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if max_score < 0.5 or (second_score > 0 and max_score < second_score * 1.35):
        return ContentProfile.MIXED

    return ranked[0][0]


def select_compression_strategy(
    profile: ContentProfile,
    task: TaskClassification | None = None,
) -> dict[str, Any]:
    """Select compression parameters based on profile and task classification."""
    base: dict[str, Any] = {
        "reference_sub": True,
        "tool_filter": True,
        "path_prefix": True,
        "output_cleanup": True,
        "format_conversion": True,
        "message_dedup": True,
        "rate_distortion": False,
        "structure_type": profile.value,
    }

    if task is not None and isinstance(task, TaskClassification) and task.is_conservative:
        base.update(
            {
                "message_dedup": False,
                "rate_distortion": False,
            }
        )

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
            "rate_distortion": False,
            "format_conversion": False,
            "reference_sub": True,
        }

    if profile == ContentProfile.TABLE_HEAVY:
        return {**base, "format_conversion": True, "rate_distortion": False}

    if profile == ContentProfile.NARRATIVE_LONG:
        return {
            **base,
            "rate_distortion": False,
            "compression_ratio": 0.3,
            "format_conversion": False,
        }

    if profile == ContentProfile.TOOL_OUTPUT:
        return {
            **base,
            "tool_filter": True,
            "rate_distortion": False,
            "format_conversion": True,
        }

    if profile == ContentProfile.LOG_OUTPUT:
        return {
            **base,
            "tool_filter": True,
            "rate_distortion": False,
            "format_conversion": False,
            "message_dedup": True,
            "reference_sub": True,
        }

    if profile == ContentProfile.DIFF_OUTPUT:
        return {
            **base,
            "reference_sub": True,
            "output_cleanup": False,
            "rate_distortion": False,
            "format_conversion": False,
        }

    if profile == ContentProfile.STACK_TRACE:
        return {
            **base,
            "reference_sub": True,
            "output_cleanup": False,
            "rate_distortion": False,
            "format_conversion": False,
            "tool_filter": False,
        }

    if profile == ContentProfile.GREP_OUTPUT:
        return {
            **base,
            "format_conversion": True,
            "reference_sub": True,
            "rate_distortion": False,
            "output_cleanup": False,
        }

    if profile == ContentProfile.FILE_TREE:
        return {
            **base,
            "reference_sub": True,
            "output_cleanup": False,
            "rate_distortion": False,
            "format_conversion": False,
            "tool_filter": False,
        }

    if profile == ContentProfile.MCP_OUTPUT:
        return {
            **base,
            "tool_filter": True,
            "rate_distortion": False,
            "format_conversion": True,
            "output_cleanup": False,
        }

    return base


__all__ = [
    "ClassifierConfig",
    "ContentProfile",
    "classify_by_signals",
    "select_compression_strategy",
]
