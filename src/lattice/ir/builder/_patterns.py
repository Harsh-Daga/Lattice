"""IR builder regex and keyword tables."""

from __future__ import annotations

import re

_INDENT = re.compile(r"^\s+")
_FENCE_START = re.compile(r"^```(\w+)?$")
_JSON_LINE = re.compile(r"^\s*[\[\{]")
_MD_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_MD_TABLE_SEP = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")
_LOG_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
_LOG_LEVEL = re.compile(r"\b(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\b")
_TRACEBACK_START = re.compile(r"^Traceback\s*\(", re.MULTILINE)
_PYTHON_FRAME = re.compile(r'^\s+File\s+"([^"]+)",\s+line\s+(\d+)', re.MULTILINE)
_JAVA_FRAME = re.compile(r"\bat\s+(\S+)\s*\(([^)]+):(\d+)\)")
_EXCEPTION_LINE = re.compile(r"^\w+(?:Error|Exception|Warning|Fault)(?::\s*.*)?$", re.MULTILINE)
_DIFF_HDR = re.compile(r"^(---|\+\+\+|diff\s+--git|index\s+\w+)", re.MULTILINE)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_URL_RE = re.compile(r"https?://[^\s)>]+", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:/[\w\.\-]+)+")
_ERROR_KEYWORDS = frozenset(
    {
        "error",
        "exception",
        "failure",
        "fail",
        "crash",
        "timeout",
        "refused",
        "denied",
        "abort",
        "panic",
        "fatal",
        "critical",
        "segfault",
        "oom",
    }
)
_ROOT_CAUSE_KEYWORDS = frozenset(
    {
        "root cause",
        "the cause was",
        "the reason is",
        "determined that",
        "because",
        "therefore",
        "consequently",
        "due to",
        "caused by",
        "triggered by",
        "resulting in",
        "leading to",
    }
)
_TASK_KEYWORDS = frozenset(
    {
        "analyze",
        "debug",
        "fix",
        "investigate",
        "explain",
        "compare",
        "optimize",
        "refactor",
        "implement",
        "review",
        "test",
        "deploy",
        "configure",
        "migrate",
        "upgrade",
        "resolve",
    }
)
_CONSTRAINT_KEYWORDS = frozenset(
    {
        "must",
        "required",
        "mandatory",
        "essential",
        "critical",
        "shall",
        "should not",
        "must not",
        "cannot",
        "do not",
        "never",
        "always",
        "ensure",
        "guarantee",
        "preserve",
        "keep",
    }
)
_FORMAT_KEYWORDS = frozenset(
    {
        "json",
        "yaml",
        "csv",
        "markdown",
        "table",
        "code block",
        "output format",
        "return format",
        "respond in",
    }
)
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "about",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "now",
        "also",
        "not",
    }
)

