"""Pre-compiled regex patterns for transforms.

All patterns are compiled at module import time for maximum performance.
There is no runtime compilation in the hot path.

Patterns are named in UPPER_SNAKE_CASE to indicate they are constants.
"""

import re

# =============================================================================
# Reference Substitution Patterns
# =============================================================================

# UUID: 550e8400-e29b-41d4-a716-446655440000
UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

# SHA-256 / MD5 hashes: 32-64 hex chars
HEX_PATTERN = re.compile(
    r"\b[0-9a-f]{32,64}\b",
    re.IGNORECASE,
)

# URLs (simplified, catches most common forms)
# Matches http://anything and https://anything up to whitespace
URL_PATTERN = re.compile(r"https?://[^\s\)\]\'\"]+", re.IGNORECASE)

# Long identifiers (>30 chars of alphanumeric + underscore)
LONG_IDENTIFIER_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{29,}\b")

# =============================================================================
# Tool Output Filter Patterns
# =============================================================================

# Default fields to remove from tool output JSON
DEFAULT_BLACKLIST = frozenset({
    "created_at",
    "updated_at",
    "metadata",
    "logs",
    "debug",
    "_internal",
    "raw",
    "__dict__",
    "_cached",
    "_version",
    "_schema",
})

# =============================================================================
# Output Cleanup Patterns
# =============================================================================

# (pattern, replacement) tuples
# Patterns are ordered from most specific to most general
DEFAULT_CLEANUP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "Sure! I'd be happy to help..." → ""
    (re.compile(
        r"^\s*Sure!?\s*(?:I'd?\s+be\s+happy\s+to\s+(?:help|assist)|I\s+can\s+help)[.!?]*\s*",
        re.IGNORECASE,
    ), ""),
    # "Here is the answer:" → ""
    (re.compile(
        r"^\s*Here\s+(?:is|are)\s+(?:the\s+)?(?:answer|response|result|what\s+you\s+(?:requested|asked\s+for))[:\.]?\s*",
        re.IGNORECASE,
    ), ""),
    # "Based on the context provided," → ""
    (re.compile(
        r"^\s*(?:Based\s+on\s+(?:the\s+)?(?:context|information)\s+(?:provided|given)[,]?\s*)",
        re.IGNORECASE,
    ), ""),
    # Trailing: "Let me know if you need anything else"
    (re.compile(
        r"[\s\n]*Let\s+me\s+know\s+if\s+(?:(?:there's|there\s+is)\s+)?(?:anything\s+else|you\s+(?:need|have|want)\s+(?:any\s+)?(?:questions?|further\s+questions?))[.!?]*\s*$",
        re.IGNORECASE,
    ), ""),
    # Trailing: "Feel free to ask..."
    (re.compile(
        r"[\s\n]*Feel\s+free\s+to\s+(?:ask|reach\s+out)[.!?]*\s*$",
        re.IGNORECASE,
    ), ""),
    # Trailing: "I hope this helps"
    (re.compile(
        r"[\s\n]*I\s+hope\s+(?:this|that)\s+(?:helps?|is\s+helpful)[.!?]*\s*$",
        re.IGNORECASE,
    ), ""),
    # Trailing: "Is there anything else I can help with?"
    (re.compile(
        r"[\s\n]*(?:Is\s+there\s+)?(?:Anything\s+else\s+I\s+can\s+(?:help|assist)\s+(?:you|with)?[?]?)[.!?]*\s*$",
        re.IGNORECASE,
    ), ""),
    # Trailing: "If you have any questions..."
    (re.compile(
        r"[\s\n]*(?:If\s+you\s+have\s+(?:any|more)\s+(?:questions|concerns)|Don't\s+hesitate\s+to\s+ask)[.!?]*\s*$",
        re.IGNORECASE,
    ), ""),
]
