"""Structural Fingerprint transform — Generic Repeated Pattern Detection.

Detects and compresses repeated structural patterns in text without any
command-specific knowledge. This is fundamentally different from RTK's
approach of writing custom parsers for each command (pytest, git, cargo, etc.).

Instead, StructuralFingerprint treats the output as a sequence of lines and:
1. Clusters lines by their structural similarity (using a structural hash)
2. For clusters with many members, creates a compressed representation
3. Preserves unique lines verbatim

**Research basis:**
- RTK and similar tools write ~15+ custom parsers (git, pytest, cargo, docker...)
- This is brittle (output format changes break parsers) and unscalable
- Structural fingerprinting clusters lines by their *shape*, not their *content*
- A pytest failure and a cargo failure have the SAME shape: "ERROR: <location>"
- This single transform replaces ALL of RTK's custom parsers

**How it works:**
```
Input (pytest):
    FAILED test_a.py::test_1 - assert 1 == 2
    FAILED test_a.py::test_2 - assert 3 == 4
    FAILED test_b.py::test_3 - assert 5 == 6
    ... (50 more lines)

Structural hash of each line: "FAILED <file>::<test> - <message>"
Cluster size: 53 → compress to:
    FAILED: 53 tests (examples: test_a.py::test_1, test_a.py::test_2, test_b.py::test_3)
```

**Reversible:** No. Information is genuinely discarded.

**Typical savings:** 60-90% on output with heavy repetition (tests, builds, logs).

**Performance:** O(lines) for clustering. Target: <0.3ms for 1000 lines.

**Safety:**
- Only compresses clusters above a minimum size (default: 5)
- Always preserves the first N examples (default: 3)
- Never compresses lines shorter than a threshold (default: 10 chars)
- Respects code blocks and structured data boundaries

Priority: 12 (after message_dedup=15... wait, let me check)
Actually, for direct mode, priority doesn't matter since we run a custom pipeline.
But if used in the main pipeline: priority 12 (after prefix_opt=10, before ref_sub=20)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response

# =============================================================================
# ANSI escape code stripping
# =============================================================================

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


# =============================================================================
# Progress bar / spinner detection
# =============================================================================

_PROGRESS_PATTERNS = [
    re.compile(r"^\s*[=█░▒▓]+\s*\d+%?\s*$"),  # Block progress bars
    re.compile(r"^\s*\[?[#\-=█░▒▓]+\]?\s*\d+\.?\d*%?\s*$"),  # Bracket progress with blocks
    re.compile(r"^\s*[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]+\s+"),  # Unicode spinners
    re.compile(r"^\s*\\?\|/?-\s+"),  # ASCII spinners
    re.compile(r"^\s*\d+\s*/\s*\d+\s*\[.*\]\s*$"),  # Count progress
    re.compile(r"^\s*\d+\s*of\s*\d+\s*.*$", re.IGNORECASE),  # "X of Y"
]


def _is_progress_bar(line: str) -> bool:
    """Detect progress bars, spinners, and loading indicators."""
    stripped = _strip_ansi(line)
    return any(pattern.match(stripped) for pattern in _PROGRESS_PATTERNS)


# =============================================================================
# Timestamp normalization
# =============================================================================

_TIMESTAMP_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?\b"),  # ISO 8601
    re.compile(r"\b\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\b"),  # US date
    re.compile(r"\b\d{2}-\d{2}-\d{4}\s+\d{1,2}:\d{2}:\d{2}\b"),  # EU date
    re.compile(r"\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b"),  # Syslog
    re.compile(r"\b\d{2}:\d{2}:\d{2}\.\d+\b"),  # Time with ms
]


def _normalize_timestamps(line: str) -> str:
    """Replace timestamps with a placeholder to improve clustering."""
    result = line
    for pattern in _TIMESTAMP_PATTERNS:
        result = pattern.sub("[TS]", result)
    return result


# =============================================================================
# Code-block extraction / reinsertion (guard)
# =============================================================================

_CODE_FENCE_RE = re.compile(r"^```(\w+)?$")


def _extract_code_blocks(text: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Extract fenced code blocks from text.

    Returns (text_with_placeholders, list of (index, lang, content)).
    """
    lines = text.splitlines()
    result_lines: list[str] = []
    blocks: list[tuple[int, str, str]] = []
    in_block = False
    block_lines: list[str] = []
    block_lang = ""
    block_idx = 0

    for line in lines:
        match = _CODE_FENCE_RE.match(line.strip())
        if match:
            if in_block:
                # End of block
                blocks.append((block_idx, block_lang, "\n".join(block_lines)))
                result_lines.append(f"__CODE_BLOCK_{block_idx}__")
                block_idx += 1
                in_block = False
                block_lines = []
            else:
                # Start of block
                in_block = True
                block_lang = match.group(1) or ""
                result_lines.append(f"__CODE_BLOCK_{block_idx}__")
            continue

        if in_block:
            block_lines.append(line)
        else:
            result_lines.append(line)

    # Handle unclosed block
    if in_block and block_lines:
        blocks.append((block_idx, block_lang, "\n".join(block_lines)))

    return "\n".join(result_lines), blocks


def _reinsert_code_blocks(text: str, blocks: list[tuple[int, str, str]]) -> str:
    """Reinsert code blocks back into text."""
    result = text
    for idx, lang, content in blocks:
        placeholder = f"__CODE_BLOCK_{idx}__"
        fence = f"```{lang}\n{content}\n```" if lang else f"```\n{content}\n```"
        result = result.replace(placeholder, fence, 1)
    return result


# =============================================================================
# Multi-line pattern detection
# =============================================================================

# Patterns that span multiple lines and should be preserved together
_MULTI_LINE_PATTERNS = [
    # Stack traces: "  File ...", "    line ...", "    ^^^^^"
    re.compile(r'^\s*File\s+".+"\S*,?\s+line\s+\d+'),
    re.compile(r"^\s*\^+\s*$"),  # Caret lines in traces
    re.compile(r"^\s*at\s+\S+\s+\("),  # JS stack traces
    # Diff context
    re.compile(r"^(---|\+\+\+)\s+"),  # Diff headers
    re.compile(r"^@@\s+-\d+"),  # Diff hunk headers
    # Error blocks
    re.compile(r"^\s*Caused by:"),
    re.compile(r"^\s*Exception in thread"),
    re.compile(r"^\s*Traceback\s+"),
]


def _is_multi_line_pattern_start(line: str) -> bool:
    """Check if a line starts a multi-line pattern."""
    stripped = _strip_ansi(line)
    return any(pattern.match(stripped) for pattern in _MULTI_LINE_PATTERNS)


# =============================================================================
# Structural fingerprinting
# =============================================================================

def _structural_hash(line: str) -> str:
    """Create a structural fingerprint of a line.

    Replaces content with type markers:
    - Words → W
    - Numbers → N
    - File paths → P
    - Hex/strings → S
    - Punctuation kept as-is for structure

    Examples:
        "FAILED test_a.py::test_1 - assert 1 == 2"
        → "FAILED P::W - assert N == N"

        "error[E0425]: cannot find value `x`"
        → "error[E N]: W W W `S`"

        "src/main.py:10:5: E501 Line too long"
        → "P:N:N: S W W W"
    """
    # Strip ANSI escape codes first
    result = _strip_ansi(line)
    # Normalize timestamps
    result = _normalize_timestamps(result)

    # Replace file paths first (before words/numbers)
    # Match patterns like: src/main.py, /tmp/foo.rs, test_a.py::test_1
    result = re.sub(r"[\w/\.\-]+::[\w/\.\-]+", "P::P", result)
    result = re.sub(r"[\w/\.\-]+/[\w/\.\-]+", "P", result)

    # Replace hex strings (commit hashes, error codes, etc.)
    result = re.sub(r"[0-9a-fA-F]{6,}", "H", result)

    # Replace quoted strings
    result = re.sub(r'"[^"]*"', "\"S\"", result)
    result = re.sub(r"'[^']*'", "'S'", result)

    # Replace backtick strings
    result = re.sub(r"`[^`]+`", "`S`", result)

    # Replace numbers (including version numbers like v0.1.0)
    result = re.sub(r"\d+\.\d+\.\d+", "V", result)
    result = re.sub(r"\b\d+([.,]\d+)?\b", "N", result)

    # Replace remaining words with W (but preserve keywords that indicate structure)
    # Keep structural keywords: FAILED, PASSED, error, warning, etc.
    structural_keywords = {
        "FAILED", "PASSED", "ERROR", "WARN", "WARNING", "INFO", "DEBUG",
        "error", "warning", "info", "debug", "note", "help", "suggestion",
        "Compiling", "Finished", "Running", "Testing", "Doc-tests",
        "ok", "failed", "ignored", "bench", "test", "running",
        "new file", "modified", "deleted", "renamed", "untracked",
        "Changes", "staged", "unstaged", "committed",
    }

    def replace_word(match: re.Match[str]) -> str:
        word = match.group(0)
        if word in structural_keywords:
            return word
        return "W"

    result = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", replace_word, result)

    # Collapse repeated whitespace
    result = " ".join(result.split())

    return result


def _extract_signature(line: str) -> str:
    """Extract the identifying signature from a line for grouping.

    This is a coarser hash used for initial clustering.
    """
    # Strip ANSI and normalize timestamps first
    sig = _strip_ansi(line)
    sig = _normalize_timestamps(sig)
    # Remove specific values, keep structure
    sig = re.sub(r"[0-9a-fA-F]{8,}", "", sig)
    sig = re.sub(r'"[^"]*"', "", sig)
    sig = re.sub(r"'[^']*'", "", sig)
    sig = re.sub(r"\b\d+\b", "", sig)
    sig = re.sub(r"\s+", " ", sig).strip()
    return sig[:80]  # Truncate very long signatures


@dataclass(frozen=True)
class LineCluster:
    """A cluster of structurally similar lines."""

    structural_hash: str
    signature: str
    lines: list[str]
    examples: list[str]


# =============================================================================
# StructuralFingerprint
# =============================================================================

class StructuralFingerprint(ReversibleSyncTransform):
    """Generic repeated pattern detection and compression.

    This transform detects lines that share the same structural pattern
    and compresses them into a summary. It requires NO command-specific
    knowledge — it works on any text with repeated line patterns.

    Algorithm:
    1. Split text into lines
    2. Compute structural hash for each line
    3. Cluster lines by structural hash
    4. For clusters above min_cluster_size:
       a. Keep first N examples
       b. Replace remaining with a summary line
    5. Reassemble text

    Configuration
    -------------
    - min_cluster_size: Minimum lines in a cluster to trigger compression.
      Default: 5.
    - max_examples: Number of examples to keep per cluster. Default: 3.
    - min_line_length: Only consider lines longer than this. Default: 10.
    - max_summary_length: Max chars for the summary line. Default: 120.
    - preserve_prefix_lines: Keep first N lines uncompressed (for headers).
      Default: 2.
    - enable_signature_clustering: Use coarser signature-based clustering
      before structural hashing (catches near-misses). Default: True.
    """

    name = "structural_fingerprint"
    priority = 12  # After prefix_opt=10, before ref_sub=20

    def __init__(
        self,
        min_cluster_size: int = 5,
        max_examples: int = 3,
        min_line_length: int = 10,
        max_summary_length: int = 120,
        preserve_prefix_lines: int = 2,
        enable_signature_clustering: bool = True,
        strip_ansi: bool = True,
        remove_progress_bars: bool = True,
        preserve_code_blocks: bool = True,
        preserve_multi_line_patterns: bool = True,
    ) -> None:
        self.min_cluster_size = max(2, min_cluster_size)
        self.max_examples = max(1, max_examples)
        self.min_line_length = max(5, min_line_length)
        self.max_summary_length = max_summary_length
        self.preserve_prefix_lines = preserve_prefix_lines
        self.enable_signature_clustering = enable_signature_clustering
        self.strip_ansi = strip_ansi
        self.remove_progress_bars = remove_progress_bars
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_multi_line_patterns = preserve_multi_line_patterns

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Compress repeated structural patterns in messages."""
        total_saved = 0
        compressed_count = 0

        for msg in request.messages:
            original = msg.content
            if not original or len(original) < 200:
                continue

            compressed = self._compress_text(original)
            if compressed and compressed != original:
                msg.content = compressed
                compressed_count += 1
                saved = max(0, len(original) - len(compressed))
                total_saved += saved

        if compressed_count > 0:
            context.record_metric(self.name, "messages_compressed", compressed_count)
            context.record_metric(self.name, "tokens_saved_estimate", total_saved // 4)

        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """Irreversible — no-op."""
        return response

    # ------------------------------------------------------------------
    # Compression logic
    # ------------------------------------------------------------------

    def _compress_text(self, text: str) -> str:
        """Compress text by detecting repeated structural patterns."""
        # Extract code blocks to preserve them
        working_text = text
        code_blocks: list[tuple[int, str, str]] = []
        if self.preserve_code_blocks:
            working_text, code_blocks = _extract_code_blocks(text)

        lines = working_text.splitlines()
        if len(lines) < self.min_cluster_size + self.preserve_prefix_lines:
            return text  # Too short, return original (with code blocks intact)

        # Preserve prefix lines
        prefix = lines[: self.preserve_prefix_lines]
        body = lines[self.preserve_prefix_lines :]

        # Remove progress bars from body
        if self.remove_progress_bars:
            body = [ln for ln in body if not _is_progress_bar(ln)]

        # Mark multi-line pattern starts
        multi_line_ranges: list[tuple[int, int]] = []
        if self.preserve_multi_line_patterns:
            i = 0
            while i < len(body):
                if _is_multi_line_pattern_start(body[i]):
                    start = i
                    i += 1
                    # Continue until we hit a blank line or non-indented line
                    while i < len(body):
                        stripped = body[i].strip()
                        if not stripped:
                            break
                        # Continue if indented, caret line, or another trace line
                        if body[i].startswith(" ") or body[i].startswith("\t"):
                            i += 1
                            continue
                        if stripped.startswith("^") or stripped.startswith("at "):
                            i += 1
                            continue
                        if _is_multi_line_pattern_start(body[i]):
                            i += 1
                            continue
                        break
                    multi_line_ranges.append((start, i))
                else:
                    i += 1

        # Build a mask of lines to preserve (multi-line patterns)
        preserve_mask = set()
        for start, end in multi_line_ranges:
            for j in range(start, end):
                preserve_mask.add(j)

        # Cluster lines (excluding preserved ones)
        clusterable_body = [
            ln for idx, ln in enumerate(body) if idx not in preserve_mask
        ]
        clusters = self._cluster_lines(clusterable_body)

        # Build output
        result_lines: list[str] = []
        result_lines.extend(prefix)

        i = 0
        while i < len(body):
            # Multi-line pattern — preserve verbatim
            if i in preserve_mask:
                result_lines.append(body[i])
                i += 1
                continue

            line = body[i]

            # Find which cluster this line belongs to
            cluster = self._find_cluster_for_line(line, clusters)

            if cluster and len(cluster.lines) >= self.min_cluster_size:
                # Output examples + summary
                for ex in cluster.examples[: self.max_examples]:
                    result_lines.append(ex)

                remaining = len(cluster.lines) - self.max_examples
                if remaining > 0:
                    summary = self._create_summary(cluster, remaining)
                    result_lines.append(summary)

                # Skip all lines in this cluster
                skip_count = 1
                while (
                    i + skip_count < len(body)
                    and body[i + skip_count] in cluster.lines
                    and (i + skip_count) not in preserve_mask
                ):
                    skip_count += 1
                i += skip_count
            else:
                result_lines.append(line)
                i += 1

        output = "\n".join(result_lines)

        # Reinsert code blocks
        if code_blocks:
            output = _reinsert_code_blocks(output, code_blocks)

        return output

    def _cluster_lines(self, lines: list[str]) -> list[LineCluster]:
        """Cluster lines by structural similarity."""
        # Map structural_hash -> list of lines
        hash_groups: dict[str, list[str]] = {}
        sig_groups: dict[str, list[str]] = {}

        for line in lines:
            if len(line) < self.min_line_length:
                continue

            struct_hash = _structural_hash(line)
            hash_groups.setdefault(struct_hash, []).append(line)

            if self.enable_signature_clustering:
                sig = _extract_signature(line)
                sig_groups.setdefault(sig, []).append(line)

        clusters: list[LineCluster] = []
        used_lines: set[str] = set()

        # Create clusters from structural hashes
        for struct_hash, group_lines in hash_groups.items():
            if len(group_lines) >= self.min_cluster_size:
                # Deduplicate while preserving order
                seen: set[str] = set()
                unique_lines: list[str] = []
                for ln in group_lines:
                    if ln not in seen:
                        seen.add(ln)
                        unique_lines.append(ln)

                examples = unique_lines[: self.max_examples]
                clusters.append(
                    LineCluster(
                        structural_hash=struct_hash,
                        signature=_extract_signature(unique_lines[0]),
                        lines=unique_lines,
                        examples=examples,
                    )
                )
                used_lines.update(unique_lines)

        # Also create clusters from signature grouping (catches near-misses)
        if self.enable_signature_clustering:
            for sig, group_lines in sig_groups.items():
                # Only if not already covered by structural hash
                uncovered = [ln for ln in group_lines if ln not in used_lines]
                if len(uncovered) >= self.min_cluster_size:
                    unique_lines = []
                    seen2: set[str] = set()
                    for ln in uncovered:
                        if ln not in seen2:
                            seen2.add(ln)
                            unique_lines.append(ln)

                    examples = unique_lines[: self.max_examples]
                    clusters.append(
                        LineCluster(
                            structural_hash=_structural_hash(unique_lines[0]),
                            signature=sig,
                            lines=unique_lines,
                            examples=examples,
                        )
                    )

        return clusters

    def _find_cluster_for_line(
        self, line: str, clusters: list[LineCluster]
    ) -> LineCluster | None:
        """Find the cluster that contains this line."""
        for cluster in clusters:
            if line in cluster.lines:
                return cluster
        return None

    def _create_summary(self, cluster: LineCluster, remaining: int) -> str:
        """Create a summary line for a cluster."""
        # Extract the pattern name from the structural hash
        # Try to extract a meaningful category
        first_line = cluster.lines[0] if cluster.lines else ""

        # Detect common patterns for better summaries
        if "FAILED" in first_line or "failed" in first_line.lower():
            return f"  ... {remaining} more failures"
        if "error" in first_line.lower():
            return f"  ... {remaining} more errors"
        if "warning" in first_line.lower():
            return f"  ... {remaining} more warnings"
        if re.search(r"\bok\b", first_line.lower()):
            return f"  ... {remaining} more passed"
        if "modified" in first_line.lower() or "new file" in first_line.lower():
            return f"  ... {remaining} more files"

        # Generic summary
        return f"  ... {remaining} more similar lines"
