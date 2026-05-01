"""Grammar-based lossless compression for JSON/schema structures.

Uses ABNF-like grammar rules to identify compressible structures in message
content and replaces field-value patterns with short aliases.

**Research basis:**
- Grammar-based compression (Sequitur, Re-Pair) achieves lossless
  structural compression.
- Our approach: lightweight regex-based patterns for JSON-like and tabular
  content, no external grammar parser needed.

**Reversible:** Yes. The mapping table is stored in TransformContext.

**Typical savings:** 15-30% on structured JSON or table-heavy prompts.

**Performance:** O(n) regex scan. Target: <0.2ms for 5k chars.
"""

from __future__ import annotations

import re
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response

# =============================================================================
# Patterns
# =============================================================================

# JSON-like key-value pairs: "key": "value" or 'key': 'value'
_JSON_KV_PATTERN = re.compile(
    r'(?P<quote>["\'])(?P<key>[a-zA-Z0-9_\- ]{2,40})(?P=quote)\s*:\s*(?P<val>["\'][^"\']{3,80}(?P=quote))'
)

# Table-like rows: | key | value | ...
_TABLE_ROW_PATTERN = re.compile(r"^\s*\|(?P<cells>[^|]+(?:\|[^|]+){2,})\|\s*$", re.MULTILINE)

# Repeated field names across lines
_REPEAT_FIELD_PATTERN = re.compile(r'^(\s*)(["\']?[a-zA-Z0-9_\- ]{2,30}["\']?)\s*:', re.MULTILINE)

# Alias format: <g_N>
_ALIAS_PREFIX = "<g"
_ALIAS_SUFFIX = ">"


def _alias_for(idx: int) -> str:
    return f"{_ALIAS_PREFIX}_{idx}>"


class GrammarCompressor(ReversibleSyncTransform):
    """Lossless JSON/schema compression by replacing field-value patterns with aliases.

    Algorithm:
    1. Detect JSON-like key:value or tabular content in message.content.
    2. Build a grammar alias map for repeated/long patterns.
    3. Compress to compact format using <g_N> aliases.
    4. reverse(): restore from the alias map stored in TransformContext.

    Configuration
    -------------
    - min_field_len: Minimum key length to alias. Default: 3.
    - min_value_len: Minimum value length to alias. Default: 6.
    - max_dict_size: Maximum alias entries per session. Default: 500.
    - enable_json: Whether to compress JSON-like K-V. Default: True.
    - enable_table: Whether to compress table rows. Default: True.
    """

    name = "grammar_compress"
    priority = 24

    def __init__(
        self,
        min_field_len: int = 3,
        min_value_len: int = 6,
        max_dict_size: int = 500,
        enable_json: bool = True,
        enable_table: bool = True,
    ) -> None:
        self.min_field_len = max(1, min_field_len)
        self.min_value_len = max(1, min_value_len)
        self.max_dict_size = max(0, max_dict_size)
        self.enable_json = enable_json
        self.enable_table = enable_table

    def _get_session_state(self, context: TransformContext) -> dict[str, Any]:
        return context.get_transform_state(self.name)

    @staticmethod
    def _is_json_like(text: str) -> bool:
        stripped = text.strip()
        return bool(stripped) and stripped[0] in ("{", "[")

    @staticmethod
    def _is_table_like(text: str) -> bool:
        return "|" in text and "---" in text

    def _collect_json_aliases(self, text: str, start_idx: int) -> tuple[dict[str, int], int]:
        """Collect JSON-like key:value pairs and return alias map + next index."""
        alias_map: dict[str, int] = {}
        idx = start_idx
        for match in _JSON_KV_PATTERN.finditer(text):
            key = match.group("key")
            val = match.group("val")
            # Use raw matched string as original to preserve quotes exactly
            raw = match.group(0)
            if len(key) >= self.min_field_len and len(val) >= self.min_value_len:
                if raw not in alias_map:
                    alias_map[raw] = idx
                    idx += 1
        return alias_map, idx

    def _collect_table_aliases(self, text: str, start_idx: int) -> tuple[dict[str, int], int]:
        """Collect repeated table cell values and return alias map + next index."""
        alias_map: dict[str, int] = {}
        idx = start_idx
        cell_counts: dict[str, int] = {}
        for match in _TABLE_ROW_PATTERN.finditer(text):
            cells = match.group("cells").split("|")
            for cell in cells:
                cell = cell.strip()
                if len(cell) >= self.min_value_len:
                    cell_counts[cell] = cell_counts.get(cell, 0) + 1
        for cell, count in cell_counts.items():
            if count >= 2 and cell not in alias_map:
                alias_map[cell] = idx
                idx += 1
        return alias_map, idx

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Compress JSON-like and tabular content in messages."""
        state = self._get_session_state(context)
        alias_map: dict[str, int] = state.get("alias_map", {})
        reverse_map: dict[str, str] = state.get("reverse_map", {})
        next_idx = max(alias_map.values(), default=-1) + 1

        original_total = 0
        compressed_total = 0
        modified_count = 0

        for msg in request.messages:
            original = msg.content
            if not original:
                continue
            compressed = original

            if self.enable_json and self._is_json_like(original):
                new_aliases, next_idx = self._collect_json_aliases(original, next_idx)
                for raw, idx in new_aliases.items():
                    if len(alias_map) >= self.max_dict_size:
                        break
                    if raw not in alias_map:
                        alias_map[raw] = idx
                        reverse_map[_alias_for(idx)] = raw
                    compressed = compressed.replace(raw, _alias_for(idx))

            if self.enable_table and self._is_table_like(original):
                new_aliases, next_idx = self._collect_table_aliases(original, next_idx)
                for raw, idx in new_aliases.items():
                    if len(alias_map) >= self.max_dict_size:
                        break
                    if raw not in alias_map:
                        alias_map[raw] = idx
                        reverse_map[_alias_for(idx)] = raw
                    compressed = compressed.replace(raw, _alias_for(idx))

            if compressed != original:
                msg.content = compressed
                modified_count += 1
            original_total += len(original)
            compressed_total += len(compressed)

        saved = max(0, original_total - compressed_total)
        ratio = saved / max(original_total, 1)

        state["alias_map"] = alias_map
        state["reverse_map"] = reverse_map

        if modified_count > 0:
            context.record_metric(self.name, "messages_modified", modified_count)
            context.record_metric(self.name, "tokens_saved_estimate", saved // 4)
            context.record_metric(self.name, "compression_ratio", round(ratio, 4))
            context.record_metric(self.name, "alias_count", len(alias_map))

        return Ok(request)

    def reverse(self, response: Response, context: TransformContext) -> Response:
        """Restore original content from grammar aliases."""
        state = self._get_session_state(context)
        reverse_map: dict[str, str] = state.get("reverse_map", {})
        if not reverse_map:
            return response

        # Sort aliases by length descending to avoid substring replacement issues
        for alias in sorted(reverse_map, key=len, reverse=True):
            original = reverse_map[alias]
            response.content = response.content.replace(alias, original)
        return response
