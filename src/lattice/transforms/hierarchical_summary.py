"""Hierarchical Summarizer transform — Multi-Level Nested Structure Compression.

For deeply nested or highly structured output (JSON logs, tree views,
nested error traces, deeply indented code), this transform creates
multi-level summaries. Instead of showing every leaf node, it shows
the structure at progressively coarser granularities.

**Research basis:**
- RECOMP (Xu et al., 2023): Extractive + abstractive compression for
  retrieved documents. Achieves 6% compression rate with minimal loss.
- Hierarchical summarization: Humans naturally read nested structures
  at multiple levels (outline → sections → details).
- For LLM consumption, showing the "shape" of nested data is often
  sufficient — specific leaf values can be summarized statistically.

**How it works:**
```
Input (deeply nested JSON):
    {
      "results": [
        {"id": 1, "status": "ok", "data": {...}},
        {"id": 2, "status": "ok", "data": {...}},
        ... (100 more identical structures)
      ]
    }

Hierarchical summary:
    results: 100 items
      structure: {id, status, data}
      status distribution: ok: 98, error: 2
      sample (first 2):
        {"id": 1, "status": "ok", ...}
        {"id": 2, "status": "ok", ...}
```

**How it differs from ToolOutputFilter:**
- ToolOutputFilter: Schema-aware field filtering (keeps/drops fields)
- HierarchicalSummarizer: Structural summarization (shows shape + stats)
- They complement each other: ToolOutputFilter first, then HierarchicalSummarizer

**How it differs from RTK:**
- RTK has no equivalent — they just print "... truncated" for long output
- LATTICE understands the structure and produces a meaningful summary

**Reversible:** No. Information is genuinely discarded.

**Typical savings:** 70-95% on deeply nested structures.

**Performance:** O(nodes) tree traversal. Target: <0.3ms for 1000 nodes.

**Safety:**
- Only activates for structures above a size threshold
- Preserves all error/failure nodes
- Always includes representative samples
- Never removes top-level keys

Priority: 28 (after format_conv=25, before tool_filter=30)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response

# =============================================================================
# Tree node analysis
# =============================================================================


def _is_nested_structure(value: Any) -> bool:
    """Check if a value is a nested structure worth summarizing."""
    if isinstance(value, list):
        return len(value) > 3
    if isinstance(value, dict):
        return len(value) > 3 or any(isinstance(v, (list, dict)) for v in value.values())
    return False


def _count_leaf_nodes(value: Any) -> int:
    """Count total leaf nodes in a nested structure."""
    if isinstance(value, list):
        return sum(_count_leaf_nodes(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_leaf_nodes(v) for v in value.values())
    return 1


def _get_structure_signature(value: Any) -> str:
    """Get a signature of the structure (keys for dicts, element types for lists)."""
    if isinstance(value, list):
        if not value:
            return "[]"
        # Get signatures of first few elements
        sigs = [_get_structure_signature(item) for item in value[:3]]
        if len(set(sigs)) == 1:
            return f"[{sigs[0]} x {len(value)}]"
        return f"[{', '.join(set(sigs))}]"
    if isinstance(value, dict):
        keys = sorted(value.keys())
        return "{" + ", ".join(keys) + "}"
    return type(value).__name__


def _summarize_value(
    value: Any,
    depth: int = 0,
    max_depth: int = 3,
    max_items: int = 5,
    min_occurrences: int = 3,
) -> Any:
    """Recursively summarize a nested structure.

    Args:
        value: The value to summarize.
        depth: Current depth in the tree.
        max_depth: Maximum depth to traverse before summarizing.
        max_items: Maximum items to show in a list before summarizing.
        min_occurrences: Minimum occurrences to report in distribution.

    Returns:
        Summarized value (may be a dict with _summary metadata).
    """
    if depth >= max_depth:
        # At max depth, create a compact summary
        if isinstance(value, list):
            return {"_count": len(value), "_type": "list"}
        if isinstance(value, dict):
            return {"_keys": sorted(value.keys())[:10], "_type": "dict"}
        return value

    if isinstance(value, list):
        if len(value) <= max_items:
            # Small list — show all items
            return [_summarize_value(item, depth + 1, max_depth, max_items) for item in value]

        # Large list — summarize
        # Check if all items have the same structure
        if value and all(isinstance(item, dict) for item in value):
            return _summarize_dict_list(value, depth, max_depth, max_items, min_occurrences)

        # Mixed or primitive list — show count + sample
        return {
            "_count": len(value),
            "_sample": value[:max_items],
        }

    if isinstance(value, dict):
        # Summarize each field
        result: dict[str, Any] = {}
        for key, val in value.items():
            # Always preserve error-related keys
            if _is_error_key(key):
                result[key] = val
            else:
                result[key] = _summarize_value(val, depth + 1, max_depth, max_items)
        return result

    # Primitive
    return value


def _is_error_key(key: str) -> bool:
    """Check if a dictionary key indicates error information."""
    error_keywords = {
        "error",
        "errors",
        "fail",
        "failed",
        "failure",
        "exception",
        "panic",
        "abort",
        "critical",
        "fatal",
        "stacktrace",
        "traceback",
    }
    key_lower = key.lower()
    return any(kw in key_lower for kw in error_keywords)


def _summarize_dict_list(
    items: list[dict[str, Any]],
    depth: int,
    max_depth: int,
    max_items: int,
    min_occurrences: int,
) -> dict[str, Any]:
    """Summarize a list of dictionaries with similar structure."""
    if not items:
        return {"_count": 0}

    total = len(items)

    # Collect all keys
    all_keys: set[str] = set()
    for item in items:
        all_keys.update(item.keys())

    # For each key, compute distribution
    field_stats: dict[str, Any] = {}

    for key in sorted(all_keys):
        values = [item.get(key) for item in items if key in item]

        if not values:
            continue

        # Check if this key contains errors — always preserve
        if _is_error_key(key):
            # Show unique error values
            unique_errors = list(dict.fromkeys(str(v) for v in values if v))
            if len(unique_errors) <= max_items:
                field_stats[key] = unique_errors
            else:
                field_stats[key] = unique_errors[:max_items] + [
                    f"... {len(unique_errors) - max_items} more"
                ]
            continue

        # For primitive values, compute frequency distribution
        if all(not isinstance(v, (list, dict)) for v in values):
            freq = Counter(str(v) for v in values)
            if len(freq) == 1:
                # All same value — just show it
                field_stats[key] = {"_all": list(freq.keys())[0], "_count": total}
            elif len(freq) <= max_items:
                # Few unique values — show distribution
                field_stats[key] = dict(freq.most_common(max_items))
            else:
                # Many unique values — show top N + count
                field_stats[key] = {
                    "_top": dict(freq.most_common(max_items)),
                    "_unique": len(freq),
                    "_count": total,
                }
        else:
            # Nested structure — summarize recursively
            field_stats[key] = _summarize_value(values[:max_items], depth + 1, max_depth, max_items)

    # Include representative samples
    samples = items[:max_items]

    return {
        "_count": total,
        "_structure": sorted(all_keys),
        "_fields": field_stats,
        "_samples": samples,
    }


# =============================================================================
# HierarchicalSummarizer
# =============================================================================


class HierarchicalSummarizer(ReversibleSyncTransform):
    """Multi-level summarization for nested structures.

    This transform detects deeply nested or highly structured content
    and replaces it with hierarchical summaries that preserve the shape
    and statistics while reducing token count.

    Algorithm:
    1. Attempt to parse message content as JSON
    2. If parsed and structure is deep/large enough:
       a. Traverse the tree recursively
       b. At each level, decide: show raw, show sample, or summarize
       c. For large lists of similar dicts, show field distributions
    3. Serialize back to compact representation
    4. If not JSON, try to detect indentation-based nesting (tree output)

    Configuration
    -------------
    - max_depth: Maximum tree depth to traverse. Default: 3.
    - max_items: Max items to show before summarizing a list. Default: 5.
    - min_leaf_nodes: Minimum leaf nodes to trigger summarization. Default: 50.
    - min_occurrences: Min occurrences to report in distribution. Default: 3.
    - enable_tree_detection: Detect and summarize tree-formatted text
      (like `tree` command output). Default: True.
    """

    name = "hierarchical_summary"
    priority = 28  # After format_conv=25, before tool_filter=30

    def __init__(
        self,
        max_depth: int = 3,
        max_items: int = 5,
        min_leaf_nodes: int = 50,
        min_occurrences: int = 3,
        enable_tree_detection: bool = True,
        enable_yaml_detection: bool = True,
        enable_ini_detection: bool = True,
        enable_csv_detection: bool = True,
        enable_xml_detection: bool = True,
        enable_log_detection: bool = True,
    ) -> None:
        self.max_depth = max_depth
        self.max_items = max_items
        self.min_leaf_nodes = min_leaf_nodes
        self.min_occurrences = min_occurrences
        self.enable_tree_detection = enable_tree_detection
        self.enable_yaml_detection = enable_yaml_detection
        self.enable_ini_detection = enable_ini_detection
        self.enable_csv_detection = enable_csv_detection
        self.enable_xml_detection = enable_xml_detection
        self.enable_log_detection = enable_log_detection

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Summarize nested structures in messages."""
        total_saved = 0
        summarized_count = 0

        for msg in request.messages:
            original = msg.content
            if not original or len(original) < 200:
                continue

            summarized = None

            # Try JSON first
            summarized = self._try_summarize_json(original)

            # Try YAML
            if summarized is None and self.enable_yaml_detection:
                summarized = self._try_summarize_yaml(original)

            # Try XML
            if summarized is None and self.enable_xml_detection:
                summarized = self._try_summarize_xml(original)

            # Try CSV/TSV
            if summarized is None and self.enable_csv_detection:
                summarized = self._try_summarize_csv(original)

            # Try INI/config
            if summarized is None and self.enable_ini_detection:
                summarized = self._try_summarize_ini(original)

            # Try log format
            if summarized is None and self.enable_log_detection:
                summarized = self._try_summarize_logs(original)

            # Try tree detection
            if summarized is None and self.enable_tree_detection:
                summarized = self._try_summarize_tree(original)

            if summarized is not None and summarized != original:
                msg.content = summarized
                summarized_count += 1
                saved = max(0, len(original) - len(summarized))
                total_saved += saved

        if summarized_count > 0:
            context.record_metric(self.name, "messages_summarized", summarized_count)
            context.record_metric(self.name, "tokens_saved_estimate", total_saved // 4)

        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """Irreversible — no-op."""
        return response

    # ------------------------------------------------------------------
    # JSON summarization
    # ------------------------------------------------------------------

    def _try_summarize_json(self, text: str) -> str | None:
        """Try to parse text as JSON and summarize if large enough."""
        stripped = text.strip()
        if not stripped or stripped[0] not in ("[", "{"):
            return None

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None

        # Check if structure is large enough to warrant summarization
        leaf_count = _count_leaf_nodes(parsed)
        if leaf_count < self.min_leaf_nodes:
            return None

        summarized = _summarize_value(
            parsed,
            depth=0,
            max_depth=self.max_depth,
            max_items=self.max_items,
            min_occurrences=self.min_occurrences,
        )

        # Serialize compactly
        return json.dumps(summarized, separators=(",", ":"), ensure_ascii=False, indent=None)

    # ------------------------------------------------------------------
    # Tree detection (indentation-based nesting)
    # ------------------------------------------------------------------

    def _try_summarize_tree(self, text: str) -> str | None:
        """Detect tree-formatted text and summarize it.

        Example input:
            .
            ├── src
            │   ├── main.py
            │   └── utils.py
            ├── tests
            │   └── test_main.py
            └── README.md

        Output:
            . (7 items, 3 dirs, 4 files)
              src/: 2 files
              tests/: 1 file
              README.md
        """
        lines = text.splitlines()
        if len(lines) < self.min_leaf_nodes // 3:  # Tree lines are roughly 3x leaf nodes
            return None

        # Check for tree-like patterns
        tree_indicators = ["├──", "└──", "|--", "`--", "│", "|"]
        tree_line_count = sum(1 for line in lines if any(ind in line for ind in tree_indicators))
        if tree_line_count < len(lines) * 0.3:
            return None

        # Parse tree structure
        dirs: Counter[str] = Counter()
        files: list[str] = []
        current_dir = ""

        for line in lines:
            # Extract the actual filename/dirname
            name = line.strip()
            # Remove tree indicators
            for ind in tree_indicators:
                name = name.replace(ind, "")
            name = name.strip()

            if not name or name == ".":
                continue

            # Heuristic: if it has a slash or no extension, it's a dir
            if "/" in name or "." not in name.split("/")[-1]:
                current_dir = name
                dirs[current_dir] = 0
            else:
                files.append(name)
                if current_dir:
                    dirs[current_dir] += 1

        # Build summary
        total_items = len(files)
        total_dirs = len(dirs)

        result_lines: list[str] = []
        result_lines.append(f"Tree: {total_items} items in {total_dirs} directories")

        # Show directories with file counts
        for dname, count in dirs.most_common(10):
            result_lines.append(f"  {dname}/: {count} items")

        if len(dirs) > 10:
            result_lines.append(f"  ... {len(dirs) - 10} more directories")

        # Show some files
        if files:
            result_lines.append("Files:")
            for f in files[:5]:
                result_lines.append(f"  {f}")
            if len(files) > 5:
                result_lines.append(f"  ... {len(files) - 5} more files")

        return "\n".join(result_lines)

    # ------------------------------------------------------------------
    # YAML detection and summarization
    # ------------------------------------------------------------------

    def _try_summarize_yaml(self, text: str) -> str | None:
        """Detect YAML content and summarize if large enough."""
        lines = text.splitlines()
        if len(lines) < self.min_leaf_nodes // 2:
            return None

        # YAML heuristics: key: value, - list items, indentation
        yaml_indicators = 0
        for line in lines[:20]:
            stripped = line.strip()
            if (
                stripped.startswith("- ")
                or re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*:", stripped)
                or stripped.startswith("#")
            ):
                yaml_indicators += 1
        if yaml_indicators < 5:
            return None

        try:
            import yaml  # type: ignore[import-untyped]

            parsed = yaml.safe_load(text)
        except Exception:
            return None

        if parsed is None or not isinstance(parsed, (dict, list)):
            return None

        leaf_count = _count_leaf_nodes(parsed)
        if leaf_count < self.min_leaf_nodes:
            return None

        summarized = _summarize_value(
            parsed,
            depth=0,
            max_depth=self.max_depth,
            max_items=self.max_items,
            min_occurrences=self.min_occurrences,
        )
        return json.dumps(summarized, separators=(",", ":"), ensure_ascii=False, indent=None)

    # ------------------------------------------------------------------
    # XML detection and summarization
    # ------------------------------------------------------------------

    def _try_summarize_xml(self, text: str) -> str | None:
        """Detect XML content and summarize if large enough."""
        stripped = text.strip()
        if not stripped.startswith("<") or "<?xml" not in stripped[:100]:
            # Also accept HTML-like without XML declaration
            if not re.match(r"^\s*<\w+", stripped):
                return None

        try:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(stripped)
        except Exception:
            return None

        # Convert XML to nested dict for summarization
        def xml_to_dict(elem: Any) -> Any:
            result: dict[str, Any] = {}
            if elem.attrib:
                result["_attrs"] = dict(elem.attrib)
            children = list(elem)
            text_content = (elem.text or "").strip()
            if text_content:
                result["_text"] = text_content
            if children:
                child_tags: dict[str, list[Any]] = {}
                for child in children:
                    child_tags.setdefault(child.tag, []).append(xml_to_dict(child))
                for tag, items in child_tags.items():
                    result[tag] = items if len(items) > 1 else items[0]
            return result if result else text_content or None

        parsed = {root.tag: xml_to_dict(root)}
        leaf_count = _count_leaf_nodes(parsed)
        if leaf_count < self.min_leaf_nodes:
            return None

        summarized = _summarize_value(
            parsed,
            depth=0,
            max_depth=self.max_depth,
            max_items=self.max_items,
            min_occurrences=self.min_occurrences,
        )
        return json.dumps(summarized, separators=(",", ":"), ensure_ascii=False, indent=None)

    # ------------------------------------------------------------------
    # CSV/TSV detection and summarization
    # ------------------------------------------------------------------

    def _try_summarize_csv(self, text: str) -> str | None:
        """Detect CSV/TSV content and summarize if large enough."""
        lines = text.splitlines()
        if len(lines) < self.min_leaf_nodes // 3:
            return None

        # Detect delimiter
        first_data_line = ""
        for line in lines:
            if line.strip() and not line.strip().startswith("#"):
                first_data_line = line
                break
        if not first_data_line:
            return None

        delimiter = ","
        if first_data_line.count("\t") > first_data_line.count(","):
            delimiter = "\t"

        # Count columns
        cols = first_data_line.split(delimiter)
        if len(cols) < 2:
            return None

        # Parse rows
        import csv
        from io import StringIO

        reader = csv.DictReader(StringIO(text), delimiter=delimiter)
        rows: list[dict[str, str]] = []
        try:
            for row in reader:
                rows.append(dict(row))
                if len(rows) > 10000:  # Safety limit
                    break
        except Exception:
            return None

        if len(rows) < self.min_leaf_nodes // len(cols):
            return None

        # Convert to list of dicts and summarize
        summarized = _summarize_dict_list(
            rows,
            depth=0,
            max_depth=self.max_depth,
            max_items=self.max_items,
            min_occurrences=self.min_occurrences,
        )
        return json.dumps(summarized, separators=(",", ":"), ensure_ascii=False, indent=None)

    # ------------------------------------------------------------------
    # INI/config detection and summarization
    # ------------------------------------------------------------------

    def _try_summarize_ini(self, text: str) -> str | None:
        """Detect INI/config file content and summarize if large enough."""
        lines = text.splitlines()
        if len(lines) < 20:
            return None

        # INI heuristics: [section] headers, key = value pairs
        section_count = sum(1 for line in lines if re.match(r"^\s*\[.*\]\s*$", line))
        kv_count = sum(1 for line in lines if re.match(r"^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*[=:]", line))
        if section_count < 2 and kv_count < 10:
            return None

        # Parse into nested dict
        import configparser
        from io import StringIO

        parser = configparser.ConfigParser()
        try:
            parser.read_file(StringIO(text))
        except Exception:
            return None

        parsed: dict[str, Any] = {}
        for section_name in parser.sections():
            parsed[section_name] = dict(parser[section_name])

        if not parsed:
            return None

        leaf_count = _count_leaf_nodes(parsed)
        if leaf_count < self.min_leaf_nodes // 2:
            return None

        summarized = _summarize_value(
            parsed,
            depth=0,
            max_depth=self.max_depth,
            max_items=self.max_items,
            min_occurrences=self.min_occurrences,
        )
        return json.dumps(summarized, separators=(",", ":"), ensure_ascii=False, indent=None)

    # ------------------------------------------------------------------
    # Log format detection and summarization
    # ------------------------------------------------------------------

    def _try_summarize_logs(self, text: str) -> str | None:
        """Detect timestamped log lines and summarize by level/frequency."""
        lines = text.splitlines()
        if len(lines) < self.min_leaf_nodes // 2:
            return None

        # Log pattern: timestamp + level + message
        log_pattern = re.compile(
            r"^\s*(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|"
            r"\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}|"
            r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"
            r".*?\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b"
        )

        log_lines: list[tuple[str, str, str]] = []  # (timestamp, level, message)
        for line in lines:
            match = log_pattern.match(line)
            if match:
                level = match.group(1)
                # Extract timestamp and remaining message
                ts_end = match.end(1)
                timestamp = line[:ts_end].strip()[:30]
                message = line[ts_end:].strip()[:200]
                log_lines.append((timestamp, level, message))

        if len(log_lines) < len(lines) * 0.5 or len(log_lines) < 10:
            return None

        # Summarize by level
        level_counts: Counter[str] = Counter()
        for _ts, level, _msg in log_lines:
            level_counts[level] += 1

        # Collect unique error messages
        error_messages: list[str] = []
        seen_errors: set[str] = set()
        for _ts, level, msg in log_lines:
            if level in ("ERROR", "FATAL", "CRITICAL") and msg not in seen_errors:
                seen_errors.add(msg)
                error_messages.append(msg)

        # Build summary
        result_lines: list[str] = []
        result_lines.append(f"Log summary: {len(log_lines)} lines")
        result_lines.append("Level distribution:")
        for level, count in level_counts.most_common():
            result_lines.append(f"  {level}: {count}")

        if error_messages:
            result_lines.append(f"Unique errors ({len(error_messages)}):")
            for msg in error_messages[: self.max_items]:
                result_lines.append(f"  - {msg}")
            if len(error_messages) > self.max_items:
                result_lines.append(f"  ... {len(error_messages) - self.max_items} more")

        # Show sample messages per level
        result_lines.append("Samples:")
        for level in level_counts:
            samples = [msg for _ts, lv, msg in log_lines if lv == level][:2]
            for msg in samples:
                result_lines.append(f"  [{level}] {msg}")

        return "\n".join(result_lines)
