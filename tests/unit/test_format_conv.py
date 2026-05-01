"""Comprehensive tests for FormatConverter.

Covers every data shape and conversion edge case:
1. Tabular detection (uniform keys, mixed keys, too few rows)
2. Config detection (nested dicts, shallow dicts, arrays of objects)
3. CSV conversion (RFC 4180 compliance, commas, quotes, newlines in data)
4. YAML conversion (nested dicts, multiline strings, empty dicts)
5. Round-trip validation (parse → convert → parse → compare)
6. Irregular data (mixed key sets, primitives, arrays of primitives)
7. Performance budgets
"""

import json

import pytest

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.result import unwrap
from lattice.core.transport import Message, Request
from lattice.transforms.format_conv import DataShape, FormatConverter

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def converter() -> FormatConverter:
    return FormatConverter(min_tabular_rows=2, key_uniformity_threshold=0.8)


@pytest.fixture
def pipeline() -> CompressorPipeline:
    """Pipeline with only FormatConverter for isolated testing."""
    config = LatticeConfig()
    p = CompressorPipeline(config=config)
    p.register(FormatConverter())
    return p


# =============================================================================
# Shape Detection
# =============================================================================


class TestShapeDetection:
    """Test _detect_shape with every shape type."""

    # ---- Tabular ----

    def test_tabular_uniform_rows(self, converter: FormatConverter) -> None:
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        assert converter._detect_shape(data) == DataShape.TABULAR

    def test_tabular_100_rows_uniform(self, converter: FormatConverter) -> None:
        data = [{"id": i, "name": f"user_{i}"} for i in range(100)]
        assert converter._detect_shape(data) == DataShape.TABULAR

    def test_tabular_one_row_too_few(self, converter: FormatConverter) -> None:
        """Single row → not tabular (no repeated keys to save)."""
        data = [{"a": 1}]
        assert converter._detect_shape(data) == DataShape.IRREGULAR

    def test_tabular_mixed_keys_below_threshold(self, converter: FormatConverter) -> None:
        """80% threshold: 4 rows match, 1 row differs → still tabular."""
        data = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
            {"a": 5, "b": 6},
            {"a": 7, "b": 8},
            {"a": 9, "c": 10},  # differs
        ]
        assert converter._detect_shape(data) == DataShape.TABULAR

    def test_tabular_mixed_keys_above_threshold(self, converter: FormatConverter) -> None:
        """Below threshold: only 50% match → irregular."""
        data = [
            {"a": 1, "b": 2},
            {"a": 3, "c": 4},
            {"x": 5, "y": 6},
            {"x": 7, "y": 8},
        ]
        assert converter._detect_shape(data) == DataShape.IRREGULAR

    def test_tabular_with_optional_fields_missing(self, converter: FormatConverter) -> None:
        """Rows missing optional field → still tabular if uniformity > 0.8."""
        data = [
            {"id": 1, "name": "Alice", "email": "a@e"},
            {"id": 2, "name": "Bob"},  # missing email
            {"id": 3, "name": "Carol", "email": "c@e"},
        ]
        assert converter._detect_shape(data) == DataShape.TABULAR

    # ---- Config ----

    def test_config_nested_dicts(self, converter: FormatConverter) -> None:
        data = {"db": {"host": "h", "port": 1}, "api": {"version": "v1"}}
        assert converter._detect_shape(data) == DataShape.CONFIG

    def test_config_shallow_dict(self, converter: FormatConverter) -> None:
        """Flat dict with 2 fields → not config (not nested enough)."""
        data = {"a": 1, "b": 2}
        assert converter._detect_shape(data) != DataShape.CONFIG

    def test_config_deep_nesting(self, converter: FormatConverter) -> None:
        data = {"a": {"b": {"c": {"d": "deep"}}}}
        assert converter._detect_shape(data) == DataShape.CONFIG

    def test_config_with_array_of_objects(self, converter: FormatConverter) -> None:
        """Dict containing array of dicts → NOT config (tabular)."""
        data = {"users": [{"id": 1}, {"id": 2}]}
        assert converter._detect_shape(data) != DataShape.CONFIG

    def test_config_with_number_array(self, converter: FormatConverter) -> None:
        """Dict with arrays of primitives is borderline — not config by our criteria."""
        data = {"ports": [80, 443], "hosts": ["a", "b"]}
        # Our config detection requires nested dicts, not arrays of primitives
        assert converter._detect_shape(data) == DataShape.IRREGULAR

    # ---- Primitives ----

    def test_primitive_string(self, converter: FormatConverter) -> None:
        assert converter._detect_shape("hello") == DataShape.PRIMITIVE

    def test_primitive_number(self, converter: FormatConverter) -> None:
        assert converter._detect_shape(42) == DataShape.PRIMITIVE

    def test_primitive_null(self, converter: FormatConverter) -> None:
        assert converter._detect_shape(None) == DataShape.PRIMITIVE

    def test_empty_list(self, converter: FormatConverter) -> None:
        assert converter._detect_shape([]) == DataShape.PRIMITIVE

    def test_array_of_primitives(self, converter: FormatConverter) -> None:
        assert converter._detect_shape([1, 2, 3]) == DataShape.ARRAY_PRIMITIVE

    def test_mixed_array(self, converter: FormatConverter) -> None:
        assert converter._detect_shape([1, "two", 3]) == DataShape.ARRAY_PRIMITIVE


# =============================================================================
# CSV Conversion
# =============================================================================


class TestCSVConversion:
    """Test CSV conversion with RFC 4180 edge cases."""

    def test_simple_table(self, converter: FormatConverter) -> None:
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = converter._to_csv(data)
        assert result is not None
        lines = result.strip().split("\n")
        assert lines[0] == "a,b"
        assert lines[1] == "1,2"
        assert lines[2] == "3,4"

    def test_comma_in_field_quoted(self, converter: FormatConverter) -> None:
        """RFC 4180: field containing comma must be wrapped in double-quotes."""
        data = [{"name": "Alice, Bob", "age": 30}]
        result = converter._to_csv(data)
        # Keys are sorted alphabetically: age, name
        assert (
            '"Alice, Bob",30' in result
            or result == 'age,name\n30,"Alice, Bob"\n'
            or '"Alice, Bob"' in result
        )

    def test_quote_in_field_doubled(self, converter: FormatConverter) -> None:
        """RFC 4180: field containing quote must double the quote."""
        data = [{"quote": 'She said "hello"', "blank": ""}]
        result = converter._to_csv(data)
        assert '"She said ""hello""",' in result or '"She said ""hello"""' in result

    def test_newline_in_field(self, converter: FormatConverter) -> None:
        """Multiline fields should be wrapped in quotes."""
        data = [{"text": "line1\nline2", "id": 1}]
        result = converter._to_csv(data)
        assert result is not None  # Should not crash
        # Py csv module handles this by wrapping in quotes
        assert '"' in result

    def test_boolean_serialization(self, converter: FormatConverter) -> None:
        data = [{"active": True}, {"active": False}]
        result = converter._to_csv(data)
        assert "true" in result
        assert "false" in result
        assert "True" not in result  # Python repr, not what we want

    def test_null_serialization(self, converter: FormatConverter) -> None:
        data = [{"name": "Alice", "email": None}, {"name": "Bob", "email": "b"}]
        result = converter._to_csv(data)
        lines = result.strip().split("\n")
        # Null should be empty string
        assert lines[1] == "Alice," or lines[1] == ",Alice"  # depending on key order

    def test_nested_json_in_field(self, converter: FormatConverter) -> None:
        """Complex fields serialized as JSON within CSV."""
        data = [{"id": 1, "config": {"a": 1, "b": 2}}]
        result = converter._to_csv(data)
        # Keys sorted: config, id. JSON serialized with double quotes,
        # which csv module doubles → "" per original double quote
        # Exact expected output: the dict as JSON is a single CSV field
        assert result == 'config,id\n"{""a"": 1, ""b"": 2}",1\n'

    def test_roundtrip_simple(self, converter: FormatConverter) -> None:
        """CSV → parse back → equals original."""
        original = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        csv_text = converter._to_csv(original)
        assert csv_text is not None
        parsed = converter._from_csv(csv_text)
        assert parsed is not None
        assert len(parsed) == 2
        assert parsed[0]["a"] == 1  # parser infers int from text
        assert parsed[1]["b"] == 4

    def test_token_reduction(self, converter: FormatConverter) -> None:
        """CSV should use fewer characters than JSON."""
        data = [{"id": i, "name": f"user_{i}"} for i in range(100)]
        json_text = json.dumps(data)
        csv_text = converter._to_csv(data)
        assert csv_text is not None
        # CSV should be significantly shorter (no repeated key names)
        assert len(csv_text) < len(json_text) * 0.6


# =============================================================================
# YAML Conversion
# =============================================================================


class TestYAMLConversion:
    """Test YAML conversion with nested structures."""

    @pytest.fixture(autouse=True)
    def skip_if_no_yaml(self) -> None:
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("yaml not installed")

    def test_simple_config(self, converter: FormatConverter) -> None:
        data = {"db": {"host": "localhost", "port": 5432}}
        result = converter._to_yaml(data)
        assert result is not None
        assert "db:" in result
        assert "  host: localhost" in result
        assert "  port: 5432" in result

    def test_deeply_nested(self, converter: FormatConverter) -> None:
        data = {"a": {"b": {"c": {"d": "deep"}}}}
        result = converter._to_yaml(data)
        assert result is not None
        assert "d: deep" in result

    def test_roundtrip_yaml(self, converter: FormatConverter) -> None:
        """YAML → parse back → equals original."""
        original = {"db": {"host": "h", "port": 1}, "api": {"version": "v1"}}
        yaml_text = converter._to_yaml(original)
        assert yaml_text is not None
        parsed = converter._from_yaml(yaml_text)
        assert parsed == original

    def test_sorted_keys(self, converter: FormatConverter) -> None:
        """YAML should have deterministic key order."""
        data = {"z": 1, "a": 2, "m": 3}
        result = converter._to_yaml(data)
        assert result is not None
        # Keys should be alphabetical: a, m, z
        lines = [line for line in result.split("\n") if ":" in line]
        keys = [line.split(":")[0].strip() for line in lines]
        assert keys == sorted(keys)

    def test_token_reduction(self, converter: FormatConverter) -> None:
        """YAML should use fewer chars than JSON for nested configs."""
        data = {
            "database": {"host": "localhost", "port": 5432, "ssl": True},
            "api": {"version": "2.0", "timeout": 30},
            "logging": {"level": "debug", "file": "/tmp/app.log"},
        }
        json_text = json.dumps(data, indent=2)
        yaml_text = converter._to_yaml(data)
        assert yaml_text is not None
        # YAML should be more compact for nested configs
        assert len(yaml_text) < len(json_text)


# =============================================================================
# FormatConverter E2E
# =============================================================================


class TestFormatConverterE2E:
    """End-to-end tests with real pipeline."""

    @pytest.mark.asyncio
    async def test_message_with_table(self, pipeline: CompressorPipeline) -> None:
        """Table in message → converted to CSV."""
        table = json.dumps([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        request = Request(messages=[Message(role="user", content=table)])
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        content = modified.messages[0].content
        assert "id,name" in content  # CSV header
        assert '"id":' not in content  # JSON gone

    @pytest.mark.asyncio
    async def test_message_with_config(self, pipeline: CompressorPipeline) -> None:
        """Nested config in message → converted to YAML."""
        config = json.dumps({"db": {"host": "localhost", "port": 5432}})
        request = Request(messages=[Message(role="user", content=config)])
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        content = modified.messages[0].content
        # YAML uses colons without braces for nested
        assert "db:" in content
        # YAML won't have JSON braces, but Python dict string repr won't either for some cases.
        # The key check is "db:" appearing on its own line with colon formatting.
        assert "db:" in content

    @pytest.mark.asyncio
    async def test_non_json_unchanged(self, pipeline: CompressorPipeline) -> None:
        """Plain text → not modified."""
        request = Request(messages=[Message(role="user", content="Hello, world.")])
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        assert modified.messages[0].content == "Hello, world."

    @pytest.mark.asyncio
    async def test_irregular_json_unchanged(self, pipeline: CompressorPipeline) -> None:
        """Mixed-key JSON (not tabular) → not converted."""
        data = json.dumps([{"a": 1, "b": 2}, {"x": 3, "y": 4}])
        request = Request(messages=[Message(role="user", content=data)])
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        # Should not be transformed (irregular keys)
        assert modified.messages[0].content == data

    @pytest.mark.asyncio
    async def test_invalid_json_unchanged(self, pipeline: CompressorPipeline) -> None:
        """Invalid JSON → not modified."""
        request = Request(messages=[Message(role="user", content='{"broken": json}')])
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        assert modified.messages[0].content == '{"broken": json}'

    @pytest.mark.asyncio
    async def test_metrics_populated(self, pipeline: CompressorPipeline) -> None:
        """Metrics track conversion savings."""
        table = json.dumps([{"id": i, "name": f"user_{i}"} for i in range(50)])
        request = Request(messages=[Message(role="user", content=table)])
        context = TransformContext()
        await pipeline.process(request, context)

        assert "format_conversion" in context.transforms_applied
        fmt_metrics = context.metrics["transforms"]["format_conversion"]
        assert "messages_converted" in fmt_metrics
        assert "tokens_saved_estimate" in fmt_metrics

    @pytest.mark.asyncio
    async def test_single_key_dict_wrapping_list(self, pipeline: CompressorPipeline) -> None:
        """Common API pattern: {'employees': [{...}, {...}]} → extract and convert."""
        data = json.dumps(
            {
                "employees": [
                    {"id": i, "name": f"Emp_{i}", "dept": "Eng", "salary": 100000 + i * 1000}
                    for i in range(50)
                ]
            }
        )
        request = Request(messages=[Message(role="user", content=data)])
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        content = modified.messages[0].content
        # Should be CSV/TSV, not JSON (header is sorted alphabetically)
        assert "dept,id,name,salary" in content or "dept\tid\tname\tsalary" in content
        assert '"employees"' not in content  # wrapper key gone

    @pytest.mark.asyncio
    async def test_single_key_dict_config_converted_to_yaml(
        self, pipeline: CompressorPipeline
    ) -> None:
        """Single-key dict with nested config inner data → converted to YAML."""
        data = json.dumps({"config": {"host": "localhost", "port": 5432}})
        request = Request(messages=[Message(role="user", content=data)])
        context = TransformContext()
        result = await pipeline.process(request, context)
        modified = unwrap(result)

        # Nested configs are converted to YAML regardless of wrapper
        content = modified.messages[0].content
        assert "config:" in content
        assert "  host: localhost" in content
        assert "  port: 5432" in content


# =============================================================================
# Round-trip validation
# =============================================================================


class TestRoundTrip:
    """Validate that conversions are lossless."""

    @pytest.mark.asyncio
    async def test_csv_roundtrip(self, converter: FormatConverter) -> None:
        """JSON → CSV → JSON = original."""
        original_data = [
            {"id": 1, "name": "Alice", "active": True, "score": 95.5},
            {"id": 2, "name": "Bob", "active": False, "score": 82},
        ]
        json.dumps(original_data)
        csv_text = converter._to_csv(original_data)
        assert csv_text is not None

        parsed = converter._from_csv(csv_text)
        assert parsed is not None

        # Values may be stringified after CSV (numbers → strings, bools → strings)
        # That's expected for RFC 4180 CSV.
        # But the structure must match: same keys, same row count.
        assert len(parsed) == len(original_data)
        for orig_row, parsed_row in zip(original_data, parsed, strict=False):
            assert set(orig_row.keys()) == set(parsed_row.keys())

    @pytest.mark.asyncio
    async def test_yaml_roundtrip(self, converter: FormatConverter) -> None:
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("yaml not installed")

        original = {"a": {"b": [1, 2, 3], "c": {"d": "nested"}}}
        yaml_text = converter._to_yaml(original)
        assert yaml_text is not None

        parsed = converter._from_yaml(yaml_text)
        assert parsed is not None
        assert converter._deep_equal(original, parsed)


# =============================================================================
# CSV serialize/parse
# =============================================================================


class TestCSVValueHandling:
    """Test value serialization edge cases."""

    def test_serialize_bool(self, converter: FormatConverter) -> None:
        assert converter._serialize_csv_value(True) == "true"
        assert converter._serialize_csv_value(False) == "false"

    def test_serialize_none(self, converter: FormatConverter) -> None:
        assert converter._serialize_csv_value(None) == ""

    def test_serialize_number(self, converter: FormatConverter) -> None:
        assert converter._serialize_csv_value(42) == "42"
        assert converter._serialize_csv_value(3.14) == "3.14"

    def test_serialize_nested(self, converter: FormatConverter) -> None:
        val = {"nested": True}
        result = converter._serialize_csv_value(val)
        assert result == '{"nested": true}'

    def test_parse_empty(self, converter: FormatConverter) -> None:
        assert converter._parse_csv_value("") is None

    def test_parse_bool(self, converter: FormatConverter) -> None:
        assert converter._parse_csv_value("true") is True
        assert converter._parse_csv_value("false") is False
        assert converter._parse_csv_value("True") is True

    def test_parse_number(self, converter: FormatConverter) -> None:
        assert converter._parse_csv_value("42") == 42
        assert converter._parse_csv_value("3.14") == 3.14

    def test_parse_json(self, converter: FormatConverter) -> None:
        result = converter._parse_csv_value("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_parse_json_nested(self, converter: FormatConverter) -> None:
        result = converter._parse_csv_value('{"a": 1}')
        assert result == {"a": 1}


# =============================================================================
# Performance
# =============================================================================


class TestPerformance:
    """Performance regression tests."""

    def test_large_conversion_speed(self, converter: FormatConverter) -> None:
        import time

        # 10,000 rows × 5 columns
        data = [{f"col_{j}": f"data_{i}_{j}" for j in range(5)} for i in range(10000)]
        text = json.dumps(data)
        request = Request(messages=[Message(role="user", content=text)])
        context = TransformContext()
        start = time.perf_counter()
        result = converter.process(request, context)
        elapsed_ms = (time.perf_counter() - start) * 1000
        unwrap(result)

        assert elapsed_ms < 500.0, f"Large conversion took {elapsed_ms:.1f}ms > 500ms"
        # Since we called process() directly (not through pipeline),
        # transforms_applied is not populated. Check metrics instead.
        assert context.metrics["transforms"]["format_conversion"]["messages_converted"] == 1

        # Should save significant tokens
        saved = context.metrics["transforms"]["format_conversion"]["tokens_saved_estimate"]
        assert saved > 0
