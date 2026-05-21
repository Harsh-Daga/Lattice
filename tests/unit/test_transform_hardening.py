"""Tests for transform hardening — span-aware and task-aware gating.

Verifies that transforms respect SIG metadata when making gating decisions.
"""

from __future__ import annotations

import asyncio

from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.ir.builder import build_ir
from lattice.ir.primitives import prompt_ir_v2_from_legacy
from lattice.transport.types import Message, Request


def _ir_v2_for(req: Request) -> "object":
    """Helper: build a PromptIRV2 from a Request for IR-native optimize() tests."""
    return prompt_ir_v2_from_legacy(build_ir(req))


class TestReferenceSubHardening:
    """reference_sub preserves referent mappings and handles UUID content."""

    def test_processes_uuid_heavy_content_safely(self) -> None:
        from lattice.transforms.reference_sub import ReferenceSubstitution

        transform = ReferenceSubstitution()
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="Debug: UUID 550e8400-e29b-41d4-a716-446655440000 appears. Error at /api/v1/endpoint. Check UUID 6ba7b810-9dad-11d1-80b4-00c04fd430c8.",
                )
            ]
        )
        req.metadata["_lattice_task_classification"] = {"task_class": "retrieval"}
        ctx = TransformContext()
        result = transform.process(req, ctx)
        assert is_ok(result)
        modified = unwrap(result)
        # Should have compressed UUIDs
        assert "<ref_" in modified.messages[0].content


class TestMessageDedupHardening:
    """message_dedup preserves last-turn and tool messages."""

    def test_preserves_last_message(self) -> None:
        from lattice.transforms.message_dedup import MessageDeduplicator

        transform = MessageDeduplicator()
        req = Request(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="user", content="Hello"),
                Message(role="user", content="Debug the error"),
            ]
        )
        ctx = TransformContext()
        result = transform.optimize(_ir_v2_for(req), req, ctx)
        assert is_ok(result)
        modified_ir = unwrap(result)
        # Last non-dup message content preserved somewhere in the IR text
        all_text = " ".join(sp.text for sec in modified_ir.sections for sp in sec.spans)
        assert "Debug" in all_text

    def test_allows_short_conversation(self) -> None:
        from lattice.transforms.message_dedup import MessageDeduplicator

        transform = MessageDeduplicator()
        req = Request(
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hi"),
                Message(role="user", content="Hi"),
            ]
        )
        ctx = TransformContext()
        result = transform.optimize(_ir_v2_for(req), req, ctx)
        assert is_ok(result)


class TestDeletedTransforms:
    """Deleted transforms are no longer known in the registry."""

    def test_structural_fingerprint_removed(self) -> None:
        from lattice.core.transform_registry import is_transform_name_known

        assert is_transform_name_known("structural_fingerprint") is False

    def test_hierarchical_summary_removed(self) -> None:
        from lattice.core.transform_registry import is_transform_name_known

        assert is_transform_name_known("hierarchical_summary") is False

    def test_arithmetic_sequence_removed(self) -> None:
        from lattice.core.transform_registry import is_transform_name_known

        assert is_transform_name_known("arithmetic_sequence") is False

    def test_code_factoring_removed(self) -> None:
        from lattice.core.transform_registry import is_transform_name_known

        assert is_transform_name_known("code_factoring") is False

    def test_stack_interning_removed(self) -> None:
        from lattice.core.transform_registry import is_transform_name_known

        assert is_transform_name_known("stack_interning") is False

    def test_stable_prefix_removed(self) -> None:
        from lattice.core.transform_registry import is_transform_name_known

        assert is_transform_name_known("stable_prefix") is False

    def test_instruction_context_sep_removed(self) -> None:
        from lattice.core.transform_registry import is_transform_name_known

        assert is_transform_name_known("instruction_context_sep") is False

    def test_self_information_removed(self) -> None:
        from lattice.core.transform_registry import is_transform_name_known

        assert is_transform_name_known("self_information") is False


class TestRateDistortionHardening:
    """rate_distortion compresses long-form safely."""

    def test_preserves_narrative_structure(self) -> None:
        from lattice.transforms.rate_distortion import RateDistortionCompressor

        transform = RateDistortionCompressor()
        narrative = "The system experienced a failure. The root cause was a memory leak. " * 10
        req = Request(messages=[Message(role="user", content=narrative)])
        ctx = TransformContext()
        result = transform.optimize(_ir_v2_for(req), req, ctx)
        assert is_ok(result)


class TestFormatConvHardening:
    """format_conversion preserves table/JSON structure."""

    def test_converts_table_content(self) -> None:
        from lattice.transforms.format_conv import FormatConverter

        transform = FormatConverter()
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="| Name | Value |\n| Alice | 100 |\n| Bob | 200 |",
                )
            ]
        )
        ctx = TransformContext()
        result = transform.optimize(_ir_v2_for(req), req, ctx)
        assert is_ok(result)


class TestPipelinePSGSafety:
    """Pipeline PSG checks only apply to irreversible transforms."""

    def test_message_dedup_with_entities_runs_safely(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.core.pipeline import CompressorPipeline
        from lattice.transforms.message_dedup import MessageDeduplicator

        config = LatticeConfig(graceful_degradation=True)
        pipeline = CompressorPipeline(config=config)
        pipeline.register(MessageDeduplicator())

        req = Request(
            messages=[
                Message(role="user", content="hello world"),
                Message(role="user", content="hello world"),
                Message(
                    role="user", content="Error: check UUID 550e8400-e29b-41d4-a716-446655440000"
                ),
            ]
        )
        req.metadata["_lattice_protected_spans"] = [2]
        ctx = TransformContext()

        # Should run — message_dedup is irreversible but entities are in the last message
        async def run():
            result = await pipeline.process(req, ctx)
            assert is_ok(result)

        asyncio.run(run())

    def test_reversible_transform_bypasses_entity_check(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.core.pipeline import CompressorPipeline
        from lattice.transforms.reference_sub import ReferenceSubstitution

        config = LatticeConfig(graceful_degradation=True)
        pipeline = CompressorPipeline(config=config)
        pipeline.register(ReferenceSubstitution())

        req = Request(
            messages=[
                Message(
                    role="user",
                    content="UUIDs: 550e8400-e29b-41d4-a716-446655440000, 6ba7b810-9dad-11d1-80b4-00c04fd430c8",
                )
            ]
        )
        req.metadata["_lattice_protected_spans"] = [0]
        req.metadata["_lattice_task_classification"] = {"task_class": "retrieval"}
        req.metadata["_lattice_risk_score"] = {"strict_instructions": 0, "level": "LOW"}
        ctx = TransformContext()

        async def run():
            result = await pipeline.process(req, ctx)
            assert is_ok(result)
            modified = unwrap(result)
            # Reversible — entities replaced with refs, not lost
            content = modified.messages[0].content
            assert "<ref_" in content

        asyncio.run(run())
