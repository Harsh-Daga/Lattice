"""Shared pipeline construction helpers.

Proxy, SDK, and MCP modes should use the same compression pipeline ordering so
runtime contracts and strategy selection do not drift across integration modes.
Execution-only transforms such as batching/speculation/delta remain opt-in.
"""

from __future__ import annotations

from typing import Any

from lattice.core.config import LatticeConfig
from lattice.core.pipeline import CompressorPipeline
from lattice.transforms.batching import BatchingTransform
from lattice.transforms.cache_arbitrage import CacheArbitrageOptimizer
from lattice.transforms.content_profiler import ContentProfiler
from lattice.transforms.context_selector import (
    InformationTheoreticSelector,
    SubmodularContextSelector,
)
from lattice.transforms.delta_encode import DeltaEncoder
from lattice.transforms.dictionary_compress import DictionaryCompressor
from lattice.transforms.format_conv import FormatConverter
from lattice.transforms.grammar_compress import GrammarCompressor
from lattice.transforms.hierarchical_summary import HierarchicalSummarizer
from lattice.transforms.message_dedup import MessageDeduplicator
from lattice.transforms.output_cleanup import OutputCleanup
from lattice.transforms.prefix_opt import PrefixOptimizer
from lattice.transforms.rate_distortion import RateDistortionCompressor
from lattice.transforms.reference_sub import ReferenceSubstitution
from lattice.transforms.runtime_contract import RuntimeContractTransform
from lattice.transforms.self_information import SelfInformationScorer
from lattice.transforms.speculative import SpeculativeTransform
from lattice.transforms.strategy_selector import StrategySelector
from lattice.transforms.structural_fingerprint import StructuralFingerprint
from lattice.transforms.tool_filter import ToolOutputFilter


def build_default_pipeline(
    config: LatticeConfig,
    *,
    include_execution_transforms: bool = False,
    session_manager: Any | None = None,
) -> CompressorPipeline:
    """Build the standard LATTICE transform pipeline.

    Args:
        config: Runtime configuration.
        include_execution_transforms: Include proxy-only batching,
            speculative, and delta transforms.
        session_manager: Required when execution transforms include delta
            encoding.
    """
    pipeline = CompressorPipeline(config=config)

    if config.transform_content_profiler:
        pipeline.register(ContentProfiler())
    if config.transform_runtime_contract:
        pipeline.register(RuntimeContractTransform())
    if config.transform_strategy_selector:
        pipeline.register(StrategySelector())
    if config.transform_cache_arbitrage:
        pipeline.register(CacheArbitrageOptimizer())
    if config.transform_prefix_opt:
        pipeline.register(PrefixOptimizer())
    if config.transform_structural_fingerprint:
        pipeline.register(StructuralFingerprint())
    if config.transform_self_information:
        pipeline.register(SelfInformationScorer())
    if config.transform_message_dedup:
        pipeline.register(MessageDeduplicator())
    if config.transform_context_selector:
        pipeline.register(SubmodularContextSelector(token_budget=config.submodular_token_budget))
        pipeline.register(InformationTheoreticSelector(token_budget=config.submodular_token_budget))
    if config.transform_reference_sub:
        pipeline.register(ReferenceSubstitution())
    if config.transform_semantic_compress:
        pipeline.register(
            RateDistortionCompressor(
                distortion_budget=config.rate_distortion_budget,
            )
        )
    if config.transform_grammar_compress:
        pipeline.register(GrammarCompressor())
    if config.transform_dictionary_compress:
        pipeline.register(DictionaryCompressor())
    if config.transform_hierarchical_summary:
        pipeline.register(HierarchicalSummarizer())
    if config.transform_tool_filter:
        pipeline.register(ToolOutputFilter())
    if config.transform_format_conversion:
        pipeline.register(FormatConverter())
    if config.transform_output_cleanup:
        pipeline.register(OutputCleanup())

    if include_execution_transforms:
        pipeline.register(BatchingTransform())
        pipeline.register(SpeculativeTransform())
        if session_manager is not None:
            pipeline.register(DeltaEncoder(session_manager=session_manager))

    return pipeline

_EXECUTION_TRANSFORMS = {"batching", "speculative", "delta_encoder"}


def pipeline_summary(pipeline: CompressorPipeline) -> dict[str, Any]:
    """Return a stable operational summary for a pipeline."""
    transforms = [t.name for t in pipeline.transforms]
    execution = [name for name in transforms if name in _EXECUTION_TRANSFORMS]
    core = [name for name in transforms if name not in _EXECUTION_TRANSFORMS]
    return {
        "count": len(transforms),
        "transforms": transforms,
        "core_transforms": core,
        "execution_transforms": execution,
        "runtime_contract_enabled": "runtime_contract" in transforms,
        "strategy_selector_enabled": "strategy_selector" in transforms,
    }


__all__ = ["build_default_pipeline", "pipeline_summary"]
