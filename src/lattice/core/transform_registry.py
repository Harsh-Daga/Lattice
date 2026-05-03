"""Central transform registry — single source of truth for all metadata.

Every built-in transform is declared exactly once in :data:`BUILTIN_TRANSFORMS`.
Config enablement, pipeline construction, safety gating, and diagnostics all
consume this registry so they cannot drift out of sync.

Design principles
-----------------
* **No circular imports** — this module stores *metadata* (strings, ints, enums).
  It does **not** import transform classes.
* **Lazy class loading** — class paths are stored as dotted strings and imported
  only when :func:`_load_transform_class` is called inside ``build_default_pipeline()``.
* **One canonical name** — aliases exist for backward compatibility, but every
  lookup resolves to the canonical spec.
* **Immutable specs** — :class:`TransformSpec` is ``frozen=True, slots=True``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

# ---------------------------------------------------------------------------
# Safety buckets (mirror lattice.utils.validation.TransformSafetyBucket values)
# ---------------------------------------------------------------------------


SAFE = "safe"
CONDITIONAL = "conditional"
DANGEROUS = "dangerous"


# ---------------------------------------------------------------------------
# TransformSpec — immutable metadata for one transform
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class TransformSpec:
    """Metadata for a single built-in transform."""

    # Canonical name used in pipeline, logs, and diagnostics.
    canonical_name: str

    # Alternative names that resolve to this spec (e.g. short aliases).
    aliases: tuple[str, ...] = ()

    # Config field that gates this transform (e.g. "transform_context_selector").
    config_flag: str = ""

    # Priority for pipeline ordering (lower = earlier).
    priority: int = 50

    # Safety classification.
    safety_bucket: str = SAFE

    # Included in the default pipeline (non-execution builds)?
    default_pipeline: bool = True

    # Only included when ``include_execution_transforms=True`` (proxy-only).
    execution_only: bool = False

    # Dotted import path, e.g. "lattice.transforms.content_profiler.ContentProfiler".
    # Empty string means the transform is handled manually in pipeline_factory.
    factory_path: str = ""

    # Mapping from LatticeConfig field names to constructor kwarg names.
    # Example: {"submodular_token_budget": "token_budget"}
    factory_kwargs: dict[str, str] = dataclasses.field(default_factory=dict)

    # Human-readable description for docs and diagnostics.
    description: str = ""


# ---------------------------------------------------------------------------
# Built-in transforms — declared in priority order
# ---------------------------------------------------------------------------

BUILTIN_TRANSFORMS: tuple[TransformSpec, ...] = (
    # Phase 0 — profiling & governance
    TransformSpec(
        canonical_name="content_profiler",
        config_flag="transform_content_profiler",
        priority=1,
        safety_bucket=SAFE,
        factory_path="lattice.transforms.content_profiler.ContentProfiler",
        description="Classifies content, computes semantic risk score",
    ),
    TransformSpec(
        canonical_name="runtime_contract",
        config_flag="transform_runtime_contract",
        priority=2,
        safety_bucket=SAFE,
        factory_path="lattice.transforms.runtime_contract.RuntimeContractTransform",
        description="Enforces per-transform budget and timeout",
    ),
    # Execution-only — proxy hot path
    TransformSpec(
        canonical_name="speculative",
        config_flag="transform_speculation",
        priority=2,
        safety_bucket=SAFE,
        default_pipeline=False,
        execution_only=True,
        factory_path="lattice.transforms.speculative.SpeculativeTransform",
        description="Speculative token generation",
    ),
    TransformSpec(
        canonical_name="batching",
        config_flag="transform_batching",
        priority=3,
        safety_bucket=SAFE,
        default_pipeline=False,
        execution_only=True,
        factory_path="lattice.transforms.batching.BatchingTransform",
        description="Request batching for multi-turn workloads",
    ),
    TransformSpec(
        canonical_name="delta_encoder",
        config_flag="transform_batching",  # gated by execution mode, not a real flag
        priority=5,
        safety_bucket=SAFE,
        default_pipeline=False,
        execution_only=True,
        description="Session-based delta encoding (needs session_manager)",
    ),
    # Phase 1 — cache & prefix
    TransformSpec(
        canonical_name="cache_arbitrage",
        aliases=("cache_optimizer",),
        config_flag="transform_cache_arbitrage",
        priority=9,
        safety_bucket=SAFE,
        factory_path="lattice.transforms.cache_arbitrage.CacheArbitrageOptimizer",
        description="Reorders messages for KV-cache alignment",
    ),
    TransformSpec(
        canonical_name="prefix_optimizer",
        aliases=("prefix_opt",),
        config_flag="transform_prefix_opt",
        priority=10,
        safety_bucket=SAFE,
        factory_path="lattice.transforms.prefix_opt.PrefixOptimizer",
        description="Deduplicates common prefixes across messages",
    ),
    # Phase 2 — fingerprinting & scoring
    TransformSpec(
        canonical_name="structural_fingerprint",
        config_flag="transform_structural_fingerprint",
        priority=12,
        safety_bucket=CONDITIONAL,
        factory_path="lattice.transforms.structural_fingerprint.StructuralFingerprint",
        description="Pattern detection for repeated structures",
    ),
    TransformSpec(
        canonical_name="self_information",
        config_flag="transform_self_information",
        priority=14,
        safety_bucket=CONDITIONAL,
        factory_path="lattice.transforms.self_information.SelfInformationScorer",
        description="Entropy-based message filtering",
    ),
    TransformSpec(
        canonical_name="message_dedup",
        aliases=("message_deduplicator",),
        config_flag="transform_message_dedup",
        priority=15,
        safety_bucket=CONDITIONAL,
        factory_path="lattice.transforms.message_dedup.MessageDeduplicator",
        description="Exact/near-duplicate message removal",
    ),
    # Phase 3 — context selection
    TransformSpec(
        canonical_name="context_selector",
        config_flag="transform_context_selector",
        priority=18,
        safety_bucket=SAFE,
        factory_path="lattice.transforms.context_selector.SubmodularContextSelector",
        factory_kwargs={"submodular_token_budget": "token_budget"},
        description="Submodular context selection with token budget",
    ),
    TransformSpec(
        canonical_name="information_theoretic_selector",
        aliases=("info_theoretic_selector",),
        config_flag="transform_context_selector",
        priority=19,
        safety_bucket=CONDITIONAL,
        factory_path="lattice.transforms.context_selector.InformationTheoreticSelector",
        factory_kwargs={"submodular_token_budget": "token_budget"},
        description="Information-theoretic context selection",
    ),
    TransformSpec(
        canonical_name="strategy_selector",
        config_flag="transform_strategy_selector",
        priority=19,
        safety_bucket=SAFE,
        factory_path="lattice.transforms.strategy_selector.StrategySelector",
        description="Bandit-based strategy selection",
    ),
    # Phase 4 — substitution & compression
    TransformSpec(
        canonical_name="reference_sub",
        aliases=("reference_substitution",),
        config_flag="transform_reference_sub",
        priority=20,
        safety_bucket=CONDITIONAL,
        factory_path="lattice.transforms.reference_sub.ReferenceSubstitution",
        description="UUID/URL/hash substitution with short aliases",
    ),
    TransformSpec(
        canonical_name="rate_distortion",
        aliases=("semantic_compress", "rate_distortion_compressor", "semantic_compressor"),
        config_flag="transform_semantic_compress",
        priority=22,
        safety_bucket=CONDITIONAL,
        factory_path="lattice.transforms.rate_distortion.RateDistortionCompressor",
        factory_kwargs={"rate_distortion_budget": "distortion_budget"},
        description="Semantic compression via rate-distortion optimization",
    ),
    TransformSpec(
        canonical_name="grammar_compress",
        aliases=("grammar_compressor",),
        config_flag="transform_grammar_compress",
        priority=24,
        safety_bucket=CONDITIONAL,
        factory_path="lattice.transforms.grammar_compress.GrammarCompressor",
        description="Grammar-based structural compression",
    ),
    TransformSpec(
        canonical_name="format_conversion",
        config_flag="transform_format_conversion",
        priority=25,
        safety_bucket=CONDITIONAL,
        factory_path="lattice.transforms.format_conv.FormatConverter",
        description="OpenAI ↔ Anthropic message format conversion",
    ),
    TransformSpec(
        canonical_name="dictionary_compress",
        aliases=("dictionary_compressor",),
        config_flag="transform_dictionary_compress",
        priority=25,
        safety_bucket=CONDITIONAL,
        factory_path="lattice.transforms.dictionary_compress.DictionaryCompressor",
        description="Phrase-dictionary compression",
    ),
    # Phase 5 — summarization
    TransformSpec(
        canonical_name="hierarchical_summary",
        aliases=("hierarchical_summarizer",),
        config_flag="transform_hierarchical_summary",
        priority=28,
        safety_bucket=DANGEROUS,
        factory_path="lattice.transforms.hierarchical_summary.HierarchicalSummarizer",
        description="Nested structure summarization",
    ),
    # Phase 6 — tool & cleanup
    TransformSpec(
        canonical_name="tool_filter",
        aliases=("tool_output_filter",),
        config_flag="transform_tool_filter",
        priority=30,
        safety_bucket=SAFE,
        factory_path="lattice.transforms.tool_filter.ToolOutputFilter",
        description="Tool output projection / filtering",
    ),
    TransformSpec(
        canonical_name="output_cleanup",
        config_flag="transform_output_cleanup",
        priority=40,
        safety_bucket=SAFE,
        factory_path="lattice.transforms.output_cleanup.OutputCleanup",
        description="Whitespace normalization and JSON repair",
    ),
)

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_NAME_TO_SPEC: dict[str, TransformSpec] | None = None
_PRIORITY_ORDER: tuple[str, ...] | None = None


def _build_indices() -> None:
    """Build lookup indices from BUILTIN_TRANSFORMS."""
    global _NAME_TO_SPEC, _PRIORITY_ORDER
    if _NAME_TO_SPEC is not None:
        return
    name_to_spec: dict[str, TransformSpec] = {}
    for spec in BUILTIN_TRANSFORMS:
        # Canonical name
        name_to_spec[spec.canonical_name] = spec
        # Aliases
        for alias in spec.aliases:
            name_to_spec[alias] = spec
    _NAME_TO_SPEC = name_to_spec
    # Sort by priority for deterministic pipeline order
    sorted_specs = sorted(BUILTIN_TRANSFORMS, key=lambda s: s.priority)
    _PRIORITY_ORDER = tuple(s.canonical_name for s in sorted_specs)


def get_transform_spec(name: str) -> TransformSpec | None:
    """Return the :class:`TransformSpec` for *name* (canonical or alias)."""
    _build_indices()
    assert _NAME_TO_SPEC is not None
    return _NAME_TO_SPEC.get(name)


def is_transform_name_known(name: str) -> bool:
    """Return *True* if *name* is a canonical name or alias in the registry."""
    return get_transform_spec(name) is not None


def get_transform_safety_bucket(name: str) -> str:
    """Return the safety bucket for *name*.

    Unknown names default to ``DANGEROUS`` — they must be explicitly registered
    to prove safety. This prevents alias-based bypass.
    """
    spec = get_transform_spec(name)
    if spec is None:
        return DANGEROUS
    return spec.safety_bucket


def list_transform_names() -> tuple[str, ...]:
    """Return all canonical names in priority order."""
    _build_indices()
    assert _PRIORITY_ORDER is not None
    return _PRIORITY_ORDER


def list_default_pipeline_names() -> tuple[str, ...]:
    """Return canonical names of transforms in the default (non-execution) pipeline."""
    return tuple(
        s.canonical_name for s in BUILTIN_TRANSFORMS if s.default_pipeline and not s.execution_only
    )


def list_execution_only_names() -> tuple[str, ...]:
    """Return canonical names of execution-only transforms."""
    return tuple(s.canonical_name for s in BUILTIN_TRANSFORMS if s.execution_only)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def resolve_config_flag(name: str) -> str:
    """Return the config field that gates *name*.

    Raises :exc:`ValueError` for unknown names.
    """
    spec = get_transform_spec(name)
    if spec is None:
        raise ValueError(f"Unknown transform name: {name!r}")
    return spec.config_flag


def is_transform_enabled(config: Any, name: str) -> bool:
    """Check whether *name* is enabled on *config*.

    Looks up the registry to find the config flag, then reads it from *config*.
    Execution-only transforms are always considered enabled for the pipeline
    builder; the caller (``build_default_pipeline``) filters them separately.
    """
    spec = get_transform_spec(name)
    if spec is None:
        return False
    # Execution-only transforms are gated by include_execution_transforms, not config flags
    if spec.execution_only:
        return True
    # Transforms with no config flag are always enabled (e.g. delta_encoder placeholder)
    if not spec.config_flag:
        return True
    return bool(getattr(config, spec.config_flag, False))


# ---------------------------------------------------------------------------
# Pipeline builder helpers
# ---------------------------------------------------------------------------


def _load_transform_class(path: str) -> type:
    """Lazy-import a transform class from a dotted path."""
    module_name, class_name = path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def build_transform_instance(config: Any, spec: TransformSpec) -> Any:
    """Instantiate the transform described by *spec* using *config* for kwargs."""
    if not spec.factory_path:
        raise RuntimeError(
            f"Transform {spec.canonical_name!r} has no factory_path — "
            "it must be constructed manually in pipeline_factory"
        )
    cls = _load_transform_class(spec.factory_path)
    kwargs: dict[str, Any] = {}
    for config_field, kwarg_name in spec.factory_kwargs.items():
        kwargs[kwarg_name] = getattr(config, config_field)
    return cls(**kwargs)


__all__ = [
    "BUILTIN_TRANSFORMS",
    "TransformSpec",
    "SAFE",
    "CONDITIONAL",
    "DANGEROUS",
    "get_transform_spec",
    "is_transform_name_known",
    "get_transform_safety_bucket",
    "list_transform_names",
    "list_default_pipeline_names",
    "list_execution_only_names",
    "resolve_config_flag",
    "is_transform_enabled",
    "build_transform_instance",
]
