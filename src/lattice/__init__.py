"""LATTICE — LLM Transport & Efficiency Layer.

Make LLM calls cheaper, faster, and smarter — without changing the model.
"""

from lattice._version import __version__
from lattice.cache import SemanticCache
from lattice.client import CompressResult, LatticeClient
from lattice.ir import PromptIR, PromptIRV2, build_ir, serialize_ir_to_text
from lattice.safety import SemanticRiskScore, compute_risk_score
from lattice.sdk.proxy_client import LatticeProxyClient
from lattice.sdk.wrappers import wrap_openai as wrap_openai_client
from lattice.state import SegmentStore, Session, SessionManager
from lattice.telemetry import DowngradeCategory, MetricsCollector

__all__ = [
    "__version__",
    # SDK clients
    "LatticeClient",
    "LatticeProxyClient",
    "CompressResult",
    "wrap_openai_client",
    # IR types (power users)
    "PromptIR",
    "PromptIRV2",
    "build_ir",
    "serialize_ir_to_text",
    # Phase 9 top-level conveniences
    "MetricsCollector",
    "DowngradeCategory",
    "Session",
    "SessionManager",
    "SegmentStore",
    "SemanticCache",
    "SemanticRiskScore",
    "compute_risk_score",
]
