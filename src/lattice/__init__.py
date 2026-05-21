"""LATTICE — LLM Transport & Efficiency Layer.

Make LLM calls cheaper, faster, and smarter — without changing the model.
"""

from lattice._version import __version__
from lattice.client import CompressResult, LatticeClient
from lattice.ir import PromptIR, PromptIRV2, build_ir, serialize_ir_to_text
from lattice.sdk.proxy_client import LatticeProxyClient
from lattice.sdk.wrappers import wrap_openai as wrap_openai_client

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
]
