"""LATTICE SDK — client surfaces for using LATTICE from Python code.

For most users:
    from lattice import LatticeClient, LatticeProxyClient, wrap_openai_client

This module re-exports the same names. Importing from ``lattice.sdk.client`` is
deprecated in v1.0.0 and will be removed in v1.1.
"""

from lattice.client import CompressResult, LatticeClient
from lattice.sdk.proxy_client import LatticeProxyClient
from lattice.sdk.wrappers import wrap_anthropic, wrap_openai

wrap_openai_client = wrap_openai

__all__ = [
    "LatticeClient",
    "CompressResult",
    "LatticeProxyClient",
    "wrap_openai",
    "wrap_openai_client",
    "wrap_anthropic",
]
