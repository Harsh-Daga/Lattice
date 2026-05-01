"""Client SDK for LATTICE.

Exports
-------
* ``LatticeClient`` — compression + direct transport
* ``LatticeProxyClient`` — HTTP client for proxy mode
* ``wrap_openai`` / ``wrap_anthropic`` — provider SDK wrappers
"""

from lattice.client import LatticeClient
from lattice.sdk.proxy_client import LatticeProxyClient
from lattice.sdk.wrappers import wrap_anthropic, wrap_openai

__all__ = [
    "LatticeClient",
    "LatticeProxyClient",
    "wrap_anthropic",
    "wrap_openai",
]
