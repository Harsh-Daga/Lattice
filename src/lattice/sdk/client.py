"""DEPRECATED: import LatticeClient from `lattice` directly.

In v1.1 this module will be removed.
"""

from __future__ import annotations

import warnings

from lattice.client import CompressResult, LatticeClient

warnings.warn(
    "lattice.sdk.client is deprecated; import LatticeClient from `lattice` "
    "or `lattice.sdk` instead. This module will be removed in v1.1.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LatticeClient", "CompressResult"]
