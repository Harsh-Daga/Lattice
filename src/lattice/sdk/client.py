"""Backward-compatible SDK re-exports.

The canonical SDK client now lives in ``lattice.client``.  This module
re-exports it so that existing imports like::

    from lattice.sdk import LatticeClient

continue to work.
"""

from __future__ import annotations

from lattice.client import CompressResult, LatticeClient

__all__ = ["LatticeClient", "CompressResult"]
