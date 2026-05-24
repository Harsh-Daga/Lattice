"""Deprecated lattice.sdk.client import path."""

from __future__ import annotations

import importlib
import sys
import warnings


def test_sdk_client_warns() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sys.modules.pop("lattice.sdk.client", None)
        importlib.import_module("lattice.sdk.client")
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected DeprecationWarning from lattice.sdk.client"
