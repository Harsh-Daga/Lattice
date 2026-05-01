"""Python version compatibility shims and backports.

All shims are centralized here so the rest of the codebase can assume
modern Python features without conditional imports scattered everywhere.

Since LATTICE requires Python 3.10+, we only provide backports for:
- `tomllib` (stdlib 3.11+; backported via tomli for 3.10)
- `typing.Self` (3.11+)

When the minimum supported version is bumped, remove the relevant shim.
"""

import sys

from typing_extensions import ParamSpec as ParamSpec
from typing_extensions import Self as Self

PY310 = sys.version_info >= (3, 10)
PY311 = sys.version_info >= (3, 11)
PY312 = sys.version_info >= (3, 12)

# tomllib is stdlib 3.11+; for 3.10 we use tomli (declared in deps)
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

__all__ = [
    "PY310",
    "PY311",
    "PY312",
    "tomllib",
    "Self",
    "ParamSpec",
]
