from __future__ import annotations

from lattice.ir.builder.analyze import is_repeated_template
from lattice.ir.builder.core import build_ir, compile_request_ir
from lattice.ir.builder.partition import *  # noqa: F403

__all__ = ["build_ir", "compile_request_ir", "is_repeated_template"]
