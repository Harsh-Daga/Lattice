"""Phase 2a regression — wire types live at lattice.transport.types.

Verifies the canonical module path after the Phase 2a move of
``core/transport.py`` → ``transport/types.py``, and that the
``lattice.core`` re-export buffer still resolves to the same objects.
"""

from __future__ import annotations

from lattice.transport.types import Message, Request, Response, Role


def test_canonical_module() -> None:
    assert Request.__module__ == "lattice.transport.types"
    assert Response.__module__ == "lattice.transport.types"
    assert Message.__module__ == "lattice.transport.types"


def test_core_reexport_is_same_object() -> None:
    from lattice.core import Message as CoreMessage
    from lattice.core import Request as CoreRequest
    from lattice.core import Response as CoreResponse
    from lattice.core import Role as CoreRole

    assert CoreRequest is Request
    assert CoreResponse is Response
    assert CoreMessage is Message
    assert CoreRole is Role


def test_transport_package_reexport() -> None:
    from lattice.transport import Message as TPMessage
    from lattice.transport import Request as TPRequest
    from lattice.transport import Response as TPResponse

    assert TPRequest is Request
    assert TPResponse is Response
    assert TPMessage is Message
