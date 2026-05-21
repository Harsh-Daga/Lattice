"""Protocol-level transport: wire types, serialization, congestion, delta wire.

NOTE: HTTP transport (DirectHTTPProvider, ConnectionPoolManager) lives in
``providers/transport.py``. This package is for protocol concerns above the
HTTP layer — what a request *is* (Message/Request/Response), how it
serializes, and how multiple in-flight requests share a provider link.
"""

from lattice.transport.congestion import ProviderCongestionState, TACCController
from lattice.transport.delta_wire import (
    DeltaWireDecoder,
    DeltaWireEncoder,
    compute_wire_savings,
    delta_wire_bytes,
)
from lattice.transport.serialization import (
    message_from_dict,
    message_to_dict,
    request_from_dict,
    request_to_dict,
    response_to_dict,
)
from lattice.transport.simulation import (
    SimulationConfig,
    SimulationMetrics,
    run_static_concurrency_simulation,
    run_tacc_simulation,
)
from lattice.transport.types import (
    Message,
    Request,
    Response,
    Role,
    SyncTransform,
    Transform,
)

__all__ = [
    # wire types (moved from core/transport.py in Phase 2a)
    "Message",
    "Request",
    "Response",
    "Role",
    "Transform",
    "SyncTransform",
    # serialization
    "message_to_dict",
    "message_from_dict",
    "request_to_dict",
    "request_from_dict",
    "response_to_dict",
    # delta wire
    "DeltaWireDecoder",
    "DeltaWireEncoder",
    "delta_wire_bytes",
    "compute_wire_savings",
    # congestion / simulation
    "ProviderCongestionState",
    "TACCController",
    "SimulationConfig",
    "SimulationMetrics",
    "run_static_concurrency_simulation",
    "run_tacc_simulation",
]
