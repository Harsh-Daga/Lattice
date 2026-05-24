"""LATTICE proxy server components.

Usage::

    from lattice.proxy.server import create_app
    app = create_app()

    from lattice.proxy import HealthManager
"""

from lattice.providers.transport import DirectHTTPProvider
from lattice.proxy.health import HealthManager

__all__ = [
    "DirectHTTPProvider",
    "HealthManager",
    "create_app",
]


def create_app(*args: object, **kwargs: object):  # noqa: ANN201
    """Lazy import to avoid a gateway.compat ↔ proxy.server cycle."""
    from lattice.proxy.server import create_app as _create_app

    return _create_app(*args, **kwargs)
