"""Connection pool HTTP/2 configuration."""

from __future__ import annotations

import gc
from unittest.mock import MagicMock, patch

import httpx

from lattice.transport import ConnectionPoolManager, ProviderRegistry
from lattice.transport.pool import ConnectionPoolManager as PoolCls


def test_one_client_per_provider() -> None:
    pool = ConnectionPoolManager(http2=False)
    c1 = pool.get_client("openai", "https://api.openai.com")
    c2 = pool.get_client("openai", "https://other.example.com")
    assert c1 is c2
    assert pool.pool_count == 1


def test_http_version_metadata() -> None:
    pool = ConnectionPoolManager(http2=True)
    pool.get_client("anthropic", "https://api.anthropic.com")
    ver = pool.get_http_version("anthropic", "")
    assert ver in ("http/2", "http/1.1")


def test_pool_requests_http2_for_openai_and_anthropic() -> None:
    pool = PoolCls(http2=True)
    created: list[dict[str, object]] = []

    def _fake_client(**kwargs: object) -> MagicMock:
        created.append(kwargs)
        return MagicMock()

    with patch("lattice.transport.pool.httpx.AsyncClient", side_effect=_fake_client):
        pool.get_client("openai", "https://api.openai.com")
        pool.get_client("anthropic", "https://api.anthropic.com")

    assert len(created) == 2
    for kwargs in created:
        assert kwargs.get("http2") is True


def test_one_pool_per_provider_gc() -> None:
    pool = ConnectionPoolManager(http2=False)
    registry = ProviderRegistry()
    before = {id(o) for o in gc.get_objects() if isinstance(o, httpx.AsyncClient)}
    for adapter in registry.adapters:
        pool.get_client(adapter.name, "")
    after = {id(o) for o in gc.get_objects() if isinstance(o, httpx.AsyncClient)}
    new_clients = after - before
    assert len(new_clients) == len(registry.adapters)
