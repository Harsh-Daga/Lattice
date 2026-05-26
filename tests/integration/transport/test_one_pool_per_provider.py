"""Steady-state: one httpx.AsyncClient per provider."""

from __future__ import annotations

import gc

import httpx

from lattice.transport import ConnectionPoolManager, ProviderRegistry


def test_one_pool_per_provider() -> None:
    pool = ConnectionPoolManager(http2=False)
    registry = ProviderRegistry()
    for adapter in registry.adapters:
        pool.get_client(adapter.name, "https://example.com")
    clients = pool.clients_by_provider()
    assert len(clients) == len(registry.adapters)
    for client in clients.values():
        assert isinstance(client, httpx.AsyncClient)


def test_gc_one_client_per_adapter_name() -> None:
    pool = ConnectionPoolManager(http2=False)
    registry = ProviderRegistry()
    before = {id(o) for o in gc.get_objects() if isinstance(o, httpx.AsyncClient)}
    for adapter in registry.adapters:
        pool.get_client(adapter.name, "")
    after = {id(o) for o in gc.get_objects() if isinstance(o, httpx.AsyncClient)}
    assert len(after - before) == len(registry.adapters)
