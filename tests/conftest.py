"""Shared fixtures. No test touches the network: the SDK LLM is FakeProvider, the app LLM is
the demo's httpx.MockTransport, chromadb runs ephemeral with HashEmbedding and telemetry off,
pinecone is a stand-in class patched through _hooks.patch_method.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Iterator

import pytest

from contexteval._llm import FakeProvider
from contexteval._store import Store

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _loopback_only(address: object) -> bool:
    return isinstance(address, tuple) and bool(address) and str(address[0]) in _LOOPBACK


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """socket.socket.connect raises for any non-loopback address, so a leak fails loudly."""

    def guarded_connect(self: socket.socket, address: object) -> None:
        if not _loopback_only(address):
            raise RuntimeError(f"network access blocked in tests: connect({address!r})")
        _real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: object) -> int:
        if not _loopback_only(address):
            raise RuntimeError(f"network access blocked in tests: connect_ex({address!r})")
        return _real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """tmp project dir with .contexteval/, chdir'd into; CONTEXTEVAL_DEMO_LLM=fake."""
    Store.open(tmp_path).close()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CONTEXTEVAL_DEMO_LLM", "fake")
    return tmp_path


@pytest.fixture
def store(project: Path) -> Iterator[Store]:
    s = Store.open(project)
    yield s
    s.close()


@pytest.fixture
def fake() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def hooks() -> Iterator[None]:
    """install("cli"), yield, _hooks.uninstall_all() and reset capture._process."""
    from contexteval import _capture, _hooks

    _hooks.install("cli")
    yield
    _hooks.uninstall_all()
    _capture.reset_process()
