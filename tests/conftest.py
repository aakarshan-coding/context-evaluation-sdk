"""Shared fixtures. No test touches the network: the SDK LLM is FakeProvider, the app LLM is
the demo's httpx.MockTransport, chromadb runs ephemeral with HashEmbedding and telemetry off,
pinecone is a stand-in class patched through _hooks.patch_method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from contexteval._llm import FakeProvider
from contexteval._store import Store


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """socket.socket.connect raises for any non-loopback address, so a leak fails loudly."""
    raise NotImplementedError


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """tmp project dir with .contexteval/, chdir'd into; CONTEXTEVAL_DEMO_LLM=fake."""
    raise NotImplementedError


@pytest.fixture
def store(project: Path) -> Store:
    return Store.open(project)


@pytest.fixture
def fake() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def hooks() -> Iterator[None]:
    """install("cli"), yield, _hooks.uninstall_all() and reset capture._process."""
    raise NotImplementedError
