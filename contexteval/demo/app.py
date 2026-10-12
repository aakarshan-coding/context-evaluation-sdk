"""Bundled demo RAG app: real chromadb + real openai SDK, written the way a user would.

    contexteval run contexteval.demo.app:answer

It uses only public contexteval API (none, in fact: the CLI's hooks capture it). The app LLM is
the real OpenAI client. CONTEXTEVAL_DEMO_LLM=fake (the default) routes it to an in-process
httpx.MockTransport backed by demo.fake_llm; =openai talks to the real API (DoD item 5).

Planted mechanics (see planted.py for which case uses which):
- TOP_K = 4 and a deterministic hashed bag-of-words embedding, so a gold chunk authored to
  rank 5th to 10th is not retrieved at baseline and is retrieved at top_k=10 (DoD item 3).
- The app drops chunks containing "[ARCHIVED]" after retrieval, like a real staleness filter.
  A gold chunk carrying that marker is retrieved but dropped (missing/dropped).
- The app's own chunking is one chunk per paragraph, so an evidence passage that spans two
  paragraphs (corpus/fitting.md) is split across two chunks; `build(chunking)` rebuilds the index
  with another chunking, and a chunk_size that packs the two paragraphs together fixes it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import tempfile
import zlib
from pathlib import Path
from typing import Any

from chromadb.api.types import EmbeddingFunction

from .. import _text as text
from .._store import PROJECT_DIR_NAME, find_project
from ..types import Chunking
from ._lock import file_lock

TOP_K = 4
ARCHIVED_MARKER = "[ARCHIVED]"
MODEL = "gpt-4.1-mini"
FAKE_MODEL = "contexteval-fake"  # unpriced, so fake runs show no cost
CHUNK_OPEN = "<chunk>"
CHUNK_CLOSE = "</chunk>"
CORPUS_DIR = Path(__file__).parent / "corpus"
COLLECTION_NAME = "contexteval_demo"
_INSTRUCTIONS = "Answer the question using only the context below."


class HashEmbedding(EmbeddingFunction):  # type: ignore[type-arg]
    """chromadb EmbeddingFunction. Deterministic, offline: no ONNX model download.

    Bag of words hashed with crc32 into DIM buckets, L2-normalized, so cosine distance is a
    pure function of shared tokens and the same across processes (hash() is salted per run).
    """

    DIM = 64

    def __init__(self) -> None:  # chromadb warns when an embedding function has no __init__
        pass

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = []
        for doc in input:
            v = [0.0] * self.DIM
            for tok in text.tokens(doc):
                v[zlib.crc32(tok.encode("utf-8")) % self.DIM] += 1.0
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vectors.append([x / norm for x in v])
        return vectors

    @staticmethod
    def name() -> str:
        return "contexteval_hash"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashEmbedding":
        return HashEmbedding()


def corpus_chunks(chunking: Chunking = Chunking()) -> dict[str, str]:
    """id -> text, ids "<file>#<n>". The demo's own chunking is one chunk per non-heading
    paragraph of corpus/*.md, which is chunk_text at chunk_size=1 (a paragraph longer than the
    limit is its own chunk). The document's heading is its title for `title=on`."""
    resolved = chunking.with_defaults(chunk_size=1, overlap=0, title=False)
    chunks = {}
    for path in sorted(CORPUS_DIR.glob("*.md")):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")) if p.strip()]
        title = next((p.lstrip("#").strip() for p in paragraphs if p.startswith("#")), path.stem)
        body = "\n\n".join(p for p in paragraphs if not p.startswith("#"))
        for n, chunk in enumerate(text.chunk_text(body, resolved, title=title)):
            chunks[f"{path.name}#{n}"] = chunk
    return chunks


def data_dir() -> Path:
    """Where the demo keeps state: <project>/.contexteval for the project around the cwd, else
    a temp directory, so the demo never creates a project the user did not ask for."""
    project = find_project(Path.cwd())
    if project is None:
        return Path(tempfile.gettempdir()) / "contexteval_demo"
    return project / PROJECT_DIR_NAME


def collection(chunking: Chunking = Chunking()) -> Any:
    """The demo collection, indexed from corpus/*.md into data_dir()/demo_chroma."""
    return _collection_at(str(data_dir() / "demo_chroma"), chunking)


@functools.lru_cache(maxsize=None)
def _collection_at(path: str, chunking: Chunking = Chunking()) -> Any:
    # Persistent, not ephemeral: chromadb 1.5's in-memory client drops an older collection's
    # vector index once enough other collections are opened in the process (count() still
    # reports every row, queries return none). A persisted index is reloaded instead.
    #
    # Concurrent CLI processes share this directory. chromadb's first-time setup is not safe
    # across processes ("table collections already exists"), and a write in one process can
    # fail another's query ("database is locked"). So one process builds the index under a
    # lock and no process writes after that. The corpus digest in the name keeps a corpus edit
    # from reusing a stale index.
    import chromadb

    chunks = corpus_chunks(chunking)
    digest = hashlib.sha256(json.dumps(chunks, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    with file_lock(Path(path + ".lock")):
        client = chromadb.PersistentClient(path=path, settings=chromadb.Settings(anonymized_telemetry=False))
        col = client.get_or_create_collection(f"{COLLECTION_NAME}_{digest}", embedding_function=HashEmbedding())
        if col.count() != len(chunks):
            col.upsert(ids=list(chunks), documents=list(chunks.values()))
    return col


def build_prompt(question: str, chunks: list[str]) -> str:
    """System instructions, then each chunk wrapped in CHUNK_OPEN/CHUNK_CLOSE in order, then the question."""
    context = "\n".join(f"{CHUNK_OPEN}{c}{CHUNK_CLOSE}" for c in chunks)
    return f"{_INSTRUCTIONS}\n\n{context}\n\nQuestion: {question}"


_CHUNK = re.compile(re.escape(CHUNK_OPEN) + r"(.*?)" + re.escape(CHUNK_CLOSE), re.DOTALL)


def parse_prompt(prompt: str) -> tuple[str, list[str]]:
    """Inverse of build_prompt: (question, chunks in order). Used by fake_llm only."""
    _, _, question = prompt.rpartition("\n\nQuestion: ")
    return question.strip(), _CHUNK.findall(prompt)


def _answer(col: Any, question: str) -> str:
    res = col.query(query_texts=[question], n_results=TOP_K)
    chunks = [d for d in res["documents"][0] if ARCHIVED_MARKER not in d]
    resp = _client().chat.completions.create(model=_model(), messages=[{"role": "user", "content": build_prompt(question, chunks)}])
    return resp.choices[0].message.content


def answer(question: str) -> str:
    return _answer(collection(), question)


def build(chunking: Chunking) -> Any:
    """The optional builder contexteval looks for beside `answer`: the same app over an index
    chunked with `chunking`. The collection name carries the chunks' digest, so each chunking has
    its own collection and the app's own one is untouched."""
    return functools.partial(_answer, collection(chunking))


def _model() -> str:
    return MODEL if os.environ.get("CONTEXTEVAL_DEMO_LLM", "fake") == "openai" else FAKE_MODEL


def _client() -> Any:
    return _client_for(os.environ.get("CONTEXTEVAL_DEMO_LLM", "fake"))


@functools.lru_cache(maxsize=None)
def _client_for(mode: str) -> Any:
    import openai

    if mode == "openai":
        return openai.OpenAI()
    if mode != "fake":
        raise ValueError(f"CONTEXTEVAL_DEMO_LLM must be 'fake' or 'openai', got {mode!r}")
    import httpx

    from . import fake_llm

    return openai.OpenAI(api_key="fake", http_client=httpx.Client(transport=fake_llm.transport()))
