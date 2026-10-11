"""Bundled demo RAG app: real chromadb + real openai SDK, written the way a user would.

    contexteval run contexteval.demo.app:answer

It uses only public contexteval API (none, in fact: the CLI's hooks capture it). The app LLM is
the real OpenAI client. CONTEXTEVAL_DEMO_LLM=fake (the default) routes it to an in-process
httpx.MockTransport backed by demo.fake_llm; =openai talks to the real API (DoD item 5).

Planted mechanics (see planted.py for which case uses which):
- TOP_K = 3 and a deterministic hashed bag-of-words embedding, so a gold chunk authored to
  rank 5th is not retrieved at baseline and is retrieved at top_k=10 (DoD item 3).
- The app drops chunks containing "[ARCHIVED]" after retrieval, like a real staleness filter.
  A gold chunk carrying that marker is retrieved but dropped (missing/dropped).
"""

from __future__ import annotations

import zlib
from typing import Any

from chromadb.api.types import EmbeddingFunction

from .. import _text as text

TOP_K = 3
ARCHIVED_MARKER = "[ARCHIVED]"
MODEL = "gpt-4o-mini"
CHUNK_OPEN = "<chunk>"
CHUNK_CLOSE = "</chunk>"


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


def collection() -> Any:
    """Ephemeral chromadb client (telemetry off), built once per process from corpus/*.md,
    one chunk per paragraph, ids "<file>#<n>".
    """
    raise NotImplementedError


def build_prompt(question: str, chunks: list[str]) -> str:
    """System instructions, then each chunk wrapped in CHUNK_OPEN/CHUNK_CLOSE in order, then the question."""
    raise NotImplementedError


def parse_prompt(prompt: str) -> tuple[str, list[str]]:
    """Inverse of build_prompt: (question, chunks in order). Used by fake_llm only."""
    raise NotImplementedError


def answer(question: str) -> str:
    # res = collection().query(query_texts=[question], n_results=TOP_K)
    # chunks = [d for d in res["documents"][0] if ARCHIVED_MARKER not in d]
    # resp = _client().chat.completions.create(model=MODEL, messages=[{"role": "user", "content": build_prompt(question, chunks)}])
    # return resp.choices[0].message.content
    raise NotImplementedError


def _client() -> Any:
    raise NotImplementedError  # OpenAI() or OpenAI(api_key="fake", http_client=httpx.Client(transport=fake_llm.transport()))
