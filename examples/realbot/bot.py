"""A plain RAG support bot for Ridgeline Cycles, written the way an app team would write it.

chromadb persistent collection, OpenAI embeddings, paragraph chunks of about 500 characters,
four results per question, one gpt-4.1-mini call. It does not import contexteval.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
from pathlib import Path

import chromadb
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

CORPUS_DIR = Path(__file__).parent / "corpus"
DB_DIR = Path(os.environ.get("REALBOT_DB", ".realbot_chroma"))
CHUNK_CHARS = 500
N_RESULTS = 4
MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = (
    "You are the customer support assistant for Ridgeline Cycles, a bike shop. "
    "Answer the customer's question using the context below. "
    "If the context does not contain the answer, say you don't know."
)


def chunk_document(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Greedy paragraph packing: consecutive paragraphs share a chunk until it would pass
    `limit` characters. A paragraph longer than `limit` is its own chunk."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + 2 + len(paragraph) > limit:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def corpus_chunks() -> dict[str, str]:
    """id -> text, ids "<file>#<n>"."""
    return {
        f"{path.name}#{n}": chunk
        for path in sorted(CORPUS_DIR.glob("*.md"))
        for n, chunk in enumerate(chunk_document(path.read_text(encoding="utf-8")))
    }


@functools.lru_cache(maxsize=None)
def collection() -> chromadb.Collection:
    chunks = corpus_chunks()
    digest = hashlib.sha256(repr(sorted(chunks.items())).encode("utf-8")).hexdigest()[:12]
    client = chromadb.PersistentClient(path=str(DB_DIR), settings=chromadb.Settings(anonymized_telemetry=False))
    embed = OpenAIEmbeddingFunction(api_key_env_var="OPENAI_API_KEY", model_name=EMBEDDING_MODEL)
    col = client.get_or_create_collection(f"ridgeline_{digest}", embedding_function=embed)
    if col.count() != len(chunks):
        col.upsert(ids=list(chunks), documents=list(chunks.values()))
    return col


@functools.lru_cache(maxsize=None)
def llm() -> openai.OpenAI:
    return openai.OpenAI()


def answer(question: str) -> str:
    hits = collection().query(query_texts=[question], n_results=N_RESULTS)
    context = "\n\n---\n\n".join(hits["documents"][0])
    reply = llm().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return reply.choices[0].message.content or ""
