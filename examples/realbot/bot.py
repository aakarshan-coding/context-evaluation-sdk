"""A plain RAG support bot for Ridgeline Cycles, written the way an app team would write it.

chromadb persistent collection, OpenAI embeddings, paragraph chunks of about 500 characters,
four results per question, one gpt-4.1-mini call. It imports contexteval only for the optional
`build(chunking)` beside `answer`, which lets `contexteval experiment` and `sweep` try other
chunkings; `answer` itself needs nothing from it.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

import chromadb
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import contexteval

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


def doc_title(text: str) -> str:
    """The document's first `#` heading, for chunks built with title on."""
    first = text.lstrip().split("\n", 1)[0]
    return first.lstrip("#").strip() if first.startswith("#") else ""


def corpus_chunks(chunking: contexteval.Chunking = contexteval.Chunking()) -> dict[str, str]:
    """id -> text, ids "<file>#<n>". The bot's own chunking is greedy paragraph packing up to
    CHUNK_CHARS with no overlap and no title; chunk_text reproduces it at those values."""
    resolved = chunking.with_defaults(chunk_size=CHUNK_CHARS, overlap=0, title=False)
    chunks = {}
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for n, chunk in enumerate(contexteval.chunk_text(text, resolved, title=doc_title(text))):
            chunks[f"{path.name}#{n}"] = chunk
    return chunks


@functools.lru_cache(maxsize=None)
def collection(chunking: contexteval.Chunking = contexteval.Chunking()) -> chromadb.Collection:
    chunks = corpus_chunks(chunking)
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


def _answer(col: chromadb.Collection, question: str) -> str:
    hits = col.query(query_texts=[question], n_results=N_RESULTS)
    context = "\n\n---\n\n".join(hits["documents"][0])
    reply = llm().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return reply.choices[0].message.content or ""


def answer(question: str) -> str:
    return _answer(collection(), question)


def build(chunking: contexteval.Chunking):
    """The optional builder contexteval looks for beside `answer`: the same bot over a collection
    chunked with `chunking`. Each chunking gets its own collection, named by its chunks' digest."""
    return functools.partial(_answer, collection(chunking))
