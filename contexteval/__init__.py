"""contexteval: find out why your RAG agent's context failed.

Public surface is deliberately two names for app code plus read-only types:

    import contexteval
    contexteval.init()                      # hook chromadb, pinecone, langchain, llama_index, openai, anthropic

    @contexteval.retriever(top_k="k")       # custom retrieval (pgvector, SQL, APIs)
    def search(query: str, k: int = 5) -> list[dict]: ...

    def build(chunking: contexteval.Chunking):  # optional, next to answer: rebuild the index
        chunks = contexteval.chunk_text(doc, chunking.with_defaults(chunk_size=500), title=name)
        ...
        return answer

Everything else is driven by the `contexteval` CLI. Modules other than this one and
`contexteval.types` are internal; their names may change without notice.
"""

from ._capture import InstallReport, retriever
from ._hooks import init
from ._intervention import InterventionNotApplicable
from ._text import chunk_text
from .types import Chunking

__all__ = ["init", "retriever", "InstallReport", "InterventionNotApplicable", "Chunking", "chunk_text"]
