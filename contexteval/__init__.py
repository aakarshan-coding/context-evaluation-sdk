"""contexteval: find out why your RAG agent's context failed.

Public surface is deliberately two names for app code plus read-only types:

    import contexteval
    contexteval.init()                      # hook chromadb, pinecone, langchain, llama_index, openai, anthropic

    @contexteval.retriever(top_k="k")       # custom retrieval (pgvector, SQL, APIs)
    def search(query: str, k: int = 5) -> list[dict]: ...

Everything else is driven by the `contexteval` CLI. Modules other than this one and
`contexteval.types` are internal; their names may change without notice.
"""

from ._capture import InstallReport, retriever
from ._hooks import init
from ._intervention import InterventionNotApplicable

__all__ = ["init", "retriever", "InstallReport", "InterventionNotApplicable"]
