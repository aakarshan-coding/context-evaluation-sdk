"""A small LangChain RAG app, written the way a user would, for the CLI end-to-end test.

Retrieval is a langchain-chroma `VectorStoreRetriever` over a real chromadb collection with a
deterministic embedding. The chat model is a stand-in for `ChatOpenAI`, which is not in the
test environment: like ChatOpenAI it calls the official openai SDK's `chat.completions.create`,
here over an `httpx.MockTransport`, so the hooked SDK path is what the test exercises. The
transport answers correctly only when a case's evidence is in the prompt.

Planted: with TOP_K = 2 the warranty passage ranks 4th for its question (missing, not
retrieved; `top_k=5` retrieves it). The office cat passage ranks 1st for its question (healthy).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import chromadb
import httpx
import openai
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from contexteval import _text
from contexteval.demo.app import HashEmbedding

TOP_K = 2
MODEL = "contexteval-fake"
DOCS = {
    "warranty": "Sleeping bags carry a two year warranty against manufacturing defects.",
    "dry": "How long does a sleeping bag take to dry? Hang the sleeping bag for a day on the line.",
    "wash": "How long should you wash a sleeping bag? Wash the bag on a gentle cycle for one long rinse.",
    "store": "Store the sleeping bag loose in the large sack, never compressed for long.",
    "cat": "The office cat is named Pickles and sleeps on the returns desk.",
    "hours": "Phone support is open from 7 am to 9 pm Mountain Time.",
}
CASES = [
    {"question": "How long is the warranty on a sleeping bag?", "expected_answer": "two years", "evidence": DOCS["warranty"]},
    {"question": "What is the office cat named?", "expected_answer": "Pickles", "evidence": DOCS["cat"]},
]
MISSING_QUESTION = CASES[0]["question"]
DONT_KNOW = "I do not know."


class HashEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return HashEmbedding()(texts)

    def embed_query(self, text: str) -> list[float]:
        return HashEmbedding()([text])[0]


def _respond(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    prompt = "\n".join(m["content"] for m in body["messages"])
    answer = next((c["expected_answer"] for c in CASES if _text.chunk_in_prompt(c["evidence"], prompt)), DONT_KNOW)
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
        },
    )


class StandInChatModel(BaseChatModel):
    """Stands in for ChatOpenAI: forwards the messages to the openai SDK client it was built with."""

    client: Any

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        reply = self.client.chat.completions.create(model=MODEL, messages=[{"role": roles[m.type], "content": m.content} for m in messages])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply.choices[0].message.content))])

    @property
    def _llm_type(self) -> str:
        return "stand-in-openai"


def build_vectorstore() -> Chroma:
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    store = Chroma(client=client, collection_name=f"lc_rag_{uuid.uuid4().hex}", embedding_function=HashEmbeddings())
    store.add_texts(list(DOCS.values()), ids=list(DOCS))
    return store


vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm = StandInChatModel(client=openai.OpenAI(api_key="fake", http_client=httpx.Client(transport=httpx.MockTransport(_respond))))


def answer(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    reply = llm.invoke([SystemMessage(content="Answer from the context only."), HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")])
    return str(reply.content)
