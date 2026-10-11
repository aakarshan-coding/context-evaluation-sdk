# Changelog

This project follows [Semantic Versioning](https://semver.org/). Each release lists what changed for people who use contexteval.

## Unreleased

### Added

- LangChain and LlamaIndex apps work without changes. contexteval hooks LangChain `BaseRetriever.invoke` and `ainvoke` and LlamaIndex `BaseRetriever.retrieve` and `aretrieve`. Captured chunks, `top_k` experiments (through `search_kwargs["k"]` or `similarity_top_k`, set per call and restored), and diagnosis replays work as for chromadb, and injected evidence arrives as a `Document` or a `NodeWithScore`. New extras `langchain` and `llamaindex`.
- `contexteval sweep` picks experiment settings from the labels of the latest diagnosed run: larger `top_k` for `missing` chunks that were never retrieved, `rerank=lexical`, a smaller `max_chunks`, or `compress` for `irrelevant`, and `order=ends`, `order=reverse`, or `compress` for `ignored`. It says which labels no setting can fix (`dropped`, `not_context`, `flaky`) and why. It tunes on a seeded split of the cases, confirms the winner on the held-out rest, and ends with one plain recommendation, such as "Set top_k=8. Accuracy 93% -> 99% (holdout 90% -> 100%). Uses 88% more input tokens". `--rounds`, `--max-settings`, `--holdout`, and `--seed` control the search; the cost prompt states the worst case.
- The dashboard lists sweeps among the experiments, labels each row as tune or holdout, and marks the recommended row.

### Changed

- The project database moves to schema version 2 (experiments carry a kind and a sweep report). A project created by 0.2.0 is upgraded in place the first time any command opens it.

### Fixed

- A retrieval that reaches a hooked store from inside another hooked retrieval (langchain-chroma over chromadb, or a `@contexteval.retriever` function that queries chromadb) is recorded once, not twice, and an experiment setting applies to it once.

## 0.2.0 (v1.1, 2026-10-11)

### Changed

- Diagnosis replays take a majority over 3 samples per step by default. Set the count with `contexteval diagnose --samples N` (odd, at least 1). A step stops sampling once a majority is decided, so a single lucky pass no longer decides a label.
- Stored diagnoses record the pass count of each replay step. The CLI and the dashboard show them, for example "gold to top 2/3".

### Added

- Async apps work. contexteval hooks `AsyncOpenAI` chat completions and responses, `AsyncAnthropic` messages, and chromadb's async collection query. `async def` functions work under `@contexteval.retriever`, and `contexteval run` accepts an `async def answer` entry point.
- Interventions and experiment settings apply to async retrievals exactly as to sync ones.
- Production capture with `contexteval.init()` works per asyncio task.
- Published on PyPI as `contexteval`, with complete package metadata.
- A GitHub Actions workflow runs the test suite on Python 3.10 to 3.13 on every push and pull request.
- A release workflow builds the package and publishes it to PyPI when a `v*` tag is pushed, through PyPI trusted publishing. No API token is stored in the repository.

## 0.1.0 (v1, not published to PyPI)

The first working version, installed from a checkout.

### Added

- `contexteval init`, `build-tests`, `run`, `diagnose`, `experiment`, and `ui` commands.
- Capture without app changes: hooks for chromadb `Collection.query`, pinecone `Index.query`, OpenAI chat completions and responses, and Anthropic messages. `@contexteval.retriever` marks any other retrieval function.
- A test set builder that generates question, answer, and evidence cases from `.md` and `.txt` documents, and keeps only cases whose evidence appears word for word. JSONL and CSV import for your own cases.
- Grading by normalized substring match, with an LLM judge as fallback.
- Diagnosis that labels each failure `missing` (`not_retrieved` or `dropped`), `irrelevant`, `ignored`, `not_context`, or `flaky` by replaying the case with changed context. Cases without evidence and production traces go to an LLM judge.
- Experiments over `top_k`, `rerank`, `order`, `max_chunks`, and `compress`, with a baseline row and a comparison table.
- A local dashboard bound to `127.0.0.1`.
- Production capture through `contexteval.init()`, stored as ungraded traces.
- A bundled demo app with planted failures of every label, runnable offline with `--provider fake`.
- Local SQLite storage, a cost estimate before every command that calls an LLM, and no telemetry.
