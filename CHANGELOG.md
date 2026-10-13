# Changelog

This project follows [Semantic Versioning](https://semver.org/). Each release lists what changed for people who use contexteval.

## Unreleased

### Fixed

- `contexteval sweep` proposes an overlap together with the chunk size it was measured against, as in `chunk_size=116 overlap=29`. It used to propose the overlap alone, so your `build` filled in your own chunk size, which could be smaller than the overlap; the demo's sweep listed `overlap=29` as not applicable for that reason.

## 0.4.0 (2026-10-13)

### Added

- Chunking settings for `contexteval experiment` and `contexteval sweep`: `chunk_size` (characters per chunk), `overlap` (characters carried from the end of one chunk into the next), and `title` (`on` or `off`, every chunk starts with its document's title). Chunking happens when your app builds its index, so these need an optional `build(chunking)` function beside `answer` in your app module. It takes a `contexteval.Chunking` and returns the answer function for an index built with it. `contexteval.chunk_text(text, chunking, title=None)` splits one document the way the real bot does, with overlap and an optional title, for use inside `build`; `Chunking.with_defaults` fills the knobs the experiment did not set with your app's own values. Each distinct chunking builds once per command and every case and setting that shares it reuses the build; the baseline never calls `build`. Without `build`, a chunking knob is a one-line error before any app call that says what to add.
- A diagnosis records when the evidence was found only by joining two adjacent chunks. The label line then ends with "the evidence was split across two adjacent chunks", and the dashboard marks the failure "split gold". A diagnosis stored by an earlier version reads as not split.
- `contexteval sweep` proposes chunking from the labels and that signal: a split evidence asks for `chunk_size` at twice and `overlap` at a quarter of the longest chunk the baseline delivered, and `ignored` and `irrelevant` failures ask for `title=on`. Among settings that as many failures ask for, the split signal's come first. These settings run only when the app offers `build(chunking)`; otherwise the sweep says how many failures pointed at chunking and what to add, and goes on with the retrieval settings. A `build` that refuses a chunking is listed as not applicable, as a refused `top_k` is. The experiment header counts the index builds a grid needs.
- The demo app and `examples/realbot/bot.py` offer `build(chunking)`. The demo has a ninth planted case whose evidence spans two paragraphs of `corpus/fitting.md`; it is labeled `not_context` with the split signal, and the demo sweep now recommends `chunk_size=232 title=on`.

### Changed

- `contexteval sweep` breaks near-ties by the number of changed knobs. Input tokens within 5% of the cheapest contender's count as equal; among those the setting that passed more wins, then the one that changes fewer knobs, then the faster one. A combination such as `order=reverse title=on` no longer beats `title=on` alone by one input token when the reorder adds nothing. With the default holdout the demo sweep now recommends `title=on`.
- When the sweep keeps the current settings, its last line is one sentence, such as "Keep the current settings, because no candidate beat the baseline on the tune cases (...)", instead of a second sentence that started in lowercase.
- `contexteval --help` describes diagnosis, experiments, the sweep, and chunking, and the `experiment` and `sweep` help lines name chunking settings.

## 0.3.0 (v2, 2026-10-11)

### Added

- LangChain and LlamaIndex apps work without changes. contexteval hooks LangChain `BaseRetriever.invoke` and `ainvoke` and LlamaIndex `BaseRetriever.retrieve` and `aretrieve`. Captured chunks, `top_k` experiments (through `search_kwargs["k"]` or `similarity_top_k`, set per call and restored), and diagnosis replays work as for chromadb, and injected evidence arrives as a `Document` or a `NodeWithScore`. New extras `langchain` and `llamaindex`.
- `contexteval sweep` picks experiment settings from the labels of the newest baseline run with diagnoses, or of `--run ID`, and names the run it read: larger `top_k` for `missing` chunks that were never retrieved, `rerank=lexical`, a smaller `max_chunks`, or `compress` for `irrelevant`, and `order=ends`, `order=reverse`, or `compress` for `ignored`. It says which labels no setting can fix (`dropped`, `not_context`, `flaky`) and why. It tunes on a seeded split of the cases, confirms the winner on the held-out rest, and ends with one plain recommendation, such as "Set top_k=8. Accuracy 93% -> 99% (holdout 90% -> 100%). Uses 88% more input tokens". `--rounds`, `--max-settings`, `--holdout`, and `--seed` control the search; the cost prompt states the worst case.
- `contexteval sweep` prefers the cheaper setting among near-ties. Every setting that beat the baseline and passed within a near-tie window of the most accurate one contends, and the one with the fewest input tokens wins, then the most passes, then the lowest latency. When the source run recorded no input tokens the window is 0, and the sweep says so. The window is 5% of the tune cases, rounded down (0 cases under 20, 1 case on 35), and the sweep prints it when it is not 0; `--tolerance N` sets it to N cases. The recommendation names the more accurate setting it passed over and its cost, such as "Set top_k=8 (top_k=12 passed 1 more tune case but uses 50% more input tokens)". A window of 0 picks the most accurate setting.
- `contexteval sweep` keeps its holdout unseen and its runs safe. It splits the cases before it counts labels, so only the tune cases' labels choose the candidates. It stops before any app call when `--holdout` rounds to 0 cases, and tells you to raise it or pass `--holdout 0`. A later round builds only on a setting that beat the baseline. A setting your app's retrieval cannot take is skipped, along with later settings on the same knob, and listed as `not applicable`. The report is saved even when a sweep stops early, and the dashboard shows an experiment whose report cannot be read without a recommendation instead of failing.
- The dashboard lists sweeps among the experiments, labels each row as tune or holdout, and marks the recommended row.

### Changed

- The project database moves to schema version 2 (experiments carry a kind and a sweep report). A project created by 0.2.0 is upgraded in place the first time any command opens it. The upgrade is safe to retry: a command interrupted during it leaves a project the next command finishes upgrading.
- In production, `contexteval.init()` hooks LangChain and LlamaIndex only when your app imported them before the call, so an app that uses neither no longer pays for their imports at startup. Call `init()` after your framework imports. `InstallReport` reports `not_imported` for a framework it left alone.

### Fixed

- A retrieval that reaches a hooked store from inside another hooked retrieval (langchain-chroma over chromadb, or a `@contexteval.retriever` function that queries chromadb) is recorded once, not twice, and an experiment setting applies to it once.
- A `top_k` experiment reaches a LangChain or LlamaIndex retriever that has no knob of its own through the hooked call underneath it: a custom `BaseRetriever` that queries chromadb, or a wrapper over a `base_retriever`. The experiment stops with an error only when no layer can apply the value. A per-call `k` passed to `invoke` no longer defeats the experiment, and the recorded event names the knob that was set.
- A LangChain or LlamaIndex retriever whose knob rejects the experiment's value keeps the user's value afterwards; before, a later experiment could leave it at the overridden value.

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
