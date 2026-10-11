# v1 implementation plan

Source design: `docs/design-notes.md`. This file resolves its open ambiguities, defines done, and lists the phases.
Progress lives in `todo.md`. Decisions live in `decisions.tsv`.

## Definition of done

v1 is done when all of these hold:

1. `pytest` passes on the whole suite.
2. A bundled demo RAG app (Chroma plus an LLM) contains planted failures of every label. With the deterministic fake LLM, `contexteval diagnose` assigns the expected label to every planted case.
3. `contexteval experiment` runs a settings grid on the demo and reports accuracy, tokens, and latency per setting. On the demo, a planted "missing" case passes under a larger `top_k`.
4. `contexteval ui` serves a page that lists failures with labels, evidence, and chunks, plus an experiment table. Checked over HTTP and in a browser.
5. The full workflow (`build-tests`, `run`, `diagnose`, `experiment`) completes once against the real OpenAI API on the demo corpus for under $0.50.

## Resolved ambiguities

Each item was a gap in the design notes. The choice and reason are recorded here.

1. **Gold evidence across chunkings.** The user's vector store chunks documents its own way, so chunk IDs from test generation cannot match. Each test case stores an `evidence` passage, quoted verbatim from the source document. A chunk "contains gold" when it contains the normalized evidence text as a run of whole tokens, or at least 80% of the evidence tokens inside a sliding window about 1.5 times the evidence length. The token rule handles evidence cut near a chunk boundary. The window stops a long text from matching on scattered common words. When neither of two adjacent delivered chunks contains gold alone but the two joined do, both are gold. The prompt contains gold when a gold delivered chunk is in it. The prompt text itself is never checked, because it also holds the question (amended in docs/architecture.md).
2. **Grading answers.** A grader decides pass or fail. Fast path: the normalized expected answer appears in the answer. Otherwise an LLM grader judges agreement at temperature 0.
3. **Nondeterminism.** Before replay, the diagnoser re-runs the baseline once. If it now passes, the label is `flaky` and replay stops. Replays never change the user's LLM settings.
4. **Labels.** `missing` (with cause `not_retrieved` or `dropped`), `irrelevant`, `ignored`, `not_context`, `flaky`. Traces without ground truth get a judge label with `source=judge`. The judge may also return `no_failure` (amended in docs/architecture.md).
5. **Replay decision table.** Only for failures where gold was in the prompt.
   - Gold moved to the top, other chunks kept, now passes: `ignored`. This is the least invasive fix, so it wins.
   - Else, only gold chunks kept, now passes: `irrelevant`.
   - Else: `not_context`.
   - For `missing`, one replay injects the evidence as the first chunk. The result is stored as `confirmed=true/false`. The label stays `missing`.
6. **Hooks in v1.** chromadb `Collection.query`, pinecone `Index.query` (tested with mocks only, no key), OpenAI chat completions and responses, Anthropic messages. Sync only in v1. pgvector has no client library to hook, so it uses the `@retriever` decorator. The design notes listed pgvector as auto-captured. That was wrong and this corrects it.
7. **Execution model.** The CLI imports `module:function` and runs questions in-process, one at a time. A context variable holds the active trace and the active intervention. No concurrency in v1, so there is no shared mutable state between runs.
8. **Several retrieval calls per answer.** A trace stores a list of retrieval events. An intervention applies to every retrieval call in the run.
9. **SDK's own LLM calls** (test generation, grading, judge). A small provider interface with OpenAI and Anthropic implementations, plus a fake for tests. Temperature 0.
10. **Storage.** stdlib `sqlite3` at `.contexteval/contexteval.db` in the project directory, JSON columns for nested data.
11. **Dashboard.** stdlib `http.server`, JSON endpoints, and one static HTML page. No web framework.
12. **CLI.** stdlib `argparse`. Core package has no required dependencies. `openai`, `anthropic`, `chromadb`, `pinecone` are optional extras.
13. **Test set builder.** Reads `.md` and `.txt` files. Asks the LLM for question, answer, and verbatim evidence as JSON. Rejects any case whose evidence is not a verbatim substring of the document.
14. **Cost estimate.** Characters divided by 4 for tokens, times a per-model price table. The CLI shows the estimate and asks to confirm unless `--yes`.
15. **Experiment knobs in v1.** `top_k` (pre-retrieval override), `rerank=lexical` (term-overlap scoring, no model download), `order` (`as_is`, `reverse`, `ends` for lost-in-the-middle), `max_chunks`, `compress` (keep the N sentences per chunk with highest query overlap). Experiments never use gold data. That would leak the answer.
16. **Packaging.** Package `contexteval`, Python 3.10+, MIT license.

## Phases

Each phase ends with `pytest` green and its acceptance check passing before the next starts.

| Phase | Builds | Acceptance |
|---|---|---|
| P0 Scaffold | Package layout, pyproject, pytest config, fake LLM, CLI entry point | `pytest` runs. `contexteval --help` prints the commands. |
| P1 Capture and store | Core types, SQLite store, hooks, decorator, prompt matching | Real chromadb plus the real OpenAI SDK over a mock HTTP transport produce a stored trace with correct `in_prompt`. |
| P2 Tests and runs | Test set builder, grader, `run` | Fake LLM builds validated cases from sample docs. `run` executes the demo app for every case and stores graded traces. |
| P3 Diagnoser | Checks, replay, judge | Every planted demo case gets its expected label. |
| P4 Experiments | Transforms, grid runner, cost estimate | Grid table is correct. The planted missing case passes with a larger `top_k`. |
| P5 Dashboard | HTTP server, JSON API, HTML page | Endpoints return stored data. Page shows failures and experiments in a browser. |
| P6 Real API and docs | OpenAI end-to-end run, README | Workflow completes on the real API under $0.50. README quickstart works as written. |
