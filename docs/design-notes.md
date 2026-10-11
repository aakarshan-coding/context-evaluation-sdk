# Context Evaluation SDK: Design Notes

Status: brainstorm decisions, not a final spec. Date: 2026-10-10.
`contexteval` is a placeholder name.

## Goal

An SDK that helps developers measure, debug, and improve the context given to AI agents.
It captures agent context, runs experiments across retrieval and compression strategies,
and finds whether failures come from **missing**, **irrelevant**, or **ignored** information.
Target outcome: higher task success with fewer tokens and lower latency.

Project type: a real product for adoption (open-source core, paid hosted version later).

## Positioning

Tracing and eval tools already exist (Langfuse, LangSmith, Arize Phoenix, Braintrust,
Ragas, DeepEval, TruLens). Capture and experiments are not the differentiator.
The differentiator is **attribution**: telling the developer *why* context failed and
*what to change*. Capture is the easy entry point.

First install must give a useful answer in about 10 minutes without rewriting the agent.

## Users

1. First: RAG / retrieval-heavy agents (support bots, doc Q&A, internal knowledge agents).
   Ground truth is clear here, so attribution is measurable.
2. Second: long-running coding / tool agents (context bloat, compression, pruning).

## Decisions

### Language
Python first.

### Capture
- One line at app start: `contexteval.init()`.
- `init()` detects installed libraries and hooks them automatically:
  - Vector DBs: Chroma, Pinecone, Qdrant, pgvector. Record query, chunks, scores.
  - LLM clients: OpenAI, Anthropic. Record final prompt and answer.
  - Frameworks (later): LangChain, LlamaIndex.
- Link retrieval to prompt by matching chunk text inside the prompt text.
- Fallback for custom retrieval code (SQL search, internal APIs, hand-written logic):
  `@contexteval.retriever` decorator on the user's search function.
- Trade-off accepted: auto-hooks need per-library code and can break on library updates.

### Storage and interface
- Local first, open source. Data stays on the user's machine in SQLite.
- CLI plus local web dashboard (`contexteval ui`).
- Reason: user documents are often private; local-first lowers the trust barrier.
- Paid hosted team version later.

### Correctness signal
- Default: synthetic test set generated from the user's documents. An LLM writes
  question/answer pairs from chunks, so each question has a known **gold chunk**.
- Also: import the user's own labeled tests.
- Later (for the background optimizer): production feedback and reference-free LLM judge.

### Diagnosis (the core)
Definitions:
- **Missing**: the right information never reached the prompt.
- **Irrelevant**: the right information was there, but noise confused the model.
- **Ignored**: the right information was there and clear, but the model did not use it.

Layered approach:
1. **Ground-truth checks** (cheap, run first):
   - Gold chunk not retrieved → missing.
   - Retrieved but cut before the prompt (truncation/compression) → missing (lost later).
   - In prompt but answer wrong → go to layer 2.
2. **Replay experiments** (only on remaining failures). Re-run with changed context:
   - Add gold chunk → fixed means missing.
   - Remove other chunks → fixed means irrelevant.
   - Move gold chunk to top → fixed means ignored (position problem).
   - Nothing fixes it → not a context problem (model or prompt issue).
3. **LLM judge**: only when no ground truth exists (e.g. unlabeled production traffic).

Users pay LLM costs with their own API key. Show estimated cost before a run.

### Automation roadmap
| Level | What | When |
|---|---|---|
| 1 | Manual experiments: user picks settings to compare | v1 |
| 2 | Automated sweep guided by diagnosis | v2 |
| 3 | Background optimizer: replay sampled production traces in shadow mode, **suggest** changes only | v3 |
| 4 | Auto-apply with canary rollout and rollback | later, opt-in |

Diagnosis-guided search:
- Mostly missing → try higher top-k, hybrid search, query rewriting.
- Mostly irrelevant → try reranker, lower top-k, stricter filters.
- Mostly ignored → try reordering (key chunks first/last), compression, less noise.

## Components (approved)

| Part | Job |
|---|---|
| 1. Capture | `init()` + decorator. Records each run as a trace. |
| 2. Store | Local SQLite: traces, test sets, experiment results, diagnoses. |
| 3. Test set builder | Synthetic questions with gold chunks; import user tests. |
| 4. Diagnoser | Ground-truth checks → replay → LLM judge. |
| 5. Experiment runner | Runs test set across settings; scores accuracy, tokens, latency. |
| 6. Interface | CLI and local web dashboard. |

## Scope (approved)

**v1**
- Capture: Chroma, Pinecone, pgvector, OpenAI, Anthropic, plus decorator.
- Synthetic test sets and user test import.
- Full diagnoser (all 3 layers).
- Manual experiments.
- CLI plus simple dashboard: failure list with labels and chunks, experiment comparison table.

**Later**
- v2: diagnosis-guided automatic sweep; LangChain and LlamaIndex plugins.
- v3: background optimizer on production traces (suggest only); OpenTelemetry import.
- v4: long-running tool agents (bloat, compression); paid hosted team version.

## Data flow (proposed, NOT yet approved)

User workflow:
```
contexteval init                  # set up local project
contexteval build-tests ./docs    # synthetic questions with gold chunks
contexteval run my_app:answer     # run each test question through the user's app
contexteval diagnose              # label each failure
contexteval experiment top_k=3,10 rerank=on,off
contexteval ui                    # open dashboard
```

The user provides one entry point:
```python
def answer(question: str) -> str: ...
```

Key idea: change context **through the hooks**, not the user's code.
- Before retrieval: change parameters (e.g. top-k).
- After retrieval: change the chunk list before user code sees it
  (rerank, filter, reorder, compress, add gold chunk, remove chunks).
- Diagnosis replay and experiments both work as: apply change at hook → re-run `answer()`.

Trace record (one per question run):
```
Trace
  question, answer, expected_answer
  retrieval: query, params, chunks [text, score, id]
  prompt: final text sent to LLM, model, settings
  in_prompt: which chunks appeared in the prompt
  metrics: tokens, latency, cost
  setting: active experiment config
  diagnosis: label + evidence (e.g. "removing chunks 3-8 fixed it")
```

Known limit: prompt text from non-retrieval sources (e.g. chat history) is visible
but cannot be changed in v1.

## Open items

- Approve or revise the data flow section above.
- Error handling and testing strategy not yet designed.
- Final product name.
- Next step: continue design with pstack (`/pstack:poteto-mode` → `architect`).
