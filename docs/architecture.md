# contexteval v1 architecture

## Problem

`contexteval` must explain why a RAG app's context failed without the user rewriting the app. The user hands over one function, `answer(question) -> str`. Everything else (capturing retrieval, swapping chunks for replays and experiments, grading, labeling) happens from the outside, through patched library methods and a context variable. Three things make the shape non-obvious. First, one mechanism has to serve capture, diagnosis replay, and experiments, and it has to work the same way for chromadb, pinecone, and a hand-written retriever. Second, experiments must never see gold data (plan item 15), while replays need it. Third, the acceptance bar is deterministic labels on planted failures with no network, so the demo app and its fake LLM are part of the architecture, not test scaffolding. The 16 resolved items in `plan.md` are fixed constraints. They include verbatim evidence with an 80% token rule, a flaky rerun before replay, the ignored-before-irrelevant ladder, sync-only hooks, in-process one-at-a-time execution, stdlib-only core, and pgvector through the decorator.

## Usage (caller's view)

Quickstart, as the README would show it.

```bash
pip install "contexteval[openai,chroma]"
contexteval init --provider openai          # creates .contexteval/{config.json,contexteval.db}
contexteval build-tests ./docs              # synthetic cases with verbatim evidence; shows cost, asks y/N
contexteval run my_app:answer               # every case through answer(); graded traces stored
contexteval diagnose                        # labels each failure of the latest run
contexteval experiment top_k=3,10 rerank=none,lexical order=as_is,ends
contexteval ui                              # http://127.0.0.1:8765
```

The user's app is unchanged unless it has custom retrieval.

```python
# my_app.py
import chromadb, openai

col = chromadb.PersistentClient("db").get_collection("docs")
llm = openai.OpenAI()

def answer(question: str) -> str:
    hits = col.query(query_texts=[question], n_results=4)       # hooked: n_results is top_k
    context = "\n\n".join(hits["documents"][0])
    r = llm.chat.completions.create(model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"{context}\n\nQ: {question}"}])  # hooked
    return r.choices[0].message.content
```

Custom retrieval names its top_k parameter in the decorator.

```python
import contexteval

@contexteval.retriever(top_k="limit", text="body")   # items are dicts; text lives under "body"
def search(query: str, limit: int = 5) -> list[dict]:
    return pg.execute("SELECT id, body FROM docs ORDER BY emb <-> %s LIMIT %s", (embed(query), limit))
```

Production capture is the same one line at app start. With no CLI run active, each LLM call closes one ungraded trace (the retrievals since the last LLM call plus that call), which `contexteval diagnose --run <production run>` labels with the judge.

```python
contexteval.init()   # returns InstallReport, e.g. {"chromadb": "installed", "pinecone": "not_installed", ...}
```

## Shape

**Module map.** The package has thirteen modules plus `ui.html`. The public surface is `contexteval` (`init`, `retriever`, `InstallReport`) and `contexteval.types`. Everything else is underscore-private (`_capture`, `_hooks`, `_intervention`, `_text`, `_llm`, `_store`, `_runner`, `_diagnose`, `_testset`, `_cli`, `_ui`), so an app that imports an internal is visibly reaching past the API, per the importable-internals red flag. `contexteval/demo/` (app, fake_llm, planted, corpus) ships in the package so `contexteval run contexteval.demo.app:answer` works after install.

**Data structures first** (per foundational-thinking, type-system-discipline). `types.py` holds the core types.
- `TestCase(id, question, expected_answer, evidence: Evidence | None, source)`. `CaseId` is a content hash, so rebuilding a test set upserts and never duplicates. `Evidence` is proven verbatim only in `_testset.parse_generated`.
- `Trace(question, outcome: Answered | Crashed, retrievals, llm_calls, metrics, test: TestOutcome | None)`. One union makes "answered and crashed" impossible. One optional field covers "graded iff from a test case". `prompt_text` and `in_prompt` are derived properties. The active Setting lives on the trace's run, not the trace, so there is one source for it (the store joins it into `StoredTrace`).
- `RetrievalEvent` keeps both `retrieved` (what the store returned) and `delivered` (what user code got after the intervention). `locate_gold` reads `delivered` and the prompt, and the UI shows both.
- `Diagnosis = Missing(cause, confirmed) | Irrelevant | Ignored | NotContext | Flaky | Judged(verdict, rationale)`. Each variant carries its replay outcomes as evidence. `label_of` and `source_of` are derived with an exhaustive `match`, never stored as a second field.
- `Setting` is the experiment knob record. It has fields only for plan item 15 knobs, so it cannot produce a gold transform. Experiments cannot leak the answer, by construction.

**One intervention mechanism** (per model-the-domain). `Intervention(top_k, transforms)` is the only way context changes. `Transform` is a closed union: knob transforms (`LexicalRerank`, `Reorder`, `KeepFirst`, `Compress`) and gold transforms (`InjectEvidence`, `GoldToTop`, `OnlyGold`), applied by one pure `apply_transform` with an exhaustive match. `Setting.intervention()` compiles knobs in one fixed order. `_diagnose.REPLAY_INTERVENTION` maps each replay step to an Intervention. Both hand it to `_runner.execute`, the single execution path for baseline, replay, and experiment.

**Hooks** (per boundary-discipline). `_capture` owns three context variables: `_active` (the run's mutable `ActiveRun` with question, intervention, and event lists), `_ambient` (production buffer), and `_suppressed`. It also owns two intercept functions every hook calls. `intercept_retrieval(codec, call, bound_args)` overrides `codec.top_k_param`, calls through, decodes to `Chunk`s, applies the intervention (query falls back to the run question for vector-only stores like pinecone), records, and re-encodes only when the intervention is not a no-op. `intercept_llm` records only. `_hooks` holds all library knowledge as one `HOOKS` table of `HookSpec(library, candidate targets, codec or adapter)`. Patching is class-level. Wrappers carry `__contexteval_original__`, so install is idempotent and `uninstall_all` restores originals for tests. The `@retriever` decorator builds a `_DecoratorCodec` from its arguments and goes through the same `intercept_retrieval`. Chromadb, pinecone, and custom code therefore share one code path. The SDK's own provider calls run inside `suppressed()`, so the grader and judge never record themselves. Hook-internal errors fall back to the unmodified call, and user exceptions propagate.

**Modes.** `Mode` is OFF, CLI, or PRODUCTION, owned by `_capture`. The CLI installs CLI mode before importing the user module, so a user `init()` inside `my_app` stays a no-op and `contexteval run` never writes production traces.

**Diagnoser** (per model-the-domain). The logic is data plus one sequencing function. `locate_gold` is pure (IN_PROMPT, DROPPED, NOT_RETRIEVED). `LADDER = ((GOLD_TO_TOP, Ignored), (ONLY_GOLD, Irrelevant))` encodes plan item 5's priority. `MISSING_CAUSE` maps location to cause. `diagnose_case(location, replay)` takes a `Replay = Callable[[ReplayStep], ReplayOutcome]`, so the full table is unit-tested with a scripted replay and no app. It runs the flaky rerun first, then the inject confirmation for missing, then the ladder lazily, so a case costs two to four app calls. Cases without evidence and production traces go to `judge`.

**Store** (per make-operations-idempotent). The tables are `test_cases`, `runs`, `traces`, `diagnoses`, and `experiments`. CHECK constraints mirror the type invariants (answer xor error, `passed` null iff no case, experiment runs have an experiment id). A run is an immutable event committed per trace, so a crash leaves a valid partial run. `diagnose` skips already-diagnosed traces unless `--force`, so rerunning after a crash resumes. Experiment tables are derived with `GROUP BY`, never stored.

**Call chain for `contexteval experiment`.** `_cli.cmd_experiment` calls `_runner.run_experiment`, which calls `run_case`, then `execute`, then the user `answer`, then a patched method, then `_capture.intercept_retrieval`. That is three package files a reader traces (`_cli`, `_runner`, `_capture`), plus `_hooks` for the library shape.

**Interface depth.** App code sees two names. `init()` hides library detection, version-drifted targets, idempotent patching, and production buffering. `@retriever` hides result decoding and encoding for three item shapes, top_k override, and intervention. The CLI hides the rest. `Store` exposes domain types only. The SQL and JSON encodings stay private.

**Test strategy.** No test uses the network. An autouse fixture makes non-loopback `connect` raise. The SDK LLM is `FakeProvider`, which dispatches on `LLMRequest.task` and the structured `inputs`, not on prompt text. The app LLM is the real `openai` SDK over `httpx.MockTransport` backed by `demo/fake_llm.respond`, so P1's real-SDK hook path is what the acceptance test exercises. Chromadb runs ephemeral with a deterministic `HashEmbedding`, so no ONNX download happens, and telemetry is off. Pinecone is a stand-in class patched through `_hooks.patch_method` with the real `PineconeCodec`. `demo/planted.py` is the single list of planted cases. The demo cases, the fake LLM script, and the expected labels all derive from it. Each fake rule names the context condition under which the model answers correctly, so the expected label follows from the rule plus the ladder (table in `fake_llm.py`). `FlipFlop` persists a per-question counter because `run` and `diagnose` are separate processes. Test layers: pure units (`_text`, transforms, `diagnose_case` table, `parse_grid`, schema round-trips), hook tests against real chromadb and the openai SDK over mock transport, and two acceptance tests through `_cli.main` for DoD items 2 and 3.

## Synthesis decision

Four runners sketched v1 independently (opus, fable, sonnet, haiku). A fable cross-judge scored them 28, 20, 21, 17 on a six-part rubric and recommended the opus candidate as base. My own read agreed. The candidates converged on the same core shape (one `Intervention` type, a hook table over shared intercepts, a suppression context variable, diagnosis as a variant union, the replay ladder as data, a demo on real chromadb and the real openai SDK over `httpx.MockTransport`). That convergence is the strongest signal in the arena.

The base is opus because it is the only candidate sound on all three load-bearing mechanisms. Its diagnoser is pure over an injected replay callable. One intercept serves chromadb, pinecone, and the decorator, with modes that stop `contexteval run` from writing production traces. Its flaky demo case survives `run` and `diagnose` as separate processes. Fable and sonnet kept the flaky counter in memory, so the shell workflow would mislabel it. Haiku never suppressed the SDK's own LLM calls, so grader calls would become production traces.

**Grafts. These are binding on implementation.**

1. From fable. Diagnosis variants carry their evidence as data. `Irrelevant.removed` and `Ignored.gold_at` hold the chunk positions (retrieval index, chunk index) involved, so the UI says "removing chunks 3 to 8 fixed it" from data. `explain()` becomes a projection.
2. From fable. Tests for DoD item 4 (UI endpoints serve stored data) and for idempotence (a second `diagnose` is a no-op). Each test docstring names the defect it catches.
3. From sonnet. Error policy at the hook. Recording errors are swallowed and logged once. Intervention errors propagate as `InterventionNotApplicable`, for example a `top_k` override on a site with no top_k parameter. The retrieval is never called twice. Replaces the base's "fall back to call(args) unchanged", which would silently turn an experiment into a baseline run.
4. From sonnet. `@retriever(..., text=, rebuild=)` accepts callables for result items that are neither `str` nor `Mapping`. Replaces the base's `copy.copy(first_item)` injection.
5. From sonnet. The 80% token rule is checked inside a sliding window of about 1.5 times the evidence length, not across the whole text. Without the window a long prompt matches short evidence on scattered common words. This amends plan item 1.
6. From haiku. `_hooks.patch_method` reads the method from `owner.__dict__`, never `getattr`, so a subclass is not wrapped with its parent's method.
7. From haiku. `init(pinecone_text_key="text")` names where pinecone keeps chunk text.

**Fixes to the base.**

- Import cycles. `Trace.in_prompt` becomes a stored field computed by the capture layer when the trace is finalized, not a property that imports `_text`. `install` moves to `_hooks`, and `_capture` no longer imports `_hooks`.
- `JudgeVerdict.NO_FAILURE` is accepted. Unlabeled production traffic is mostly fine, and forcing a failure label would mislead. This amends plan item 4 for judge labels only.
- The flaky rerun runs first for every ground-truth case, including missing ones (plan item 3 read literally).
- The DoD item 3 assertion excludes the flaky planted case, because its alternating answers are noise in an experiment table.
- `_DecoratorCodec` with `top_k=None` and a `top_k` intervention raises `InterventionNotApplicable` (graft 3).

**Rejected.** Fable's opaque callable transforms (not storable or showable). Sonnet's flat `Trace` with optional fields. Haiku's deterministic trace ids, which overwrite history on a second run. Fable's `hooks/` subpackage (six one-line pass-through modules).

## Tradeoffs accepted

- We accept class-level monkeypatching, which can break on library updates, in exchange for zero app changes. Candidate target lists and an `incompatible` status make drift visible in `InstallReport`, not silent.
- We accept that an intervention applies to every retrieval call in a run, including injecting evidence into each one, in exchange for a single rule that needs no call-site identity (plan item 8).
- We accept that production trace boundaries come from a heuristic (one LLM call closes the buffer) in exchange for capture without a run wrapper. Multi-call agents get split into several traces.
- We accept that replay traces are not stored, only their outcomes inside the diagnosis, in exchange for a schema with one trace kind per run.
- We accept the `Trace.in_prompt` column as a write-only denormalization of a derived property, in exchange for a UI that reads one row.
- We accept that `run_experiment` always prepends a baseline row, which costs one extra pass, in exchange for every table having a comparison.
- We accept that chromadb multi-query calls are recorded but never intervened on, in exchange for codecs that handle one result row.
- We accept underscore module names, which look noisy in imports, in exchange for an enforced public boundary.

## Alternatives considered

- **Per-library hook classes with their own intercept logic.** Each hook would apply interventions and record events itself. That hides less, because the policy is repeated in four places, and a new transform would need edits in each. The codec-plus-shared-intercept shape keeps policy in one function and library knowledge in one table.
- **A `@contexteval.trace` or explicit `with contexteval.run():` wrapper instead of a context variable set by the runner.** That puts lifecycle on the caller. It is clearer for production, but it breaks the "no app changes" promise and gives two ways to start a run.
- **Labels as an enum plus optional fields (`label`, `cause`, `confirmed`, `verdict`).** That admits `label=ignored, cause=dropped`. The variant union deletes those states and the comment that would explain them.
- **Demo app LLM as a direct call into a fake (no openai SDK).** It is simpler, but the acceptance test would never cross the real hook path. The MockTransport costs a dev dependency and buys fidelity.

## Implementation reconciliation

None yet.

## Open questions and risks

- Should the flaky rerun also run before the missing confirmation? The sketch runs it first for every ground-truth case, reading plan item 3's "before replay" literally. The other reading skips it for missing, which saves one call per missing case.
- Is `JudgeVerdict.NO_FAILURE` acceptable? Plan item 4 lists no "ok" label, but unlabeled production traffic is mostly fine, and forcing a failure label would mislead.
- Pinecone chunk text key defaults to `metadata["text"]`. Should `init()` take a `pinecone_text_key`, or should a config file own it?
- Risk: the hashed-embedding demo needs a hand-tuned corpus so the planted gold ranks exactly 5th, 2nd, and 1st. The acceptance test should assert ranks directly so a corpus edit fails loudly.
- Risk: `FlipFlop` makes the flaky case alternate across experiment runs, so its row is noise in the experiment table. Should the experiment test exclude it?
- Risk: production mode writes to SQLite on the user's hot path. WAL mode and swallow-and-log on write failure are the plan. Is sync write latency acceptable for v1?
- `_DecoratorCodec` injection for object items uses `copy.copy(first_item)` with the text field set. Is that acceptable, or should object-returning retrievers be required to pass an explicit builder?

## Next implementation step

Write `types.py`, `_text.py`, and `_intervention.py` for real, with the `diagnose_case` table test against a scripted replay, because every other module consumes these types and the table is the product's core claim.
