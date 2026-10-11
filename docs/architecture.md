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

P0 and P1 (branch `p1-capture`). Where the code departs from the sketch, and why.

- `install()` and `init()` live in `_hooks`, not `_capture`. The sketch kept `init` in `_capture` with a lazy import of `_hooks`. Moving both makes the import graph a line (`types <- _text <- _intervention <- _capture <- _hooks <- __init__`), and `init(pinecone_text_key=)` (graft 7) is library knowledge anyway. `_capture` keeps `InstallReport`, `Mode`, `set_mode(mode, make_sink)`, and a `reset_process()` test seam. The sink factory runs only when the mode actually escalates, so a user `init()` under `contexteval run` opens no database.
- `InterventionNotApplicable` is defined in `_intervention` and exported from the package root. It propagates through the user's own retrieval call, so app code may need to name it.
- `Trace.in_prompt` is a required constructor field. `_capture.finalize_trace(run, outcome, latency_ms, test)` is the one place that fills it (via `_text.in_prompt_pairs`) along with `Metrics`. The runner (P2) and the production sink both call it.
- `@retriever` takes `text=` as a key, attribute, or callable, and `rebuild=` for building injected items (graft 4). Items that are neither `str` nor `Mapping`, and empty results, raise `InterventionNotApplicable` on injection unless `rebuild` is given. The sketch's `copy.copy(result[0])` is gone.
- Chroma multi-query calls: `decode` reads row 0 and `encode` raises `InterventionNotApplicable`. The sketch said "recorded per row"; one event per call keeps `source_index` meaningful.
- Production traces take their question from the first retrieval's query text, or `""` when the call had none. The LLM adapter exposes only the flattened prompt.
- `Store` gained `close()`, `find_project()`, and `PROJECT_DIR_NAME`; the connection uses `check_same_thread=False` because the production sink flushes from whichever thread made the LLM call.
- `_cli.COMMANDS` maps names to `Command(run, help, add_arguments)` rather than bare callables, so the parser is generated from the same table that dispatches. `save_config` and `require_project` were added.
- `_llm.cost_usd` is implemented (finalize needs it). `PRICES` still holds only `fake`, so app-call cost is `None` until P6 fills the table.
- `demo/app.HashEmbedding` is implemented early because the hook tests need an offline embedding. chromadb 1.x requires `name()`, `get_config()`, `build_from_config()` and an explicit `__init__` on custom embedding functions.
- anthropic 1.x ships its own `httpx2`; the anthropic hook test uses `httpx2.MockTransport`. The openai test uses `httpx.MockTransport` as planned.
- The P3/P4 test sketches (`test_acceptance.py`, `test_diagnose_table.py`) carry a module-level skip marker naming the phase instead of failing on `NotImplementedError`.
- The pinecone codec synthesizes injected matches as `type(matches[0])(id=, score=, values=, metadata=)` and falls back to a small attribute-and-item object when that constructor refuses the kwargs.

P2 (branch `p2-runs`).

- The demo retrieves `TOP_K = 4`, not 3. The not_retrieved gold ranks 5th for its question, inside the 5-to-10 band, so `top_k=10` retrieves it. `tests/test_demo.py` asserts every planted rank directly, plus the replay outcomes each label needs (inject, gold to top, only gold) through `_runner.execute` with the real gold transforms. A corpus edit that would change a P3 label fails there first.
- `Planted.expected` is `Label | None`. `None` marks a healthy case that passes at baseline (`passes_baseline`). PLANTED holds six failure cases and two healthy ones. The irrelevant case's distractor is a chunk tagged `[FORUM]`. `planted.write_jsonl` and `python -m contexteval.demo.planted` write the import file.
- `fake_llm.evaluate(rule, question, chunks, state_file)` holds the rule match so tests can call it without HTTP. The transport resolves the default state file per request, so a cached client follows the cwd's project.
- The demo collection is persistent, not ephemeral. It lives at `app.data_dir()/demo_chroma`, beside the flaky state file. `data_dir()` is `<project>/.contexteval`, or a temp directory when no project is found, so the demo never creates a project the user did not ask for. The collection is upserted once per process per path, which is idempotent. Reason: chromadb 1.5.9's ephemeral client drops an older collection's vector index once enough other collections are opened in the same process. `count()` still reports every row, but queries return nothing. In the full suite this happened about one run in three (`test_hooks` opens many collections). A standalone probe reproduced it after 23 and 53 extra collections. A persistent index never lost results in 3 runs of 120 extra collections each. Hook tests still use ephemeral clients.
- Providers take an optional `client` (an SDK client) so tests inject a mock transport. The default OpenAI model is `gpt-4.1-mini`. `AnthropicProvider` keeps `claude-haiku-4-5` as its default because current Claude models (Haiku 5.5, Sonnet 5.5) return a 400 on non-default sampling parameters. For those models temperature is omitted. For older models it is sent as `extra_body={"temperature": 0}`, because anthropic 1.x removed `temperature` from `messages.create`'s signature. `PRICES` covers gpt-4.1-mini, gpt-4.1-nano, gpt-4o-mini, claude-haiku-4-5, claude-haiku-5-5, claude-sonnet-4-6, and claude-sonnet-5-5.
- `_runner.execute` re-raises `InterventionNotApplicable` instead of storing it as a crash, per graft 3. Every other exception still becomes `Crashed`.
- The sketch's `run_suite` name is kept. `_runner.estimate_grading` is new. It is the `run` command's estimate, and it assumes that every case needs one LLM grade (worst case). The app's own calls are reported from the last baseline run when one exists, because the app model is unknown before the run.
- `build-tests --import` makes no LLM calls, so it prints no estimate and asks for no confirmation. The import's test set is the file stem. Generated cases go to the `synthetic` test set. `import_tests` also reads `.csv` files.
- `confirm` treats EOF or an unreadable stdin as "no" and exits 2, so a script that omits `--yes` never blocks.
- `run_experiment` is still `NotImplementedError` (P4).

P4 (branch `p4-experiment`).

- `parse_grid` derives each knob's parser from `typing.get_type_hints(Setting)`. An Enum field parses by value, a `Literal` field by membership, and an `int | None` field with `int`. A field of any other type raises `TypeError` at parse time, so a new kind of knob fails loudly instead of being skipped. A knob given twice is an error, not a merge.
- `_runner.experiment_settings(grid)` is new. It returns BASELINE first and then the grid with duplicates and BASELINE removed. `run_experiment` runs that list and the CLI cost estimate counts it, so the two cannot disagree. Each setting is one `run_suite` call with the experiment id, so experiments use the same path as `contexteval run`.
- The experiment command takes the app as an optional leading positional (`contexteval experiment [module:function] knob=v1,v2 ...`), the same as `run`. The sketch's `--app` flag is gone. A first token with `:` and no `=` is the app. Without one, the app comes from `config.app`, which `run` remembers. Experiments run every stored case. There is no `--test-set` flag yet.
- Setting names keep the space-separated form from P1 (`top_k=10 order=ends`), not a comma-separated one. A comma already separates alternative values in the grid syntax, so `top_k=10,order=ends` would read as one knob with two values.
- `InterventionNotApplicable` during an experiment becomes a one-line `UserError`. The runs finished before it stay stored under the experiment.
- The DoD item 3 test lives in `tests/test_experiment.py`, not `tests/test_acceptance.py`, so the P3 and P4 owners never edit one file at once. The skipped DoD item 3 placeholder in `test_acceptance.py` is left for that file's owner to delete.
- The transform tests asked for in P4 (lexical rerank, ends order, compress, max_chunks) already existed in `tests/test_intervention.py` from P1. No new ones were added.

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
