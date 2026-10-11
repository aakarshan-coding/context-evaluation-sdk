# contexteval

contexteval tells you why your RAG app gave a wrong answer. It runs your app on test questions, records what was retrieved and what reached the prompt, and labels each failure.

Most RAG failures are context failures, and they come in three kinds:

- **Missing.** The right passage never reached the prompt. Retrieval did not find it, or your code dropped it after retrieval.
- **Irrelevant.** The right passage was in the prompt, but other chunks crowded it out. The model answers correctly once they are removed.
- **Ignored.** The right passage was in the prompt, and the model skipped over it. The model answers correctly once that passage moves to the top.

Each label points at a different fix. Missing calls for better retrieval, such as a larger `top_k`. Irrelevant calls for a reranker or fewer chunks. Ignored calls for a different chunk order or shorter chunks. contexteval then lets you test those fixes against your test set with one command.

You do not change your app. contexteval hooks chromadb, pinecone, LangChain and LlamaIndex retrievers, OpenAI, and Anthropic from the outside. Custom retrieval needs one decorator.

## Install

contexteval needs Python 3.10 or later. The core package has no dependencies.

Once contexteval is published on PyPI, install it with the demo extras:

```bash
pip install "contexteval[demo]"
```

Until then, install from source:

```bash
git clone https://github.com/aakarshan-coding/context-evaluation-sdk.git
cd context-evaluation-sdk
pip install ".[demo]"
```

The extras are `openai`, `anthropic`, `chroma`, `pinecone`, `langchain`, `llamaindex`, and `demo` (chromadb plus openai). Install the ones your app uses, for example `pip install "contexteval[openai,chroma]"`, or `pip install ".[openai,chroma]"` from source. An app that already has its libraries installed needs no extra; the extras only install those libraries.

Release steps for maintainers are in [docs/RELEASING.md](https://github.com/aakarshan-coding/context-evaluation-sdk/blob/main/docs/RELEASING.md).

## Quickstart

The bundled demo is a small RAG app over a customer-support corpus, built on chromadb and the OpenAI SDK. It has planted failures of every label. With `--provider fake` and the default demo settings, nothing calls a real API and nothing costs money.

Create a project in an empty directory:

```bash
mkdir contexteval-demo
cd contexteval-demo
contexteval init --provider fake
```

Write the planted test cases to `planted.jsonl` and import them:

```bash
python -m contexteval.demo.planted
contexteval build-tests --import planted.jsonl
```

Run every case through the demo app's `answer()` function:

```bash
contexteval run contexteval.demo.app:answer --yes
```

You see `run 1: 8 cases, 2 passed, 6 failed, 0 crashed`.

Label each failure:

```bash
contexteval diagnose --yes
```

Each failure gets a label and a line of evidence:

```text
app: up to 54 replay calls to contexteval.demo.app:answer (6 failures with evidence, 3 steps x 3 samples each, worst case)
run 1:
  not_context  When is phone support open?
               keeping only gold (0/2) did not fix it; moving gold to the top was skipped because gold was already at the top
  flaky        Where is Trailhead Gear headquartered?
               the unchanged baseline passed on rerun (2/2)
  irrelevant   Can fuel canisters be shipped by air?
               keeping only the gold chunks fixed it (2/2); removed chunks 2, 3, 4 of retrieval 1; moving gold to the top was skipped because gold was already at the top
  ignored      How long do warranty repairs take?
               moving the gold chunk to the top fixed it (2/2); gold was at chunk 3 of retrieval 1
  missing      How long is the warranty on a sleeping bag?
               the gold chunk was never retrieved; injecting the evidence fixed it (2/2)
  missing      What is store credit for a returned tent worth?
               the gold chunk was retrieved but dropped before the prompt; injecting the evidence fixed it (2/2)
diagnosed 6, skipped 0 already labeled, skipped 0 crashed, skipped 0 with no answer text
```

The count after each step is samples that passed over samples drawn. The demo's fake model is deterministic, so every step settles at 2 of 2. A real model at a non-zero temperature shows splits like `(2/3)`.

Test a fix. The "never retrieved" case ranks 5th, and the demo retrieves 4 chunks:

```bash
contexteval experiment top_k=10 --yes
```

The table compares the baseline with the new setting:

```text
setting   accuracy  passed  mean input tokens  mean latency ms  total cost
baseline  38%       3/8     124                62.0             unknown
top_k=10  50%       4/8     272                7.9              unknown
```

The "never retrieved" case passes under `top_k=10`, so accuracy rises by one case. The flaky case failed only on its first call, so it passes in both rows. The cost column reads `unknown` because the fake model has no price.

Or let contexteval pick the settings from the labels. The demo has only 8 cases, too few to hold any out, so this run tunes on all of them:

```bash
contexteval sweep --holdout 0 --yes
```

The sweep says which labels no setting can fix and why, tries the settings the other labels call for, combines the winners, and ends with one recommendation:

```text
Set top_k=8 rerank=lexical. Accuracy 38% -> 62% (no holdout cases, so this is unconfirmed). Uses 76% more input tokens; mean latency 36 ms -> 4 ms.
```

The [Sweep](#sweep) section shows the whole output and explains how the search works.

Open the dashboard:

```bash
contexteval ui
```

### Run the demo on the real OpenAI API

The same workflow runs against the real API. Set `OPENAI_API_KEY`, then set `CONTEXTEVAL_DEMO_LLM=openai` so the demo app calls OpenAI too. This example generates test cases from the demo corpus instead of importing the planted ones:

```bash
mkdir contexteval-real
cd contexteval-real
export CONTEXTEVAL_DEMO_LLM=openai
contexteval init --provider openai
contexteval build-tests "$(python -c "from contexteval.demo.app import CORPUS_DIR; print(CORPUS_DIR)")" --yes
contexteval run contexteval.demo.app:answer --yes
contexteval diagnose --yes
contexteval experiment top_k=10 --yes
```

The whole workflow made 39 calls to `gpt-4.1-mini` and cost $0.0038 in our run. A real model fails fewer cases than the planted script, so expect fewer labels.

## Point it at your own app

contexteval needs one function that takes a question and returns an answer string. You pass it to the CLI as `module:function`. The CLI imports the module from the current directory.

If your app queries chromadb or pinecone, or retrieves through a LangChain or LlamaIndex retriever, and calls OpenAI or Anthropic, directly or through the framework's chat model, it works as is. contexteval patches those classes before it imports your module.

### Supported stacks

| Layer | Hooked |
|---|---|
| Vector stores | chromadb `Collection.query` and `AsyncCollection.query`; pinecone `Index.query` |
| Frameworks | LangChain `BaseRetriever.invoke` and `ainvoke`, so any retriever, including `VectorStoreRetriever` over langchain-chroma or another vector store; LlamaIndex `BaseRetriever.retrieve` and `aretrieve`, so any retriever, including `VectorIndexRetriever` |
| LLM clients | OpenAI chat completions and responses, Anthropic messages, sync and async. Framework chat models such as `ChatOpenAI` call these SDKs, so they are covered through them |
| Anything else | `@contexteval.retriever` on your search function |

A framework retriever that queries a hooked store underneath, such as langchain-chroma over chromadb, records one retrieval, owned by the retriever. A `top_k` experiment sets the framework's own knob for each call and puts it back afterwards: `search_kwargs["k"]` (or a `k` field) on a LangChain retriever, `similarity_top_k` on a LlamaIndex retriever. Injected evidence reaches your code as a `Document` or a `NodeWithScore`, so your prompt code needs no change.

If your app retrieves some other way (pgvector, SQL, a search API, hand-written code), mark the search function with `@contexteval.retriever`. Name the parameter that controls how many results come back, so experiments can change it. Each result item is a string or a dict whose text sits under `text`:

```python
# my_app.py
import contexteval
import openai

DOCS = [
	"Refunds are issued within 30 days of delivery.",
	"Standard shipping takes four to six business days.",
	"Phone support is open from 7 am to 9 pm Mountain Time.",
]
llm = openai.OpenAI()


@contexteval.retriever(top_k="limit")
def search(query: str, limit: int = 2) -> list[dict]:
	words = set(query.lower().split())
	ranked = sorted(DOCS, key=lambda doc: -len(words & set(doc.lower().split())))
	return [{"id": str(i), "text": doc} for i, doc in enumerate(ranked[:limit])]


def answer(question: str) -> str:
	context = "\n\n".join(hit["text"] for hit in search(question))
	reply = llm.chat.completions.create(
		model="gpt-4.1-mini",
		messages=[{"role": "user", "content": f"Answer from this context only.\n\n{context}\n\nQuestion: {question}"}],
	)
	return reply.choices[0].message.content
```

Use `text=` to name a different key or attribute, or pass a function that returns the text. Use `top_k=None` if the result count is fixed.

Generate tests from your documents, or import your own:

```bash
contexteval init --provider openai
contexteval build-tests ./docs
contexteval run my_app:answer
```

An import file is JSONL or CSV with `question`, `expected_answer`, and an optional `evidence` passage copied from your documents. Cases with evidence get the full diagnosis. Cases without it go to an LLM judge.

To capture production traffic, call `contexteval.init()` once at app startup. It hooks the same libraries and stores each answered question as an ungraded trace in a production run. Find the run id in `contexteval ui` and label it with `contexteval diagnose --run ID`. Under `contexteval run`, `init()` does nothing, so you can leave it in.

Call `init()` after importing LangChain or LlamaIndex. In production it hooks a framework only when your app has already imported it, so an app that uses neither does not pay for their imports. The returned `InstallReport` says `not_imported` for a framework it left alone.

## Commands

Every command except `init` finds the project by walking up from the current directory to a `.contexteval/` directory. Commands that call an LLM print a cost estimate and ask before they start. `--yes` skips the question.

| Command | What it does |
|---|---|
| `contexteval init [--provider openai\|anthropic\|fake] [--model M] [--app module:function]` | Creates `.contexteval/` with `config.json` and the database. The provider is the LLM that contexteval itself uses to write tests, grade, and judge. |
| `contexteval build-tests DOCS_DIR [--per-passage N] [--yes]` | Reads every `.md` and `.txt` file, asks the LLM for question, answer, and evidence, and keeps only cases whose evidence appears word for word in the document. |
| `contexteval build-tests --import FILE` | Imports your own JSONL or CSV cases. Makes no LLM calls. |
| `contexteval run [module:function] [--test-set NAME] [--yes]` | Runs every case through your function and grades each answer. Remembers the app for later commands. |
| `contexteval diagnose [--run ID] [--samples N] [--force] [--yes]` | Labels each failure of the latest baseline run, or of run `ID`. Each replay step draws up to `N` samples (default 3, odd) and a majority decides. Skips failures that already have a label unless you pass `--force`. |
| `contexteval experiment [module:function] KNOB=v1,v2 ... [--yes]` | Runs every case under each combination of settings and prints a comparison table. |
| `contexteval sweep [module:function] [--rounds N] [--max-settings N] [--holdout F] [--seed S] [--tolerance N] [--yes]` | Picks settings from the labels of the latest diagnosed baseline run, tries them on a tune split, confirms the winner on a holdout split, and prints one recommendation. |
| `contexteval ui [--port 8765] [--no-open]` | Serves the dashboard at `http://127.0.0.1:8765/`. |

Grading checks first whether the expected answer appears in the answer, after normalizing case and punctuation. If it does not, the LLM decides whether the two agree, at temperature 0.

## How diagnosis picks a label

contexteval works through these steps for each failed case and stops at the first one that decides:

1. If the case has no evidence passage, an LLM judge reads the trace and picks a label. Production traces always take this path. The judge can also say there was no failure.
2. contexteval reruns the case unchanged. If it passes now, the label is `flaky`.
3. contexteval looks for the evidence in what the app sent. If it is not in the prompt, the label is `missing`. The cause is `not_retrieved` when no retrieval returned it and `dropped` when retrieval returned it but your code left it out of the prompt. One more run puts the evidence first in the context and records whether that fixes the answer.
4. If the evidence was in the prompt, contexteval reruns with that chunk moved to the top and every other chunk kept. If the answer is now right, the label is `ignored`.
5. Otherwise it reruns with only the chunks that contain the evidence. If the answer is now right, the label is `irrelevant`.
6. Otherwise the label is `not_context`. The context held the answer and the model still got it wrong. Look at the prompt, the model, or the expected answer.

The order matters. Moving one chunk is a smaller change than removing chunks, so `ignored` wins when both would work. Replays never change your app's LLM settings.

Each step is sampled, because an app that calls its model at a non-zero temperature can pass a replay by luck. A step draws up to `--samples N` replays (default 3; `N` must be odd) and stops as soon as a majority of `N` agree, so a step that passes twice in a row costs 2 calls, and a split step costs 3. A step passes when a majority of its samples pass. `flaky` means a majority of the unchanged reruns passed. Every label line shows the count for the step that decided it, as in `(2/3)`, and the dashboard shows every sample's answer under its step. A case costs between 2 and 3 times `N` runs of your app; the cost prompt states the worst case. `--samples 1` decides from one replay per step, as versions before 0.2 did. Labels stored by those versions still load and show as `(1/1)`.

Chunk and retrieval positions in label lines and in the dashboard count from 1.

A passage counts as present when the chunk or prompt contains the evidence text after normalization. It also counts when 80% of the evidence words appear inside a window about 1.5 times the evidence length. The second rule catches evidence that your chunker split in two.

## Experiment knobs

Each knob takes a comma-separated list of values. contexteval runs every combination, always with a baseline row first.

| Knob | Values | Effect |
|---|---|---|
| `top_k` | a positive integer | Overrides the result count passed to the retrieval call. |
| `rerank` | `none`, `lexical` | `lexical` reorders chunks by word overlap with the question. No model download. |
| `order` | `as_is`, `reverse`, `ends` | `ends` puts the best chunks at the start and end of the context, for models that lose the middle. |
| `max_chunks` | a positive integer | Keeps only the first N chunks. |
| `compress` | a positive integer | Keeps the N sentences of each chunk with the most word overlap with the question. |

For example, `contexteval experiment top_k=3,10 rerank=none,lexical` runs five settings: the baseline and four combinations. The table shows accuracy, cases passed, mean input tokens, mean latency, and total app cost per setting. Experiments never see the evidence passages, so a setting cannot win by peeking at the answer.

## Sweep

`contexteval sweep` chooses the experiments for you from the labels of the latest diagnosed baseline run. Run it after `contexteval diagnose`.

Each label asks for different settings. The sweep tries the ones the labels you have call for, relative to how many chunks your app retrieved at baseline (k):

| Label | Settings tried | Why |
|---|---|---|
| `missing`, never retrieved | `top_k` at 2k and 3k | The right chunk ranked below k. |
| `irrelevant` | `rerank=lexical`, `max_chunks` at k/2, `compress=2` | Other chunks crowded the right one out. |
| `ignored` | `order=ends`, `order=reverse`, `compress=2` | The model skipped a chunk in the middle. |
| `missing`, dropped | none | Retrieval found the chunk and your own code removed it before the prompt. No retrieval setting changes that filter. |
| `not_context` | none | The context held the answer and the model still got it wrong. Look at the prompt, the model, or the expected answer. |
| `flaky` | none | The unchanged app passed on rerun. A setting cannot make the model deterministic. |

A setting that more failures ask for is tried first. `--max-settings N` caps how many are tried per round (default 6).

The search runs in rounds. Round 1 runs the baseline and each setting on its own. Round 2 combines the best setting so far with each round-1 setting that changes a different knob and did not lose to the baseline. `--rounds N` sets the depth (default 2). A round with nothing new to try ends the search early, so a run whose labels are all `missing` stops after round 1.

To keep the sweep from fitting your test set, it splits the cases into a tune set and a holdout set with a fixed seed (`--holdout F`, default 0.3 of the cases, half the failures and half the passes in proportion; `--seed S` changes the split). It chooses on the tune cases, then runs the baseline and the winner on the holdout cases. It recommends the winner only when it beat the baseline on tune and did not lose on holdout. Otherwise it recommends keeping your current settings and says why. `--holdout 0` tunes on every case; use it for a set too small to split, and read the result as unconfirmed.

Among settings that pass about as many tune cases, the sweep prefers the cheaper one. Every setting that beat the baseline and passed within a near-tie window of the most accurate one is a contender. The window is 5% of the tune cases, rounded down, so it is 0 cases on a set under 20, 1 case on 35, and 5 cases on 100. `--tolerance N` sets it to N cases instead. The contender with the fewest mean input tokens wins, then the one with the lowest latency, then the one that passed more. The same rule picks the best setting in every round. When the winner is not the most accurate setting, the recommendation names the one it passed over and what that one costs. When the window is not 0, the sweep prints it before running, as in `near-tie window: 1 case (5% of 35)`. A window of 0, the default on a small set or `--tolerance 0`, picks the most accurate setting, with ties going to the cheaper one.

Before running, the sweep prints the worst case, every round full plus the two holdout runs, and asks. Every candidate is a plain experiment setting, so the sweep never sees the evidence passages either.

The whole output for the demo (`contexteval sweep --holdout 0 --yes` after `diagnose`):

```text
sweep: run 1, 6 labeled failures (not_context 1, flaky 1, irrelevant 1, ignored 1, missing/not_retrieved 1, missing/dropped 1); the baseline retrieved up to 4 chunks
  no setting can fix 1 not_context failure: the context held the answer and the model still got it wrong; look at the prompt, the model, or the expected answer
  no setting can fix 1 flaky failure: the unchanged app passed on rerun; a setting cannot make the model deterministic
  no setting can fix 1 missing/dropped failure: retrieval found the gold chunk and your own code removed it before the prompt; no retrieval setting changes that filter
candidates, heaviest first (6 of 7; the number is how many failures ask for it): compress=2 (2), top_k=8 (1), top_k=12 (1), rerank=lexical (1), max_chunks=2 (1), order=ends (1)
split: 8 tune cases, 0 holdout cases (seed 0)
warning: no holdout cases, so the recommendation is not confirmed on cases the search did not see
app: up to 104 calls to contexteval.demo.app:answer (8 tune cases x (1 baseline + 6 settings x 2 rounds) + 0 holdout cases x 2, worst case)
estimate: 104 LLM calls, ~5824 input and ~208 output tokens, about $0.0000
experiment 1 (sweep)
round 1, 8 tune cases
setting         accuracy  passed  mean input tokens  mean latency ms  total cost
baseline        38%       3/8     124                35.6             unknown
compress=2      38%       3/8     124                3.8              unknown
top_k=8         50%       4/8     217                4.1              unknown
top_k=12        50%       4/8     323                4.4              unknown
rerank=lexical  50%       4/8     124                5.3              unknown
max_chunks=2    38%       3/8     74                 3.3              unknown
order=ends      38%       3/8     124                3.4              unknown
round 2, 8 tune cases
setting                      accuracy  passed  mean input tokens  mean latency ms  total cost
baseline                     38%       3/8     124                35.6             unknown
rerank=lexical compress=2    50%       4/8     124                3.9              unknown
top_k=8 rerank=lexical       62%       5/8     217                3.6              unknown
top_k=12 rerank=lexical      62%       5/8     323                3.8              unknown
rerank=lexical max_chunks=2  50%       4/8     74                 3.8              unknown
rerank=lexical order=ends    50%       4/8     124                3.9              unknown
Set top_k=8 rerank=lexical. Accuracy 38% -> 62% (no holdout cases, so this is unconfirmed). Uses 76% more input tokens; mean latency 36 ms -> 4 ms.
```

`top_k=8` retrieves the chunk that ranked 5th, and `rerank=lexical` moves the ignored case's chunk to the top, so `top_k=8 rerank=lexical` passes the most cases. On 8 tune cases the window rounds down to 0, so accuracy decides, and `top_k=8 rerank=lexical` beats `top_k=12 rerank=lexical`, which passes as many cases, on input tokens. With `--tolerance 1` the sweep would take `rerank=lexical max_chunks=2`, which passes one case fewer with 74 input tokens instead of 217 and leaves the planted not_retrieved case failing. The baseline row's latency includes the demo's one-time index warm-up, as the `experiment` table's does.

The sweep is stored as an experiment. The dashboard lists it with its rows and marks the recommended one.

## Dashboard

`contexteval ui` serves a local page with every run, each failure with its label and evidence line, the retrieved and delivered chunks, and the experiment tables, including each sweep with its recommendation. It binds to `127.0.0.1` only and rejects requests with any other `Host` header.

## Cost and privacy

Everything stays on your machine. contexteval stores test cases, traces (prompts, answers, and chunks), and labels in `.contexteval/contexteval.db`, a local SQLite file. There is no hosted service and no telemetry.

LLM calls go to your provider with your own key, read from `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. The default models are `gpt-4.1-mini` for OpenAI and `claude-haiku-4-5` for Anthropic. Change them with `contexteval init --model`. contexteval's own calls are never recorded as traces. They run at temperature 0 on every model that accepts it (see Limits).

The estimate counts characters divided by 4 as tokens, priced from a built-in table. It assumes every answer needs an LLM grade, so the real cost is usually lower. contexteval does not report actual spend after a command. Your provider's usage page has the exact figure.

## Limits

- Hooked calls are chromadb `Collection.query` and `AsyncCollection.query`, pinecone `Index.query`, LangChain `BaseRetriever.invoke` and `ainvoke`, LlamaIndex `BaseRetriever.retrieve` and `aretrieve`, OpenAI chat completions and responses (`OpenAI` and `AsyncOpenAI`), and Anthropic messages (`Anthropic` and `AsyncAnthropic`). The pinecone hook is tested against mocks only, and pinecone's async index is not hooked.
- The framework hooks are tested with langchain-core 1.6, langchain-chroma, and llama-index-core 0.14. `ChatOpenAI` and the LlamaIndex OpenAI LLM are covered through the SDK hooks; the test suite exercises that path with a stand-in chat model that calls the openai SDK, not the framework LLM packages themselves.
- A `top_k` experiment on a LangChain or LlamaIndex retriever needs `search_kwargs`, a `k` field, or `similarity_top_k` on the retriever, or a hooked call underneath it that has a knob (a chromadb collection it queries, a `base_retriever` it wraps). Without any of these the experiment stops with an error instead of running the baseline. A per-call `k` passed to `invoke` is overridden too. The knob is set per call and restored, also when the call raises. Two runs that want different `top_k` values on one shared retriever at the same time are not supported; the CLI runs one question at a time, so this does not arise there.
- The outermost hooked retrieval owns the event. A `@contexteval.retriever` function that queries chromadb inside records one retrieval, as a framework retriever does. If it runs that query on a thread pool of its own, the inner query is recorded as well, because the thread has no copy of the calling context.
- A retriever class that overrides `invoke` or `retrieve` itself and calls child retrievers (an ensemble or fusion retriever) is not hooked at the top, because the hook is on the base class method it replaced. Each child's call is then the outermost hooked retrieval, so one question records one event per child, and an experiment's `top_k` and transforms apply to each child rather than to the merged result.
- In production, `init()` hooks LangChain and LlamaIndex only when the app imported them before the call. The CLI hooks every library it can import.
- `@contexteval.retriever` accepts `async def` functions, and `answer` may be `async def`. The CLI runs an async `answer` on one event loop for the whole command, so clients bound to the loop stay valid across questions. In production each asyncio task has its own capture buffer.
- pgvector has no client library to hook. Use `@contexteval.retriever` on your query function.
- Production traces are split by a heuristic. Each LLM call closes one trace that holds the retrievals since the previous LLM call. An agent that calls the LLM several times per question produces several traces.
- Cases run one at a time, in the same process as the CLI.
- An experiment setting applies to every retrieval call in a run.
- A chromadb call with several query texts is recorded but never changed by an experiment.
- Replay runs are not stored as traces. Their outcomes are stored inside the label.
- Current Claude models such as `claude-haiku-5-5` and `claude-sonnet-5-5` reject non-default sampling parameters. On those models contexteval omits `temperature`, so grading and judging do not run at temperature 0 and can vary between runs. The Anthropic default stays `claude-haiku-4-5` for that reason.

## Roadmap

- v2. LangChain and LlamaIndex support (done). An automatic sweep that picks experiments from the labels (done, `contexteval sweep`).
- v3. A background optimizer that replays sampled production traces and suggests changes. OpenTelemetry import.
- v4. Support for long-running tool agents. A hosted team version.

## License

MIT. See [LICENSE](LICENSE).
