# contexteval

contexteval tells you why your RAG app gave a wrong answer. It runs your app on test questions, records what was retrieved and what reached the prompt, and labels each failure.

Most RAG failures are context failures, and they come in three kinds:

- **Missing.** The right passage never reached the prompt. Retrieval did not find it, or your code dropped it after retrieval.
- **Irrelevant.** The right passage was in the prompt, but other chunks crowded it out. The model answers correctly once they are removed.
- **Ignored.** The right passage was in the prompt, and the model skipped over it. The model answers correctly once that passage moves to the top.

Each label points at a different fix. Missing calls for better retrieval, such as a larger `top_k`. Irrelevant calls for a reranker or fewer chunks. Ignored calls for a different chunk order or shorter chunks. contexteval then lets you test those fixes against your test set with one command.

You do not change your app. contexteval hooks chromadb, pinecone, OpenAI, and Anthropic from the outside. Custom retrieval needs one decorator.

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

The extras are `openai`, `anthropic`, `chroma`, `pinecone`, and `demo` (chromadb plus openai). Install the ones your app uses, for example `pip install "contexteval[openai,chroma]"`, or `pip install ".[openai,chroma]"` from source.

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

If your app queries chromadb or pinecone and calls OpenAI or Anthropic, it works as is. contexteval patches those clients before it imports your module.

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

## Dashboard

`contexteval ui` serves a local page with every run, each failure with its label and evidence line, the retrieved and delivered chunks, and the experiment tables. It binds to `127.0.0.1` only and rejects requests with any other `Host` header.

## Cost and privacy

Everything stays on your machine. contexteval stores test cases, traces (prompts, answers, and chunks), and labels in `.contexteval/contexteval.db`, a local SQLite file. There is no hosted service and no telemetry.

LLM calls go to your provider with your own key, read from `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. The default models are `gpt-4.1-mini` for OpenAI and `claude-haiku-4-5` for Anthropic. Change them with `contexteval init --model`. contexteval's own calls are never recorded as traces. They run at temperature 0 on every model that accepts it (see Limits).

The estimate counts characters divided by 4 as tokens, priced from a built-in table. It assumes every answer needs an LLM grade, so the real cost is usually lower. contexteval does not report actual spend after a command. Your provider's usage page has the exact figure.

## Limits in v1

- Hooked calls are chromadb `Collection.query` and `AsyncCollection.query`, pinecone `Index.query`, OpenAI chat completions and responses (`OpenAI` and `AsyncOpenAI`), and Anthropic messages (`Anthropic` and `AsyncAnthropic`). The pinecone hook is tested against mocks only, and pinecone's async index is not hooked.
- `@contexteval.retriever` accepts `async def` functions, and `answer` may be `async def`. The CLI runs an async `answer` on one event loop for the whole command, so clients bound to the loop stay valid across questions. In production each asyncio task has its own capture buffer.
- pgvector has no client library to hook. Use `@contexteval.retriever` on your query function.
- Production traces are split by a heuristic. Each LLM call closes one trace that holds the retrievals since the previous LLM call. An agent that calls the LLM several times per question produces several traces.
- Cases run one at a time, in the same process as the CLI.
- An experiment setting applies to every retrieval call in a run.
- A chromadb call with several query texts is recorded but never changed by an experiment.
- Replay runs are not stored as traces. Their outcomes are stored inside the label.
- Current Claude models such as `claude-haiku-5-5` and `claude-sonnet-5-5` reject non-default sampling parameters. On those models contexteval omits `temperature`, so grading and judging do not run at temperature 0 and can vary between runs. The Anthropic default stays `claude-haiku-4-5` for that reason.

## Roadmap

- v2. An automatic sweep that picks experiments from the labels: larger `top_k` for missing, reranking for irrelevant, reordering for ignored. LangChain and LlamaIndex plugins.
- v3. A background optimizer that replays sampled production traces and suggests changes. OpenTelemetry import.
- v4. Support for long-running tool agents. A hosted team version.

## License

MIT. See [LICENSE](LICENSE).
