# Real bot example

A support bot for Ridgeline Cycles, a fictional bike shop. It shows contexteval on a real RAG app with a real model instead of the scripted demo. `AUDIT.md` checks every label contexteval gave this bot by hand.

The bot in `bot.py` reads the 22 documents in `corpus/` and splits them into paragraph chunks of about 500 characters with `contexteval.chunk_text`. It embeds the chunks with `text-embedding-3-small` into a persistent chromadb collection at `./.realbot_chroma`, or at the path in `REALBOT_DB`. For each question it retrieves 4 chunks and asks `gpt-4.1-mini`. `answer` needs nothing from contexteval; the bot imports it only for the optional `build(chunking)` beside `answer`, which rebuilds the collection with another chunk size, overlap, or a document-title prefix so that `contexteval experiment` and `sweep` can try them. Each chunking gets its own collection, named by its chunks' digest.

`customer_questions.jsonl` holds 28 hand-written questions in a customer's voice, each with a verbatim evidence passage. They were written before the bot was run.

## Run it

You need `OPENAI_API_KEY` and the `openai` and `chroma` extras. Run every command from an empty project directory, with the repository root on `PYTHONPATH` so `examples.realbot.bot` imports.

```bash
export PYTHONPATH=/path/to/context-evaluation-sdk
contexteval init --provider openai --model gpt-4.1-mini
contexteval build-tests "$PYTHONPATH/examples/realbot/corpus" --per-passage 2 --yes
contexteval build-tests --import "$PYTHONPATH/examples/realbot/customer_questions.jsonl"
contexteval run examples.realbot.bot:answer --yes
contexteval diagnose --yes
contexteval experiment top_k=8 rerank=none,lexical --yes
contexteval experiment top_k=15 order=reverse,ends --yes
contexteval experiment title=on,off --yes
contexteval diagnose --run 5 --yes
```

`diagnose --run ID` labels the failures of one experiment row. `contexteval ui` lists the run ids.

The whole audit, including the stability replays below, made 1,675 OpenAI calls and cost $0.27.

## Sweep with chunking

`AUDIT.md` found the cause of every in-prompt failure in the chunker: a spec list cut from its heading carries no product name. `contexteval sweep` can now test the fix, because `bot.py` offers `build(chunking)`. On a fresh project (`init`, the import above, `build-tests --per-passage 1`, `run`, `diagnose`, `sweep` with defaults) the run passed 46 of 51, the labels were 4 `missing` and 1 `ignored` (a Trailhawk Pro spec with the gold at chunk 4), and the `ignored` label proposed `title=on`. On the 35 tune cases `title=on` passed 35 of 35 at 435 mean input tokens against the baseline's 32 of 35 at 399; `top_k=12` also passed 35, at 1104 tokens. On the 16 holdout cases the winner passed 16 of 16 against the baseline's 13. The line read:

```text
Set order=reverse title=on. Accuracy 91% -> 100% (holdout 81% -> 100%). Uses 9% more input tokens; mean latency 917 ms -> 868 ms.
```

`title=on` is the change to make. The `order=reverse` beside it won a one-token near-tie over `title=on` alone and adds nothing. The check made 1,138 OpenAI calls and cost $0.15, of which the sweep was $0.12.

## Audit tools

`audit/` holds the scripts behind `AUDIT.md`. They use contexteval internals, so they are not an example of app code. Run them from the project directory.

- `audit/ce.py ARGS` runs the contexteval CLI and appends each OpenAI request's token usage to `./tally.jsonl`. `audit/spend.py` turns that file into calls and dollars.
- `audit/dump.py RUN [failed|passed|all] [--full]` prints each trace with its retrieved chunks, gold marks, prompt membership, diagnosis, and replay answers.
- `audit/goldcheck.py RUN...` counts where the gold sat in every trace, split by pass and fail.
- `audit/stability.py N TRACE...` repeats each diagnosis replay N times under the run's setting.
- `audit/passes.py RUN` lists passing answers next to the expected answer. `audit/polarity.py` finds passes whose answer opens with the opposite yes or no.
