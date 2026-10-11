# v1.1 plan

v1.1 makes v1 trustworthy and installable before v2 adds features. Source of the scope: the real-bot audit (`examples/realbot/AUDIT.md`) and the gap review.

## Definition of done

1. Diagnosis replays take a majority over 3 samples per step by default (`--samples N`, odd, at least 1). A step stops sampling once a majority is decided. Stored diagnoses record pass counts per step, and the CLI and dashboard show them (for example "gold to top 2/3"). The planted demo labels do not change, and a test proves a single lucky pass no longer decides a label.
2. Async apps work: `AsyncOpenAI` chat completions and responses, `AsyncAnthropic` messages, chromadb's async collection query, `async def` functions under `@contexteval.retriever`, and an `async def answer` entry point. Interventions apply to async retrievals exactly as to sync ones. Production capture works per asyncio task.
3. The package is release-ready: version 0.2.0, complete PyPI metadata, sdist and wheel build and pass `twine check`, a GitHub Actions CI workflow runs the suite on Python 3.10 to 3.13, and a release workflow publishes to PyPI on a version tag through PyPI trusted publishing (no stored token). Publishing itself waits for the owner.
4. The full suite passes, and the README documents all of the above.

## Workstreams

| Stream | Owns | Acceptance |
|---|---|---|
| A. Majority replays | `_diagnose.py`, `types.py` diagnosis types, `_store.py` diagnosis encoding, `_cli.py` diagnose command, `_ui.py`/`ui.html` replay display | Item 1 |
| B. Async | `_capture.py`, `_hooks.py`, `_runner.py`, `_intervention.py` if needed | Item 2 |
| C. Release | `pyproject.toml`, `.github/workflows/`, `README.md` install section, `CHANGELOG.md` | Item 3 |
