"""Local dashboard: stdlib http.server, JSON endpoints, one static page (plan item 11).

Routes are a table of pure functions Store -> JSON-able dict, so they are tested without a socket.
The page (ui.html, package data) fetches /api/* and renders the failure list and experiment table.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ._store import Store

Json = dict[str, Any]


def api_runs(store: Store, query: Mapping[str, str]) -> Json:
    raise NotImplementedError


def api_failures(store: Store, query: Mapping[str, str]) -> Json:
    """?run=ID (default latest baseline). Each failure: question, answer, label, source,
    evidence line (types.explain), chunks per retrieval with in_prompt and gold flags.
    """
    raise NotImplementedError


def api_experiments(store: Store, query: Mapping[str, str]) -> Json:
    raise NotImplementedError


def api_experiment(store: Store, query: Mapping[str, str]) -> Json:
    """?id=ID -> rows from Store.experiment_table."""
    raise NotImplementedError


ROUTES: Mapping[str, Callable[[Store, Mapping[str, str]], Json]] = {
    "/api/runs": api_runs,
    "/api/failures": api_failures,
    "/api/experiments": api_experiments,
    "/api/experiment": api_experiment,
}


def serve(store: Store, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Bind localhost only. GET / -> ui.html; GET ROUTES -> JSON; else 404. Blocks until Ctrl-C."""
    raise NotImplementedError
