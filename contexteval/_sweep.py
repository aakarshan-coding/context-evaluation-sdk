"""The diagnosis-guided sweep (v2): labels in, one plain recommendation out.

The policy is a table from failure kind to candidate settings. The search is greedy over
rounds on a tune split and confirms the winner on a holdout split. Every run goes through
_runner.run_suite under one experiment, so a sweep is an experiment the dashboard already
shows, marked `sweep` and carrying its report.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields
from enum import Enum
from typing import Iterable, Literal, Mapping, Sequence, Union

from ._intervention import BASELINE, Order, Setting
from .types import CaseId, Diagnosis, Flaky, Ignored, Irrelevant, Judged, JudgeVerdict, Missing, NotContext, TestCase, never


class FailureKind(Enum):
    """A label, split by cause where the cause decides the fix. The value is what the report prints."""

    NOT_RETRIEVED = "missing/not_retrieved"
    DROPPED = "missing/dropped"
    IRRELEVANT = "irrelevant"
    IGNORED = "ignored"
    NOT_CONTEXT = "not_context"
    FLAKY = "flaky"


def kind_of(d: Diagnosis) -> FailureKind | None:
    """None for a judge verdict of no failure. A judged `missing` has no cause, so it takes the
    retrieval candidates: a larger top_k is the only setting that can help it."""
    match d:
        case Missing(cause="not_retrieved"):
            return FailureKind.NOT_RETRIEVED
        case Missing():
            return FailureKind.DROPPED
        case Irrelevant():
            return FailureKind.IRRELEVANT
        case Ignored():
            return FailureKind.IGNORED
        case NotContext():
            return FailureKind.NOT_CONTEXT
        case Flaky():
            return FailureKind.FLAKY
        case Judged(verdict=JudgeVerdict.NO_FAILURE):
            return None
        case Judged(verdict=v):
            return {
                JudgeVerdict.MISSING: FailureKind.NOT_RETRIEVED,
                JudgeVerdict.IRRELEVANT: FailureKind.IRRELEVANT,
                JudgeVerdict.IGNORED: FailureKind.IGNORED,
                JudgeVerdict.NOT_CONTEXT: FailureKind.NOT_CONTEXT,
            }[v]
        case _:
            never(d)


# --- policy -----------------------------------------------------------------------


@dataclass(frozen=True)
class Scaled:
    """An integer knob set relative to k, the most chunks one retrieval returned in the baseline
    run: top_k=2k widens retrieval, max_chunks=k/2 narrows the prompt."""

    knob: Literal["top_k", "max_chunks"]
    factor: float

    def resolve(self, k: int) -> Setting | None:
        """None when the value would change nothing: a max_chunks at or above k, or top_k == k."""
        value = max(1, round(k * self.factor))
        if (self.knob == "max_chunks" and value >= k) or (self.knob == "top_k" and value == k):
            return None
        return Setting(**{self.knob: value})  # type: ignore[arg-type]


Candidate = Union[Setting, Scaled]


@dataclass(frozen=True)
class Fixable:
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class Unfixable:
    reason: str  # shown in the report; no setting is tried for this kind


# The label-to-candidates policy (design-notes "diagnosis-guided search"). Table order inside a
# kind is the tie-break between candidates of equal weight.
POLICY: Mapping[FailureKind, Fixable | Unfixable] = {
    FailureKind.NOT_RETRIEVED: Fixable((Scaled("top_k", 2), Scaled("top_k", 3))),
    FailureKind.DROPPED: Unfixable(
        "retrieval found the gold chunk and your own code removed it before the prompt; "
        "no retrieval setting changes that filter"
    ),
    FailureKind.IRRELEVANT: Fixable((Setting(rerank="lexical"), Scaled("max_chunks", 0.5), Setting(compress=2))),
    FailureKind.IGNORED: Fixable((Setting(order=Order.ENDS), Setting(order=Order.REVERSE), Setting(compress=2))),
    FailureKind.NOT_CONTEXT: Unfixable(
        "the context held the answer and the model still got it wrong; look at the prompt, the model, or the expected answer"
    ),
    FailureKind.FLAKY: Unfixable("the unchanged app passed on rerun; a setting cannot make the model deterministic"),
}


def candidates_for(kind: FailureKind, k: int) -> list[Setting]:
    rule = POLICY[kind]
    if isinstance(rule, Unfixable):
        return []
    settings = []
    for candidate in rule.candidates:
        setting = candidate.resolve(k) if isinstance(candidate, Scaled) else candidate
        if setting is not None and setting != BASELINE and setting not in settings:
            settings.append(setting)
    return settings


@dataclass(frozen=True)
class Ranked:
    setting: Setting
    weight: int  # failures whose label proposes this setting
    kinds: tuple[FailureKind, ...]


def rank_candidates(counts: Mapping[FailureKind, int], k: int) -> list[Ranked]:
    """Heaviest first; a setting two labels propose carries both counts. Ties keep policy order."""
    weight: dict[Setting, int] = {}
    kinds: dict[Setting, list[FailureKind]] = {}
    for kind in FailureKind:
        n = counts.get(kind, 0)
        if n == 0:
            continue
        for setting in candidates_for(kind, k):
            weight[setting] = weight.get(setting, 0) + n
            kinds.setdefault(setting, []).append(kind)
    ranked = [Ranked(s, weight[s], tuple(kinds[s])) for s in weight]
    return sorted(ranked, key=lambda r: -r.weight)


def knobs(setting: Setting) -> frozenset[str]:
    """The knobs a setting changes from the baseline."""
    return frozenset(f.name for f in fields(setting) if getattr(setting, f.name) != f.default)


def merge(a: Setting, b: Setting) -> Setting:
    """a with b's changed knobs applied. Callers combine settings with disjoint knobs only."""
    values = {f.name: getattr(a, f.name) for f in fields(a)}
    values.update({name: getattr(b, name) for name in knobs(b)})
    return Setting(**values)  # type: ignore[arg-type]


# --- split ------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    seed: int
    holdout_fraction: float
    tune: tuple[CaseId, ...]
    holdout: tuple[CaseId, ...]


def split_cases(cases: Sequence[TestCase], failed: Iterable[CaseId], holdout: float, seed: int) -> Split:
    """Seeded and stratified: the baseline's failures and passes are each shuffled and split, so
    both sides hold failures the candidates can fix. Holdout counts are rounded per stratum, so
    a tiny set can end with an empty holdout; the caller warns about that."""
    failed = set(failed)
    rng = random.Random(seed)
    tune: list[CaseId] = []
    held: list[CaseId] = []
    for stratum in ({c.id for c in cases if c.id in failed}, {c.id for c in cases if c.id not in failed}):
        ids = sorted(stratum)
        rng.shuffle(ids)
        n = round(holdout * len(ids))
        held += ids[:n]
        tune += ids[n:]
    return Split(seed, holdout, tuple(sorted(tune)), tuple(sorted(held)))
