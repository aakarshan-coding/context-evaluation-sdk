"""v2 auto-sweep: the label-to-candidates policy, the seeded split, the greedy search, and the
recommendation, through `_cli.main` where the user would see it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contexteval import _cli, _sweep
from contexteval._intervention import BASELINE, GoldToTop, InjectEvidence, OnlyGold, Order, Setting
from contexteval._sweep import FailureKind, Fixable, Unfixable
from contexteval._store import Store
from contexteval.types import (
    CaseId,
    Evidence,
    Flaky,
    Ignored,
    Irrelevant,
    Judged,
    JudgeVerdict,
    Label,
    Missing,
    NotContext,
    ReplayOutcome,
    ReplayStep,
    TestCase,
)

GOLD_TRANSFORMS = (InjectEvidence, GoldToTop, OnlyGold)
RERUN = (ReplayOutcome(ReplayStep.BASELINE_RERUN, False, "wrong"),)
K = 4


def _knobs(setting: Setting) -> dict[str, object]:
    return {name: value for name, value in setting.to_json().items() if value != BASELINE.to_json()[name]}


# --- policy -----------------------------------------------------------------------


def test_policy_maps_each_label_to_its_candidates_and_the_unfixable_to_none() -> None:
    """Defect: a label whose candidates do not address it (a top_k change for an irrelevant case,
    a reorder for a never-retrieved chunk), a fixable label with no candidates, or dropped,
    not_context, and flaky offered a setting when no setting can change them."""
    assert set(_sweep.POLICY) == set(FailureKind)
    resolved = {kind: [_knobs(s) for s in _sweep.candidates_for(kind, K)] for kind in FailureKind}

    assert resolved[FailureKind.NOT_RETRIEVED] and all(set(c) == {"top_k"} and c["top_k"] > K for c in resolved[FailureKind.NOT_RETRIEVED])  # type: ignore[operator]
    irrelevant = resolved[FailureKind.IRRELEVANT]
    assert {"rerank": "lexical"} in irrelevant
    assert any(set(c) == {"max_chunks"} and c["max_chunks"] < K for c in irrelevant)  # type: ignore[operator]
    assert any(set(c) == {"compress"} for c in irrelevant)
    ignored = resolved[FailureKind.IGNORED]
    assert {"order": "ends"} in ignored and {"order": "reverse"} in ignored and any(set(c) == {"compress"} for c in ignored)

    for kind in (FailureKind.DROPPED, FailureKind.NOT_CONTEXT, FailureKind.FLAKY):
        rule = _sweep.POLICY[kind]
        assert isinstance(rule, Unfixable) and rule.reason and resolved[kind] == []
    assert "your own code" in _sweep.POLICY[FailureKind.DROPPED].reason  # type: ignore[union-attr]
    for kind in (FailureKind.NOT_RETRIEVED, FailureKind.IRRELEVANT, FailureKind.IGNORED):
        assert isinstance(_sweep.POLICY[kind], Fixable)


def test_relative_candidates_follow_the_observed_baseline_k_and_drop_no_ops() -> None:
    """Defect: top_k candidates fixed at 10 whatever the app retrieves (a no-op for an app at
    k=12), or a max_chunks candidate at or above k, which changes nothing and wastes a run."""
    assert [_knobs(s)["top_k"] for s in _sweep.candidates_for(FailureKind.NOT_RETRIEVED, 12)] == [24, 36]
    assert [_knobs(s)["top_k"] for s in _sweep.candidates_for(FailureKind.NOT_RETRIEVED, 1)] == [2, 3]
    assert all(_knobs(s).get("max_chunks", 0) < 12 for s in _sweep.candidates_for(FailureKind.IRRELEVANT, 12))
    assert not any("max_chunks" in _knobs(s) for s in _sweep.candidates_for(FailureKind.IRRELEVANT, 1))


def test_every_diagnosis_has_a_failure_kind() -> None:
    """Defect: a judged label or a Missing cause the policy does not know, so a labeled failure
    silently contributes nothing to the search."""
    assert _sweep.kind_of(Missing("not_retrieved", True, RERUN)) is FailureKind.NOT_RETRIEVED
    assert _sweep.kind_of(Missing("dropped", True, RERUN)) is FailureKind.DROPPED
    assert _sweep.kind_of(Irrelevant(RERUN, ((0, 1),))) is FailureKind.IRRELEVANT
    assert _sweep.kind_of(Ignored(RERUN, ((0, 2),))) is FailureKind.IGNORED
    assert _sweep.kind_of(NotContext(RERUN)) is FailureKind.NOT_CONTEXT
    assert _sweep.kind_of(Flaky(RERUN)) is FailureKind.FLAKY
    for verdict in JudgeVerdict:
        kind = _sweep.kind_of(Judged(verdict, "why"))
        if verdict is JudgeVerdict.NO_FAILURE:
            assert kind is None
        else:
            assert kind is not None and kind.value.startswith(verdict.value)
    assert {kind.value.split("/")[0] for kind in FailureKind} == {label.value for label in Label}


def test_candidates_are_ranked_by_how_many_failures_carry_their_label() -> None:
    """Defect: candidates in table order regardless of the labels seen, so one ignored case
    ranks a reorder above the top_k change three missing cases call for, and --max-settings
    then cuts the wrong candidate; or a candidate shared by two labels counted once."""
    counts = {FailureKind.NOT_RETRIEVED: 3, FailureKind.IGNORED: 1, FailureKind.IRRELEVANT: 1, FailureKind.DROPPED: 2}
    ranked = _sweep.rank_candidates(counts, K)
    names = [r.setting.name for r in ranked]
    assert names[:2] == ["top_k=8", "top_k=12"] and [r.weight for r in ranked[:2]] == [3, 3]
    compress = next(r for r in ranked if r.setting.name.startswith("compress="))
    assert compress.weight == 2 and set(compress.kinds) == {FailureKind.IGNORED, FailureKind.IRRELEVANT}
    assert names.index(compress.setting.name) == 2
    assert len(names) == len(set(names))
    assert all(r.weight >= 1 for r in ranked) and ranked == sorted(ranked, key=lambda r: -r.weight)
    assert _sweep.rank_candidates({FailureKind.DROPPED: 2, FailureKind.FLAKY: 1}, K) == []


def test_no_candidate_setting_can_carry_gold() -> None:
    """Plan item 15 for the sweep. Defect: a candidate built from a Diagnosis or a TestCase that
    reaches the app with evidence in it, so a sweep row wins by peeking at the answer."""
    counts = {kind: 1 for kind in FailureKind}
    for ranked in _sweep.rank_candidates(counts, K):
        transforms = ranked.setting.intervention().transforms
        assert not any(isinstance(t, GOLD_TRANSFORMS) for t in transforms), ranked.setting.name
    assert all(isinstance(c, (Setting, _sweep.Scaled)) for rule in _sweep.POLICY.values() if isinstance(rule, Fixable) for c in rule.candidates)


def test_merge_combines_disjoint_knobs_and_names_overlaps() -> None:
    """Defect: a round-2 combination that silently lets one knob overwrite the other's value, or
    that reports two settings as combinable when they set the same knob."""
    a, b = Setting(top_k=8), Setting(order=Order.ENDS, compress=2)
    assert _sweep.merge(a, b) == Setting(top_k=8, order=Order.ENDS, compress=2)
    assert _sweep.knobs(a).isdisjoint(_sweep.knobs(b))
    assert not _sweep.knobs(Setting(top_k=8)).isdisjoint(_sweep.knobs(Setting(top_k=12)))
    assert _sweep.knobs(BASELINE) == frozenset()


# --- split ------------------------------------------------------------------------


def _cases(n: int) -> list[TestCase]:
    return [TestCase(CaseId(f"c{i:02d}"), f"q{i}?", f"a{i}", Evidence(f"e{i}"), "t") for i in range(n)]


def test_split_is_deterministic_by_seed_and_keeps_failures_on_both_sides() -> None:
    """Defect: a split that changes between two runs with the same seed (so a user cannot
    reproduce a recommendation), one that ignores the seed, or one that puts every failure on one
    side so the holdout check is vacuous."""
    cases = _cases(20)
    failed = {c.id for c in cases[:6]}
    first = _sweep.split_cases(cases, failed, holdout=0.3, seed=7)
    again = _sweep.split_cases(cases, failed, holdout=0.3, seed=7)
    other = _sweep.split_cases(cases, failed, holdout=0.3, seed=8)
    assert first == again and first.holdout != other.holdout
    assert set(first.tune) | set(first.holdout) == {c.id for c in cases} and not set(first.tune) & set(first.holdout)
    assert len(first.holdout) == 6  # round(0.3 * 6) failures + round(0.3 * 14) passes
    assert len(failed & set(first.holdout)) == 2 and len(failed & set(first.tune)) == 4
    assert first.tune == tuple(sorted(first.tune)) and first.holdout == tuple(sorted(first.holdout))


def test_split_with_no_holdout_keeps_every_case_for_tuning() -> None:
    """Defect: --holdout 0 still withholds a case, or a set too small to hold out loses a case
    silently instead of reporting an empty holdout."""
    cases = _cases(3)
    none = _sweep.split_cases(cases, {cases[0].id}, holdout=0.0, seed=1)
    assert none.holdout == () and set(none.tune) == {c.id for c in cases}
    tiny = _sweep.split_cases(cases[:1], {cases[0].id}, holdout=0.3, seed=1)
    assert tiny.holdout == () and tiny.tune == (cases[0].id,)
