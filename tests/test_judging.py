"""Tests for the graded rubric evaluator (raglab#8).

Hermetic: the judge is a scripted test double keyed by artifact id; no model,
no network. Content comes from each hit's snippet (nothing to load from disk).
"""

import json
import random

from ir import SearchHit

from raglab import (
    DEFAULT_LEVELS,
    Budget,
    GradedLabel,
    Judgement,
    SubTask,
    make_graded_rubric_evaluator,
    make_search_agent,
)

TASK = SubTask(goal="graph library", sources=("pkgs",))


def _hit(aid, text, score=1.0):
    return SearchHit(aid, "k", score, text, {})


def _judge(replies):
    """A judge returning ``replies[artifact_id]`` and recording what it was shown."""
    calls = []

    def judge(**kw):
        calls.append(kw)
        reply = replies[kw["artifact_id"]]
        if isinstance(reply, Exception):
            raise reply
        return reply

    judge.calls = calls
    return judge


def _reply(level, *, about=True, supported=True, useful=True, quote=""):
    return {
        "level": level,
        "is_about": about,
        "is_supported": supported,
        "is_useful": useful,
        "evidence_quote": quote,
    }


POOL = [
    _hit("graphy", "A graph library: nodes, edges, traversals."),
    _hit("au", "An async task framework; jobs form a DAG."),
    _hit("plotter", "Draws charts with networkx graph layouts."),
]


def test_every_candidate_is_graded_and_the_supported_gate_caps_on_word_distractors():
    judge = _judge(
        {
            "graphy": _reply("core", quote="A graph library"),
            # on-word but not supported: the judge over-grades, the gate caps it
            "au": _reply("strong", supported=False, quote="jobs form a DAG"),
            "plotter": _reply("uses-tools", quote="networkx graph layouts"),
        }
    )
    judged = make_graded_rubric_evaluator(judge=judge)(TASK, POOL)
    assert isinstance(judged, Judgement)
    assert {aid: g.level for aid, g in judged.graded.items()} == {
        "graphy": "core",
        "au": "tangential",
        "plotter": "uses-tools",
    }
    assert [h.artifact_id for h in judged.relevant] == ["graphy", "plotter"]
    assert judged.sufficient and judged.refinement is None
    assert len(judge.calls) == 3


def test_a_quote_not_found_verbatim_caps_the_level_unless_quotes_are_optional():
    judge = _judge({"graphy": _reply("core", quote="the best graph toolkit")})
    strict = make_graded_rubric_evaluator(judge=judge)(TASK, POOL[:1])
    assert strict.graded["graphy"].level == "tangential"
    assert strict.graded["graphy"].evidence_quote == ""
    lax = make_graded_rubric_evaluator(judge=judge, require_evidence_quote=False)(
        TASK, POOL[:1]
    )
    assert lax.graded["graphy"].level == "core"


def test_the_calibration_offset_demotes_an_over_grading_judge():
    judge = _judge({"graphy": _reply("core", quote="graph library")})
    ev = make_graded_rubric_evaluator(judge=judge, calibration_offset=1.0)
    assert ev(TASK, POOL[:1]).graded["graphy"].level == "strong"


def test_a_failing_or_unparseable_judge_leaves_the_candidate_ungraded():
    judge = _judge(
        {
            "graphy": json.dumps(_reply("core", quote="graph library")),  # raw text ok
            "au": RuntimeError("rate limited"),
            "plotter": '{"level": "excellent"}',  # not a level
        }
    )
    judged = make_graded_rubric_evaluator(judge=judge)(TASK, POOL)
    assert set(judged.graded) == {"graphy"}
    assert [h.artifact_id for h in judged.relevant] == ["graphy"]


def test_level_options_are_shown_in_a_fresh_order_each_call():
    judge = _judge({h.artifact_id: _reply("none") for h in POOL})
    ev = make_graded_rubric_evaluator(judge=judge, rng=random.Random(0))
    for _ in range(4):
        ev(TASK, POOL)
    orders = {c["levels"] for c in judge.calls}
    assert len(orders) > 1
    assert all(sorted(o.split("\n")) == sorted(f"- {lv}" for lv in DEFAULT_LEVELS) for o in orders)


def test_the_refine_policy_drives_the_back_edge_through_the_agent():
    corpus = {
        "graph library": [_hit("au", "An async task framework; jobs form a DAG.")],
        "graph library nodes edges": [_hit("graphy", "A graph library: nodes, edges.")],
    }

    def retriever(query, **kw):
        return list(corpus.get(query, []))

    judge = _judge(
        {
            "au": _reply("core", supported=False, quote="jobs form a DAG"),
            "graphy": _reply("core", quote="A graph library"),
        }
    )

    def refine(task, graded):
        if any(g.level in ("core", "strong") for g in graded.values()):
            return None
        return SubTask(goal=task.goal + " nodes edges", sources=task.sources)

    agent = make_search_agent(
        {"pkgs": retriever},
        evaluator=make_graded_rubric_evaluator(judge=judge, refine=refine),
        budget=Budget(max_rounds=3),
    )
    assert [h.artifact_id for h in agent("graph library")] == ["graphy"]


def test_binary_judgements_are_unchanged():
    assert Judgement(relevant=[], sufficient=True).graded == {}
    label = GradedLabel("core", True, True, True)
    assert label.evidence_quote == "" and label.facet is None


def test_unknown_level_names_are_refused():
    import pytest

    with pytest.raises(ValueError, match="relevant_level"):
        make_graded_rubric_evaluator(judge=_judge({}), relevant_level="great")


def test_the_same_id_in_two_sources_is_judged_per_source():
    a = SearchHit("x", "k", 1.0, "A graph library for nodes.", {}, source="A")
    b = SearchHit("x", "k", 1.0, "An audio decoder.", {}, source="B")

    def judge(**kw):
        if "graph" in kw["content"]:
            return _reply("core", quote="A graph library")
        return _reply("none")

    judged = make_graded_rubric_evaluator(judge=judge)(TASK, [a, b])
    assert [h.source for h in judged.relevant] == ["A"]
    assert judged.graded["x"].level == "core"  # the best label across sources


def test_custom_levels_match_whatever_their_case():
    judge = _judge({"graphy": _reply("core", quote="A graph library")})
    ev = make_graded_rubric_evaluator(
        judge=judge, levels=("Core", "Tangential", "None"),
        unsupported_cap="Tangential", relevant_level="Core",
    )
    assert ev(TASK, POOL[:1]).graded["graphy"].level == "Core"


def test_a_judge_that_fails_on_every_candidate_raises():
    import pytest

    judge = _judge({h.artifact_id: RuntimeError("no key") for h in POOL})
    with pytest.raises(RuntimeError, match="no key"):
        make_graded_rubric_evaluator(judge=judge)(TASK, POOL)


def test_a_candidate_is_judged_once_per_goal_and_its_label_kept():
    judge = _judge({"graphy": _reply("core", quote="A graph library")})
    ev = make_graded_rubric_evaluator(judge=judge)
    ev(TASK, POOL[:1])
    judge_down = ev(TASK, POOL[:1])  # no new call: the label is remembered
    assert len(judge.calls) == 1
    assert [h.artifact_id for h in judge_down.relevant] == ["graphy"]


def test_a_half_level_offset_shifts_every_level_alike():
    from raglab import apply_gates

    content = "A graph library."
    shifted = [
        apply_gates(GradedLabel(lv, True, True, True, "graph library"), content,
                    calibration_offset=0.5).level
        for lv in DEFAULT_LEVELS[:-1]
    ]
    assert shifted == list(DEFAULT_LEVELS[1:])


def test_a_one_word_quote_is_not_evidence():
    from raglab import apply_gates

    label = GradedLabel("core", True, True, True, "a")
    assert apply_gates(label, "A graph library.").level == "tangential"
