"""Tests for the corpus-labeling recipe (raglab#8, part B). Hermetic: a fake retriever and a scripted judge."""

import pytest
from ir import SearchHit

from raglab import Budget
from raglab.recipes import CorpusLabeling, label_corpus

DOCS = {
    "graphy": "A graph library: nodes, edges, traversals.",
    "au": "An async task framework; jobs form a DAG.",
    "layout": "Graph layout algorithms for drawing networks.",
    "decoder": "An audio decoder.",
}
INDEX = {
    "graph data structures": ["graphy", "au"],
    "graph drawing": ["decoder"],
    "network layout algorithms": ["layout"],
}


def _retriever(query, **kw):
    return [SearchHit(aid, "k", 1.0 - i / 10, DOCS[aid], {}) for i, aid in enumerate(INDEX.get(query, []))]


def _judge(*, goal, artifact_id, content, **kw):
    level = {"graphy": "core", "au": "strong", "layout": "core", "decoder": "none"}[artifact_id]
    supported = artifact_id != "au"
    quote = content[:20]
    return {"level": level, "is_about": True, "is_supported": supported, "is_useful": supported, "evidence_quote": quote}


def test_every_facet_is_labelled_and_the_gate_holds():
    result = label_corpus(
        None,
        facets={"structures": "graph data structures", "drawing": "graph drawing"},
        retriever=_retriever,
        judge=_judge,
    )
    assert isinstance(result, CorpusLabeling)
    assert {aid: g.level for aid, g in result.labels.items()} == {
        "graphy": "core",
        "au": "tangential",
        "decoder": "none",
    }
    assert result.labels["graphy"].facet == "structures"
    assert result.coverage["structures"].sufficient
    assert not result.coverage["drawing"].sufficient
    assert result.at_least("strong") == ["graphy"]


def test_a_thin_facet_is_reformulated_and_judged_against_its_own_goal():
    goals_seen = []

    def judge(**kw):
        goals_seen.append(kw["goal"])
        return _judge(**kw)

    result = label_corpus(
        None,
        facets={"drawing": "graph drawing"},
        retriever=_retriever,
        judge=judge,
        reformulate=lambda goal, labels: "network layout algorithms",
        budget=Budget(max_rounds=3),
    )
    assert result.labels["layout"].level == "core"
    assert result.coverage["drawing"].rounds == 2 and result.coverage["drawing"].sufficient
    assert set(goals_seen) == {"graph drawing"}


def test_a_reformulation_that_repeats_itself_stops():
    result = label_corpus(
        None,
        facets=["graph drawing"],
        retriever=_retriever,
        judge=_judge,
        reformulate=lambda goal, labels: goal,
        budget=Budget(max_rounds=5),
    )
    assert result.coverage["graph drawing"].rounds == 1


def test_facets_are_required():
    with pytest.raises(ValueError, match="at least one facet"):
        label_corpus(None, facets=[], retriever=_retriever, judge=_judge)
