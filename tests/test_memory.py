"""Tests for the purpose-indexed memory (raglab#6, stage 0). Hermetic: fakes only."""

import json

from ir import SearchHit

from raglab import Judgement, PurposeStore, make_search_agent
from raglab.memory import ExtractionUnit, make_relevance_recency_reader, purpose_id


def _hits(*specs, source=None):
    return [SearchHit(aid, "k", s, f"text {aid}", {}, source) for aid, s in specs]


def _fake_retriever(hits):
    calls = []

    def retrieve(query, **kw):
        calls.append(query)
        return list(hits)

    retrieve.calls = calls
    return retrieve


VOCAB = ("invoice", "invoices", "2023", "receipts", "cats")


def _bag_embedder(text):
    """A light fake embedder: word counts over a fixed vocabulary."""
    words = text.split()
    return [float(words.count(w)) + (w == "invoice") * words.count("invoices") for w in VOCAB]


def test_same_purpose_twice_reuses_prior_extractions():
    retriever = _fake_retriever(_hits(("a", 0.9), ("b", 0.4)))
    store = PurposeStore()
    agent = make_search_agent({"s": retriever}, memory=store)
    first = agent("Invoices from 2023")
    second = agent("invoices  from 2023")  # same purpose, once normalized
    assert len(retriever.calls) == 1  # the second run did not re-sweep
    assert [h.artifact_id for h in second] == [h.artifact_id for h in first]
    assert [h.score for h in second] == [h.score for h in first]


def test_insufficient_recall_still_sweeps_with_recalled_hits_kept():
    retriever = _fake_retriever(_hits(("c", 0.7)))
    store = PurposeStore()
    store.write("q", [ExtractionUnit("q", _hits(("a", 0.9), source="s")[0].to_dict(), 0.0)])
    seen = []

    def evaluator(task, pool):
        seen.append([h.artifact_id for h in pool])
        return Judgement(relevant=list(pool), sufficient=len(pool) > 1)

    out = make_search_agent({"s": retriever}, evaluator=evaluator, memory=store)("q")
    assert seen == [["a"], ["a", "c"]]
    assert {h.artifact_id for h in out} == {"a", "c"}


def test_related_purpose_recalls_via_embedding():
    store = PurposeStore(embedder=_bag_embedder)
    retriever = _fake_retriever(_hits(("a", 0.9)))
    agent = make_search_agent({"s": retriever}, memory=store)
    agent("invoices 2023")
    assert [u.hit["artifact_id"] for u in store["invoice 2023"]] == ["a"]
    assert "cats" not in store
    agent("2023 invoices")
    assert len(retriever.calls) == 1


def test_no_embedder_means_only_the_same_purpose_is_related():
    store = PurposeStore()
    store.write("invoices 2023", [ExtractionUnit("p", {"artifact_id": "a", "surface_kind": "k", "score": 1.0, "text": "", "metadata": {}, "source": "s", "surface_index": None}, 0.0)])
    assert "2023 invoices" not in store
    assert list(store) == ["invoices 2023"]


def test_records_are_json_clean_and_raw():
    backend = {}
    store = PurposeStore(backend=backend, clock=lambda: 100.0)
    make_search_agent({"s": _fake_retriever(_hits(("a", 0.9)))}, memory=store)("q")
    rec = json.loads(json.dumps(backend[purpose_id("q")]))
    (unit,) = rec["units"]
    assert unit["hit"]["score"] == 0.9 and unit["hit"]["source"] == "s"
    assert unit["recorded_at"] == 100.0 and unit["level"] == "raw"


def test_upsert_write_updates_the_same_artifact():
    store = PurposeStore()
    hit = _hits(("a", 0.9), source="s")[0].to_dict()
    store.write("q", [ExtractionUnit("q", hit, 1.0)])
    store.write("q", [ExtractionUnit("q", {**hit, "score": 0.5}, 2.0)])
    (unit,) = store["q"]
    assert unit.hit["score"] == 0.5 and unit.recorded_at == 2.0


def test_reader_prefers_recent_and_relevant():
    read = make_relevance_recency_reader(half_life=10.0)
    old = ExtractionUnit("p", {"artifact_id": "old", "source": "s"}, recorded_at=0.0)
    new = ExtractionUnit("p", {"artifact_id": "new", "source": "s"}, recorded_at=20.0)
    assert [u.hit["artifact_id"] for u in read([(1.0, old), (1.0, new)], now=20.0)] == ["new", "old"]
    assert [u.hit["artifact_id"] for u in read([(1.0, old), (0.1, new)], now=20.0)] == ["old", "new"]


def test_no_memory_leaves_the_agent_unchanged():
    hits = _hits(("a", 0.9), ("b", 0.4))
    assert make_search_agent({"s": _fake_retriever(hits)}).memory is None
    plain = make_search_agent({"s": _fake_retriever(hits)})("q")
    assert plain == make_search_agent({"s": _fake_retriever(hits)}, memory=PurposeStore())("q")
