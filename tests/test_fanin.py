"""Tests for the Reranker at fan-in — rank-based cross-source merge (ir_09 §3).

The property under test: raw scores never cross a source boundary. Within one
source they order and dedup that source's hits; across sources only ranks
interact (RRF via ``ir.fuse_hits``). Hermetic: fake retrievers with canned
hits; one end-to-end test wires two REAL ir corpora (light embedder, in-memory
stores) whose artifact ids deliberately collide.
"""

import pytest

import raglab
from ir import SearchHit
from raglab import (
    Judgement,
    SubTask,
    make_rrf_reranker,
    make_search_agent,
    rrf_reranker,
)


def _hits(*specs, source=None):
    """``(artifact_id, score)`` pairs -> ir.SearchHits (optionally source-tagged)."""
    return [
        SearchHit(aid, "k", score, f"text {aid}", {}, source) for aid, score in specs
    ]


def _fake_retriever(hits):
    """A Retriever that records its calls and returns canned hits."""
    calls = []

    def retrieve(query, **kw):
        calls.append((query, kw))
        return list(hits)

    retrieve.calls = calls
    return retrieve


# ----- rrf_reranker: the role in isolation ---------------------------------- #


def test_heterogeneous_scales_interleave_by_rank():
    # A cosine-scale source (~[0,1]) and a BM25-scale source (~[0,50]): a raw
    # score sort would bury the cosine source entirely; rank fusion interleaves.
    pool = _hits(("c1", 0.92), ("c2", 0.85), source="cos") + _hits(
        ("b1", 31.0), ("b2", 24.0), source="bm25"
    )
    fused = rrf_reranker(pool)
    assert {h.artifact_id for h in fused[:2]} == {"c1", "b1"}  # both rank-1s lead
    assert {h.artifact_id for h in fused[2:]} == {"c2", "b2"}


def test_colliding_ids_across_sources_stay_distinct():
    pool = _hits(("dol", 0.9), source="skills") + _hits(("dol", 28.0), source="pkgs")
    fused = rrf_reranker(pool)
    assert len(fused) == 2  # same id, different corpus = different artifact
    assert {h.source for h in fused} == {"skills", "pkgs"}


def test_single_source_pool_keeps_raw_scores():
    pool = _hits(("a", 0.9), ("b", 0.5), source="s")
    fused = rrf_reranker(pool)
    assert [h.score for h in fused] == [0.9, 0.5]  # = score_reranker's ordering
    assert list(fused) == list(raglab.score_reranker(pool))


def test_untagged_pool_keeps_raw_scores_and_no_stamp():
    pool = _hits(("a", 0.9), ("b", 0.5))  # no source anywhere
    fused = rrf_reranker(pool)
    assert [h.score for h in fused] == [0.9, 0.5]
    assert all(h.source is None for h in fused)  # passthrough, no "" stamping


def test_cross_round_duplicates_collapse_per_source():
    # The same (source, artifact) re-retrieved across back-edge rounds counts
    # once, at its best rank — no duplicate ids, no double RRF mass.
    pool = (
        _hits(("a", 0.7), ("a", 0.9), source="s1")
        + _hits(("z", 5.0), source="s2")
        + _hits(("a", 0.8), source="s1")
    )
    fused = rrf_reranker(pool)
    keyed = [(h.source, h.artifact_id) for h in fused]
    assert len(keyed) == len(set(keyed)) == 2
    a = next(h for h in fused if h.artifact_id == "a")
    assert a.metadata["source_score"] == 0.9  # the best raw magnitude survives
    assert a.metadata["source_rank"] == 1


def test_make_rrf_reranker_weights_bias_the_merge():
    pool = _hits(("a", 0.9), source="s1") + _hits(("b", 0.9), source="s2")
    fused = make_rrf_reranker(weights={"s2": 2.0})(pool)
    assert fused[0].artifact_id == "b"  # trust dial, no score comparability needed


# ----- the agent loop: tagging + fan-in ------------------------------------- #


def test_loop_stamps_registry_key_on_untagged_hits():
    sources = {
        "s1": _fake_retriever(_hits(("a", 0.9))),
        "s2": _fake_retriever(_hits(("b", 7.0))),
    }
    results = make_search_agent(sources)("q")
    assert {h.source for h in results} == {"s1", "s2"}


def test_retriever_self_attribution_wins_over_registry_key():
    # An ir-backed retriever stamps the corpus name itself; the loop must not
    # overwrite it with the (possibly different) registry key.
    sources = {"alias": _fake_retriever(_hits(("a", 0.9), source="corpus_x"))}
    results = make_search_agent(sources)("q")
    assert [h.source for h in results] == ["corpus_x"]


def test_agent_keeps_colliding_ids_from_two_sources():
    sources = {
        "skills": _fake_retriever(_hits(("dol", 0.9))),
        "pkgs": _fake_retriever(_hits(("dol", 28.0))),
    }
    results = make_search_agent(sources)("q")
    assert sorted((h.source, h.artifact_id) for h in results) == [
        ("pkgs", "dol"),
        ("skills", "dol"),
    ]


def test_back_edge_rounds_do_not_duplicate_across_sources():
    sources = {
        "s1": _fake_retriever(_hits(("a", 0.9))),
        "s2": _fake_retriever(_hits(("a", 6.0))),  # same id, other source
    }
    rounds = {"n": 0}

    def refining(task, results):
        rounds["n"] += 1
        if rounds["n"] < 3:
            return Judgement(
                list(results),
                sufficient=False,
                refinement=SubTask(task.goal, task.sources),
            )
        return Judgement(list(results), sufficient=True)

    results = make_search_agent(sources, evaluator=refining)("q")
    keyed = [(h.source, h.artifact_id) for h in results]
    assert sorted(keyed) == [("s1", "a"), ("s2", "a")]  # one per (source, artifact)


# ----- the LLM evaluator's per-round prerank --------------------------------- #


def test_llm_evaluator_prerank_is_scale_safe_by_default():
    from raglab import make_llm_evaluator

    # A round's pool already mixes sources: the per-round merge must keep both
    # same-id artifacts (distinct sources) visible to ir.select and the judge.
    pool = _hits(("readme", 0.9), source="s1") + _hits(("readme", 30.0), source="s2")
    evaluator = make_llm_evaluator(judge=lambda *, goal, results: (True, None))
    judgement = evaluator(SubTask("g", ("s1", "s2")), pool)
    assert sorted((h.source, h.artifact_id) for h in judgement.relevant) == [
        ("s1", "readme"),
        ("s2", "readme"),
    ]


def test_llm_evaluator_prerank_is_injectable():
    from raglab import make_llm_evaluator, score_reranker

    pool = _hits(("readme", 0.9), source="s1") + _hits(("readme", 30.0), source="s2")
    evaluator = make_llm_evaluator(
        judge=lambda *, goal, results: (True, None), prerank=score_reranker
    )
    judgement = evaluator(SubTask("g", ("s1", "s2")), pool)
    # The explicit magnitude opt-in keeps both too (identity is per source)…
    # but ranks them by raw score: the BM25-scale hit wins the top slot.
    assert judgement.relevant[0].source == "s2"


# ----- end-to-end over two REAL ir corpora (hermetic: light embedder) -------- #


def test_agent_federates_two_real_corpora_with_colliding_ids():
    import ir
    from ir.store import CorpusStore

    def corpus(docs, name):
        return ir.build(
            ir.CorpusSource.from_mapping(docs, name=name, strategy=ir.WholeText()),
            store=CorpusStore.memory(),
            embedder="light",
        )

    docs_a = {"shared": "zebra zephyr zucchini", "alpha": "alpha apple avocado"}
    docs_b = {"shared": "zebra zephyr zucchini", "beta": "beta banana blueberry"}
    agent = make_search_agent(
        {
            "one": ir.as_retriever(corpus(docs_a, "one"), k=3),
            "two": ir.as_retriever(corpus(docs_b, "two"), k=3),
        }
    )
    results = agent("zebra zephyr zucchini")
    keyed = {(h.source, h.artifact_id) for h in results}
    # The colliding "shared" artifact survives from BOTH corpora, attributed.
    assert {("one", "shared"), ("two", "shared")} <= keyed
    assert results[0].to_dict()["source"] in {"one", "two"}  # serialization-clean
    fused_scores = [h.score for h in results]
    assert fused_scores == sorted(fused_scores, reverse=True)  # best-first


def test_score_reranker_remains_the_homogeneous_opt_in():
    pytest.importorskip("ir")
    pool = _hits(("a", 0.3), source="s1") + _hits(("b", 0.9), source="s2")
    assert [h.artifact_id for h in raglab.score_reranker(pool)] == ["b", "a"]
