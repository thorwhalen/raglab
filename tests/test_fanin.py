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


def test_llm_evaluator_relevant_carries_prefusion_magnitudes():
    from raglab import make_llm_evaluator

    # Judgement.relevant re-enters the loop's pool, so it must carry RAW
    # per-source scores — the fused (ordinal) view is local to selection.
    pool = _hits(("a", 0.9), source="s1") + _hits(("b", 30.0), source="s2")
    evaluator = make_llm_evaluator(judge=lambda *, goal, results: (True, None))
    judgement = evaluator(SubTask("g", ("s1", "s2")), pool)
    assert {h.score for h in judgement.relevant} == {0.9, 30.0}  # not ~1/61
    assert all("source_rank" not in h.metadata for h in judgement.relevant)


def test_agent_with_llm_evaluator_never_mixes_fused_and_raw_scores():
    from raglab import make_llm_evaluator

    # Regression for the double-fusion feedback: round-1 fused scores must not
    # re-enter the pool, or round 2 compares ordinal RRF values against raw
    # magnitudes INSIDE one source group (and source_score gets overwritten by
    # an already-fused value).
    def varying_retriever(rounds_hits):
        state = {"round": 0}

        def retrieve(query, **kw):
            hits = rounds_hits[min(state["round"], len(rounds_hits) - 1)]
            state["round"] += 1
            return list(hits)

        return retrieve

    sources = {
        "s1": varying_retriever([_hits(("a1", 0.9)), _hits(("a2", 0.5))]),
        "s2": varying_retriever([_hits(("b1", 30.0)), _hits(("b2", 24.0))]),
    }
    verdicts = iter([(False, "refined q"), (True, None)])
    evaluator = make_llm_evaluator(judge=lambda *, goal, results: next(verdicts))
    results = make_search_agent(
        sources, evaluator=evaluator, formulator=raglab.identity_formulator
    )("q")
    raw = {("s1", "a1"): 0.9, ("s1", "a2"): 0.5, ("s2", "b1"): 30.0, ("s2", "b2"): 24.0}
    for h in results:
        assert h.metadata["source_score"] == raw[(h.source, h.artifact_id)]
    # Within one source, the final order follows raw magnitudes.
    s1 = [h.artifact_id for h in results if h.source == "s1"]
    assert s1 == sorted(s1, key=lambda a: -raw[("s1", a)])


def test_llm_evaluator_min_score_works_single_source_raises_multi():
    import pytest

    from raglab import make_llm_evaluator

    judge = lambda *, goal, results: (True, None)  # noqa: E731
    single = _hits(("a", 0.9), ("b", 0.1), source="s1")
    evaluator = make_llm_evaluator(judge=judge, select_kwargs={"min_score": 0.5})
    judgement = evaluator(SubTask("g", ("s1",)), single)
    assert [h.artifact_id for h in judgement.relevant] == ["a"]  # floor met raw scores

    multi = _hits(("a", 0.9), source="s1") + _hits(("b", 30.0), source="s2")
    with pytest.raises(ValueError, match="rank-fused"):
        evaluator(SubTask("g", ("s1", "s2")), multi)  # never silently mass-abstain


def test_mixed_tagged_untagged_pool_keeps_none_provenance():
    pool = _hits(("u", 0.5)) + _hits(("t", 9.0), source="s1")
    fused = rrf_reranker(pool)
    by_id = {h.artifact_id: h for h in fused}
    assert by_id["u"].source is None  # untagged pseudo-source, never ""
    assert by_id["t"].source == "s1"
    assert by_id["u"].to_dict()["source"] is None


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


# ----- the budget cap: rank-fair, scale-safe (raglab#5) ---------------------- #


def _capturing_evaluator(seen):
    def evaluator(task, results):
        seen.append(list(results))
        return Judgement(list(results), sufficient=True)

    return evaluator


def test_small_cap_lets_both_disparate_sources_top_hits_through():
    # source "big" arrives first with many BM25-scale hits; "cos" arrives last with
    # cosine-scale hits. Arrival-order truncation would never show "cos" at all.
    big = _hits(*[(f"b{i}", 40.0 - i) for i in range(10)], source="big")
    cos = _hits(("c_best", 0.95), ("c2", 0.4), source="cos")
    seen = []
    make_search_agent(
        {"big": _fake_retriever(big), "cos": _fake_retriever(cos)},
        evaluator=_capturing_evaluator(seen),
        budget=raglab.Budget(max_results_per_task=4),
    )("q")
    pool = seen[0]
    assert len(pool) == 4
    ids = [h.artifact_id for h in pool]
    assert ids[:2] == ["b0", "c_best"]  # each source's best, dealt round-robin
    assert set(ids) == {"b0", "b1", "c_best", "c2"}


def test_capped_pool_keeps_raw_per_source_magnitudes():
    big = _hits(*[(f"b{i}", 40.0 - i) for i in range(10)], source="big")
    cos = _hits(("c_best", 0.95), ("c2", 0.4), source="cos")
    originals = {(h.source, h.artifact_id): h for h in big + cos}
    seen = []
    make_search_agent(
        {"big": _fake_retriever(big), "cos": _fake_retriever(cos)},
        evaluator=_capturing_evaluator(seen),
        budget=raglab.Budget(max_results_per_task=3),
    )("q")
    for hit in seen[0]:
        assert hit == originals[(hit.source, hit.artifact_id)]  # untouched
        assert "source_rank" not in hit.metadata
        assert "source_score" not in hit.metadata  # no fused-score stamp either


def test_rank_fair_cap_orders_within_source_by_its_own_score():
    # one source, scores out of arrival order (e.g. two formulator queries)
    pool = _hits(("lo", 0.1), ("hi", 0.9), ("mid", 0.5), source="s")
    assert [h.artifact_id for h in raglab.rank_fair_cap(pool, 2)] == ["hi", "mid"]
    # untagged hits are one group of their own
    mixed = _hits(("u1", 3.0), ("u2", 2.0)) + _hits(("t1", 0.5), source="t")
    assert [h.artifact_id for h in raglab.rank_fair_cap(mixed, 2)] == ["u1", "t1"]


def test_rank_fair_cap_within_limit_and_degenerate_limits():
    pool = _hits(("a", 0.1), ("b", 0.9), source="s")
    # within budget nothing is cut, and the order is the same dealt order as above it
    assert [h.artifact_id for h in raglab.rank_fair_cap(pool, 5)] == ["b", "a"]
    assert raglab.rank_fair_cap(pool, 0) == []
    assert raglab.rank_fair_cap(pool, -1) == []
    uneven = _hits(*[(f"x{i}", 1.0 - i / 10) for i in range(5)], source="x") + _hits(
        ("y0", 9.0), source="y"
    )
    # a short source runs dry; the long one fills the rest
    assert [h.artifact_id for h in raglab.rank_fair_cap(uneven, 4)] == [
        "x0",
        "y0",
        "x1",
        "x2",
    ]


def test_pool_cap_is_injectable_and_arrival_order_is_the_opt_out():
    big = _hits(*[(f"b{i}", 40.0 - i) for i in range(10)], source="big")
    cos = _hits(("c_best", 0.95), source="cos")
    seen = []
    make_search_agent(
        {"big": _fake_retriever(big), "cos": _fake_retriever(cos)},
        evaluator=_capturing_evaluator(seen),
        budget=raglab.Budget(max_results_per_task=3),
        pool_cap=raglab.arrival_order_cap,
    )("q")
    assert [h.artifact_id for h in seen[0]] == ["b0", "b1", "b2"]
    assert isinstance(raglab.rank_fair_cap, raglab.PoolCap)


def test_cap_collapses_duplicates_and_tolerates_missing_scores():
    dup = _hits(("a", 0.9), ("a", 0.5), ("b", 0.8), ("c", 0.7), source="s")
    assert [h.artifact_id for h in raglab.rank_fair_cap(dup, 3)] == ["a", "b", "c"]
    assert raglab.rank_fair_cap(dup, 3)[0].score == 0.9  # the best surface survives
    odd = _hits(("nan", float("nan")), ("none", None), ("ok", 0.1), source="s")
    assert raglab.rank_fair_cap(odd, 1)[0].artifact_id == "ok"


def _refine_once(seen, keep):
    """An evaluator that keeps only `keep` in round one and asks for a re-query."""

    def evaluator(task, results):
        seen.append(list(results))
        if len(seen) == 1:
            kept = [h for h in results if h.artifact_id in keep]
            return Judgement(
                kept, sufficient=False, refinement=SubTask("again", task.sources)
            )
        return Judgement(list(results), sufficient=True)

    return evaluator


def _rounds(*per_round):
    """A retriever returning a different canned hit list on each call."""
    calls = iter(per_round)
    return lambda query, **kw: list(next(calls))


def test_back_edge_never_evicts_a_hit_the_evaluator_kept():
    round1 = _hits(
        ("weak_but_relevant", 0.1), ("junk1", 0.9), ("junk2", 0.8), source="s"
    )
    round2 = _hits(("new1", 0.95), ("new2", 0.85), ("new3", 0.7), source="s")
    seen = []
    out = make_search_agent(
        {"s": _rounds(round1, round2)},
        evaluator=_refine_once(seen, {"weak_but_relevant"}),
        budget=raglab.Budget(max_results_per_task=3),
    )("q")
    assert [h.artifact_id for h in seen[1]] == ["weak_but_relevant", "new1", "new2"]
    assert "weak_but_relevant" in {h.artifact_id for h in out}


def test_back_edge_does_not_spend_slots_on_re_retrieved_hits():
    same = _hits(("a", 0.9), ("b", 0.8), ("c", 0.7), source="s")
    seen = []
    make_search_agent(
        {"s": _rounds(same, same)},
        evaluator=_refine_once(seen, {"a"}),
        budget=raglab.Budget(max_results_per_task=3),
    )("q")
    assert [h.artifact_id for h in seen[1]] == ["a", "b", "c"]  # no second "a"
