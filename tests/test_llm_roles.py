"""Tests for raglab's LLM-backed roles (the Formulator and Evaluator).

Hermetic and deterministic: every "LLM" is an injected test double (no model, no
network). The Formulator adapts an ir-style ``str -> [str]`` rewriter; the
Evaluator delegates relevance to ``ir.select`` and sufficiency to the injected
judge. One end-to-end demo wires a real ``ir`` corpus (the light, numpy-only
embedder, in-memory store) and shows the **back-edge** recovering a gold document
that single-shot retrieval misses.
"""

import ir
from ir import SearchHit
from ir.store import CorpusStore

from raglab import (
    Budget,
    LowLevelQuery,
    SubTask,
    make_llm_evaluator,
    make_llm_formulator,
    make_search_agent,
)


def _hits(*specs):
    """``(artifact_id, score)`` pairs -> ir.SearchHits (no corpus needed)."""
    return [SearchHit(aid, "k", score, f"text {aid}", {}) for aid, score in specs]


def _fake_retriever(hits):
    """A Retriever that records its calls and returns canned hits."""
    calls = []

    def retrieve(query, **kw):
        calls.append((query, kw))
        return list(hits)

    retrieve.calls = calls
    return retrieve


# ----- LLM Formulator: adapt str -> [str] to (SubTask, source) -> [LLQ] ------ #


def test_llm_formulator_fans_out_one_llq_per_query():
    formulator = make_llm_formulator(formulate=lambda q: [q, q + " alt"])
    llqs = formulator(SubTask(goal="deploy", sources=("s",)), "s")
    assert [q.query for q in llqs] == ["deploy", "deploy alt"]
    assert all(isinstance(q, LowLevelQuery) and q.source == "s" for q in llqs)


def test_llm_formulator_accepts_a_bare_string():
    formulator = make_llm_formulator(formulate=lambda q: q + "!")
    llqs = formulator(SubTask(goal="x", sources=("s",)), "s")
    assert [q.query for q in llqs] == ["x!"]


def test_llm_formulator_attaches_params_to_every_query():
    formulator = make_llm_formulator(
        formulate=lambda q: [q, q + " b"], params={"mode": "hybrid", "k": 5}
    )
    llqs = formulator(SubTask(goal="g", sources=("s",)), "s")
    assert all(q.params == {"mode": "hybrid", "k": 5} for q in llqs)


def test_llm_formulator_empty_output_falls_back_to_the_goal():
    # A formulator must never make retrieval worse than the raw sub-goal.
    formulator = make_llm_formulator(formulate=lambda q: [])
    llqs = formulator(SubTask(goal="the goal", sources=("s",)), "s")
    assert [q.query for q in llqs] == ["the goal"]


def test_llm_formulator_swallows_a_raising_callable():
    # A failing custom formulator must fall back to the goal, never propagate.
    def boom(_q):
        raise RuntimeError("rewriter down")

    formulator = make_llm_formulator(formulate=boom)
    llqs = formulator(SubTask(goal="the goal", sources=("s",)), "s")
    assert [q.query for q in llqs] == ["the goal"]


def test_llm_formulator_drives_the_agent_loop():
    retr = _fake_retriever(_hits(("a", 0.9)))
    formulator = make_llm_formulator(formulate=lambda q: [q, q + " expanded"])
    make_search_agent({"s": retr}, formulator=formulator)("q")
    assert [c[0] for c in retr.calls] == ["q", "q expanded"]


# ----- LLM Evaluator: ir.select owns relevance, the LLM owns sufficiency ----- #


def test_evaluator_relevance_comes_from_ir_select():
    # Conservative selection keeps only the near-top hit; "b" is a distractor.
    evaluator = make_llm_evaluator(judge=lambda **kw: (True, None))
    judged = evaluator(SubTask("g", ("s",)), _hits(("a", 0.9), ("b", 0.1)))
    assert [h.artifact_id for h in judged.relevant] == ["a"]
    assert judged.sufficient and judged.refinement is None


def test_evaluator_emits_a_refinement_when_insufficient():
    evaluator = make_llm_evaluator(judge=lambda **kw: (False, "better query"))
    judged = evaluator(SubTask("g", ("s1", "s2")), _hits(("a", 0.9)))
    assert judged.sufficient is False
    assert judged.refinement == SubTask(goal="better query", sources=("s1", "s2"))


def test_evaluator_insufficient_without_a_query_stops_the_loop():
    # Insufficient but no refinement query -> nothing better to try -> stop.
    evaluator = make_llm_evaluator(judge=lambda **kw: (False, None))
    judged = evaluator(SubTask("g", ("s",)), _hits(("a", 0.9)))
    assert judged.sufficient is True and judged.refinement is None


def test_evaluator_parses_a_raw_text_reply():
    evaluator = make_llm_evaluator(
        judge=lambda **kw: "INSUFFICIENT\nvector database filtering"
    )
    judged = evaluator(SubTask("g", ("s",)), _hits(("a", 0.9)))
    assert judged.refinement.goal == "vector database filtering"


def test_evaluator_judge_error_falls_back_to_signal_no_loop():
    def boom(**kw):
        raise RuntimeError("model down")

    evaluator = make_llm_evaluator(judge=boom)
    judged = evaluator(SubTask("g", ("s",)), _hits(("a", 0.9)))
    # refinement=None is the loop's break condition: a judge error never spins.
    assert judged.refinement is None
    assert judged.sufficient is True  # ir.select committed to "a" -> sufficient


def test_evaluator_renders_abstention_to_the_judge():
    seen = {}

    def judge(*, goal, results):
        seen["results"] = results
        return (True, None)

    evaluator = make_llm_evaluator(judge=judge)
    evaluator(SubTask("g", ("s",)), [])  # no results -> ir.select abstains
    assert "abstained" in seen["results"]


def test_evaluator_forwards_select_kwargs_to_ir_select():
    # Two near-tied hits: conservative keeps only "a" by default, but a loose
    # rel threshold (forwarded via select_kwargs) admits "b" too.
    hits = _hits(("a", 0.9), ("b", 0.6))
    strict = make_llm_evaluator(judge=lambda **kw: (True, None))
    loose = make_llm_evaluator(
        judge=lambda **kw: (True, None), select_kwargs={"rel": 0.5}
    )
    assert [h.artifact_id for h in strict(SubTask("g", ("s",)), hits).relevant] == ["a"]
    assert [h.artifact_id for h in loose(SubTask("g", ("s",)), hits).relevant] == [
        "a",
        "b",
    ]


def test_evaluator_ranks_heterogeneous_results_before_selecting():
    # Accumulated cross-source results arrive unordered; the evaluator must rank
    # best-first before ir.select (which trusts input order).
    captured = {}

    def judge(*, goal, results):
        captured["results"] = results
        return (True, None)

    evaluator = make_llm_evaluator(judge=judge, select_kwargs={"rel": 0.0})
    unordered = _hits(("lo", 0.1), ("hi", 0.9), ("mid", 0.5))
    judged = evaluator(SubTask("g", ("s",)), unordered)
    assert [h.artifact_id for h in judged.relevant] == ["hi", "mid", "lo"]


# ----- the back-edge end-to-end, wired through the agent loop ---------------- #


def test_evaluator_back_edge_loops_until_sufficient():
    retr = _fake_retriever(_hits(("a", 0.9)))
    rounds = {"n": 0}

    def judge(*, goal, results):
        rounds["n"] += 1
        if rounds["n"] < 2:
            return (False, goal + " more")
        return (True, None)

    make_search_agent({"s": retr}, evaluator=make_llm_evaluator(judge=judge))("q")
    assert rounds["n"] == 2  # looped once via the back-edge
    assert [c[0] for c in retr.calls] == ["q", "q more"]  # refinement re-queried


def test_evaluator_back_edge_is_bounded_by_budget():
    retr = _fake_retriever(_hits(("a", 0.9)))
    evaluator = make_llm_evaluator(judge=lambda **kw: (False, "again"))
    make_search_agent({"s": retr}, evaluator=evaluator, budget=Budget(max_rounds=3))(
        "q"
    )
    assert len(retr.calls) == 3  # the safety net holds even if never sufficient


# ----- end-to-end over a REAL ir corpus (hermetic: light embedder) ---------- #


def _light_corpus():
    docs = {
        "embed": "embed and cache model vectors",
        "systemd": "configure systemd units and restart services",
        "filtering": "narrow similarity search using metadata filters",
    }
    return ir.build(
        ir.CorpusSource.from_mapping(docs, name="t", strategy=ir.WholeText()),
        store=CorpusStore.memory(),
        embedder="light",
    )


def test_back_edge_recovers_a_doc_single_shot_misses():
    """A query that overlaps a distractor misses the gold; the refinement recovers it.

    Deterministic with the light embedder: the round-1 query shares vocabulary
    with ``embed`` (a positive-score distractor) but none with the gold
    ``filtering``, so single-shot ranks ``embed`` first; the injected judge
    declares it insufficient and reformulates to the gold doc's own vocabulary, so
    round 2 retrieves ``filtering`` to the top via the back-edge.
    """
    corpus = _light_corpus()
    sources = {"t": ir.as_retriever(corpus, k=3)}
    vague = "cache model results"  # overlaps the `embed` distractor, not the gold
    gold_query = "narrow similarity search using metadata filters"

    # Baseline: single-shot (no LLM evaluator) surfaces the distractor, not the gold.
    baseline = make_search_agent(sources)(vague)
    assert baseline[0].artifact_id == "embed"

    # With the back-edge: reformulate to the gold's vocabulary, then it wins.
    rounds = {"n": 0}

    def judge(*, goal, results):
        rounds["n"] += 1
        if rounds["n"] < 2:
            return (False, gold_query)
        return (True, None)

    agent = make_search_agent(sources, evaluator=make_llm_evaluator(judge=judge))
    results = agent(vague)
    assert rounds["n"] == 2  # the back-edge fired
    assert results[0].artifact_id == "filtering"  # gold recovered
    assert isinstance(results[0], SearchHit)
