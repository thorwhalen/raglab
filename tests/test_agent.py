"""Tests for the raglab Composable Search Agent foundation (ir_09).

Hermetic: the control loop is exercised with a fake in-memory retriever (no
model, no network); one end-to-end test wires a real ``ir`` corpus via
``ir.as_retriever`` with the light (numpy-only) embedder and an in-memory store.
"""

import raglab
from ir import SearchHit
from raglab import Budget, Judgement, LowLevelQuery, Query, SubTask, make_search_agent


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


# ----- value types & protocols --------------------------------------------- #


def test_result_is_ir_searchhit():
    assert raglab.Result is SearchHit


def test_defaults_satisfy_role_protocols():
    assert isinstance(raglab.single_subtask_planner, raglab.Planner)
    assert isinstance(raglab.identity_formulator, raglab.Formulator)
    assert isinstance(raglab.passthrough_evaluator, raglab.Evaluator)
    assert isinstance(raglab.score_reranker, raglab.Reranker)
    assert isinstance(raglab.identity_citer, raglab.Citer)


# ----- the thin-slice loop (no LLM) ----------------------------------------- #


def test_make_search_agent_thin_slice_runs():
    sources = {"s": _fake_retriever(_hits(("a", 0.9), ("b", 0.5)))}
    results = make_search_agent(sources)("anything")
    assert [r.artifact_id for r in results] == ["a", "b"]


def test_query_string_or_object_equivalent():
    sources = {"s": _fake_retriever(_hits(("a", 0.9)))}
    agent = make_search_agent(sources)
    assert agent("q") == agent(Query(text="q"))


def test_cross_source_merge_reranks_by_score():
    sources = {
        "s1": _fake_retriever(_hits(("a", 0.3))),
        "s2": _fake_retriever(_hits(("b", 0.9))),
    }
    results = make_search_agent(sources)("q")
    assert [r.artifact_id for r in results] == ["b", "a"]  # by score desc


def test_passthrough_evaluator_does_not_loop():
    retr = _fake_retriever(_hits(("a", 0.9)))
    make_search_agent({"s": retr})("q")
    assert len(retr.calls) == 1  # one round only


def test_formulator_fan_out_issues_each_query():
    retr = _fake_retriever(_hits(("a", 0.9)))

    def multi(task, source):
        return [
            LowLevelQuery(source, task.goal),
            LowLevelQuery(source, task.goal + " alt"),
        ]

    make_search_agent({"s": retr}, formulator=multi)("q")
    assert len(retr.calls) == 2


def test_unknown_source_is_skipped_not_an_error():
    retr = _fake_retriever(_hits(("a", 0.9)))

    def planner(query, sources):
        return [SubTask(query.text, ("s", "missing"))]

    results = make_search_agent({"s": retr}, planner=planner)("q")
    assert [r.artifact_id for r in results] == ["a"]


# ----- the back-edge (the property that makes it an agent) ------------------ #


def test_back_edge_reformulates_until_sufficient():
    retr = _fake_retriever(_hits(("a", 0.9)))
    rounds = {"n": 0}

    def refining_evaluator(task, results):
        rounds["n"] += 1
        if rounds["n"] < 2:  # first round: not enough, re-query (back-edge)
            return Judgement(
                relevant=list(results),
                sufficient=False,
                refinement=SubTask(goal=task.goal + " more", sources=task.sources),
            )
        return Judgement(relevant=list(results), sufficient=True)

    make_search_agent({"s": retr}, evaluator=refining_evaluator)("q")
    assert rounds["n"] == 2  # looped once via the back-edge
    assert len(retr.calls) == 2


def test_budget_bounds_a_never_sufficient_loop():
    retr = _fake_retriever(_hits(("a", 0.9)))

    def never_sufficient(task, results):
        return Judgement(
            relevant=list(results),
            sufficient=False,
            refinement=SubTask(task.goal, task.sources),
        )

    make_search_agent(
        {"s": retr}, evaluator=never_sufficient, budget=Budget(max_rounds=3)
    )("q")
    assert len(retr.calls) == 3  # exactly max_rounds — the safety net holds


# ----- end-to-end over a REAL ir corpus (hermetic: light embedder) ---------- #


def test_agent_over_real_ir_corpus():
    import ir
    from ir.store import CorpusStore

    docs = {
        "deploy": "deploy the app to the server with systemd units",
        "embed": "embed text with a model and cache the vectors",
        "search": "vector similarity search with metadata filters",
    }
    corpus = ir.build(
        ir.CorpusSource.from_mapping(docs, name="t", strategy=ir.WholeText()),
        store=CorpusStore.memory(),
        embedder="light",
    )
    agent = make_search_agent({"t": ir.as_retriever(corpus, k=3)})
    results = agent("deploy the app to the server")
    assert results
    assert results[0].artifact_id == "deploy"
    assert isinstance(results[0], SearchHit)
    results[0].to_dict()  # the substrate edge is serialization-clean
