"""``raglab`` — the agentic-search / RAG orchestration layer on top of ``ir``.

`raglab` turns a retrieval substrate into a *Composable Search Agent* (the ir_09
architecture): a small set of injected **roles** — Planner, Formulator,
Retriever, Evaluator, Reranker, Citer — wired by a control loop whose back-edge
(evaluator → reformulate) is what makes it an agent rather than a DAG. Concrete
tools live at the leaves: an ``ir`` corpus becomes one ``Retriever`` via
``ir.as_retriever``.

Quick start (the no-LLM thin slice — runs offline)::

    import ir
    import raglab

    # register ir corpora as the agent's sources, then search across them:
    sources = raglab.ir_sources("skills", "reports", mode="hybrid")
    agent = raglab.make_search_agent(sources)
    results = agent("how do I deploy the app")      # ranked ir.SearchHits

Inject an LLM ``formulator`` (query rewrite/HyDE) and ``evaluator`` (sufficiency
+ refinement) to turn on query understanding and the back-edge. Dependency
direction is one-way: ``raglab`` imports ``ir``; ``ir`` never imports ``raglab``.

> Fresh start (v0.2.0+). This repo took over the ``raglab`` PyPI name; the older
> backend now lives at ``raglab_bak``. Development is just beginning.
"""

from .agent import (
    Budget,
    Citer,
    Evaluator,
    Formulator,
    Judgement,
    LowLevelQuery,
    Planner,
    Query,
    Reranker,
    Result,
    Retriever,
    SingleContextAgent,
    SubTask,
    identity_citer,
    identity_formulator,
    ir_sources,
    make_rrf_reranker,
    make_search_agent,
    passthrough_evaluator,
    rrf_reranker,
    score_reranker,
    single_subtask_planner,
)
from .llm import EVALUATION_PROMPT, make_llm_evaluator, make_llm_formulator

__all__ = [
    "Query",
    "SubTask",
    "LowLevelQuery",
    "Judgement",
    "Result",
    "Retriever",
    "Planner",
    "Formulator",
    "Evaluator",
    "Reranker",
    "Citer",
    "Budget",
    "SingleContextAgent",
    "make_search_agent",
    "ir_sources",
    "single_subtask_planner",
    "identity_formulator",
    "passthrough_evaluator",
    "score_reranker",
    "rrf_reranker",
    "make_rrf_reranker",
    "identity_citer",
    "make_llm_formulator",
    "make_llm_evaluator",
    "EVALUATION_PROMPT",
]
