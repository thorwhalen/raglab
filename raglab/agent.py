"""The Composable Search Agent — roles wired by a control loop (ir_09).

`raglab` is the **orchestration layer on top of `ir`** (the retrieval substrate).
This module is the v1 foundation: the immutable value types, the role *Protocols*
(open-closed strategy seams), and a `SingleContextAgent` whose fixed control loop
is fully parametrized by injected roles. Concrete tools live at the leaves — an
`ir` corpus becomes one `Retriever` via :func:`ir.as_retriever`.

The shape follows ir_09 §3/§6: a small set of named roles —
``Planner / Formulator / Retriever / Evaluator / Reranker / Citer`` — and a loop
whose defining feature is the **back-edge** (evaluator → reformulate) that makes
it an *agent* rather than a DAG. v1 ships the loop with smart defaults (a trivial
planner + a pass-through evaluator), so the thin slice runs end-to-end with no
LLM; turning on the back-edge is just injecting an LLM ``Evaluator`` that returns
a ``refinement`` (ir_09 §8 step 3).

Progressive disclosure: :func:`make_search_agent` gives sensible defaults for
every role, so the simple path is ``make_search_agent(sources)("query")``.

Dependency direction is one-way: `raglab` imports `ir`; `ir` never imports
`raglab`. The ``Result`` type and the ``Retriever`` contract are ir's (SSOT).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ir owns the retrieval substrate: the Result type and the Retriever leaf
# contract live there (one-way dependency, ir is the SSOT).
from ir import Retriever, SearchHit
from ir.base import best_per_artifact

#: A retrieved item — ir's :class:`~ir.base.SearchHit` (ir_09's ``Result``):
#: a *pointer + snippet* (``text``) with a ``score`` and ``metadata``.
Result = SearchHit

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
    "identity_citer",
]


# --------------------------------------------------------------------------- #
# Value types (immutable, plain data) — ir_09 §3
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Query:
    """A user intent: free text plus optional structured constraints."""

    text: str
    constraints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubTask:
    """A planner's unit of work: a sub-goal bound to a set of registered sources."""

    goal: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class LowLevelQuery:
    """One concrete query against one source.

    ``query`` is the text handed to the source's :data:`Retriever`; ``params``
    are per-call retriever overrides (e.g. ``mode`` / ``filter`` / ``k`` for an
    ir corpus). A formulator turns a :class:`SubTask` into these.
    """

    source: str
    query: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Judgement:
    """An evaluator's verdict over a round's results.

    ``relevant`` is the kept subset; ``sufficient`` says whether to stop; a
    non-``None`` ``refinement`` is the **back-edge** — the next sub-task to
    re-query with. The pass-through default returns ``sufficient=True`` and no
    refinement (no loop).
    """

    relevant: Sequence[Result]
    sufficient: bool
    refinement: SubTask | None = None


# --------------------------------------------------------------------------- #
# Role Protocols (the open-closed strategy seams) — ir_09 §3
# --------------------------------------------------------------------------- #


@runtime_checkable
class Planner(Protocol):
    """Decompose a query into sub-tasks and select sources for each."""

    def __call__(
        self, query: Query, sources: Mapping[str, Retriever]
    ) -> list[SubTask]: ...


@runtime_checkable
class Formulator(Protocol):
    """Turn a sub-task + one source into concrete low-level queries."""

    def __call__(self, task: SubTask, source: str) -> list[LowLevelQuery]: ...


@runtime_checkable
class Evaluator(Protocol):
    """Judge relevance + sufficiency; optionally emit a refinement (back-edge)."""

    def __call__(self, task: SubTask, results: Sequence[Result]) -> Judgement: ...


@runtime_checkable
class Reranker(Protocol):
    """Produce the final ordering over the (cross-source) merged results."""

    def __call__(self, results: Sequence[Result]) -> Sequence[Result]: ...


@runtime_checkable
class Citer(Protocol):
    """Confirm/annotate that each result supports its use (identity by default)."""

    def __call__(self, results: Sequence[Result]) -> Sequence[Result]: ...


# --------------------------------------------------------------------------- #
# Budget governor — ir_09 §4
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Budget:
    """Loop bounds: the safety net under the (harder) sufficiency decision."""

    max_rounds: int = 3
    max_sources_per_task: int = 4
    max_results_per_task: int = 50


# --------------------------------------------------------------------------- #
# Default role implementations (the no-LLM thin slice) — ir_09 §8 step 2
# --------------------------------------------------------------------------- #


def single_subtask_planner(
    query: Query, sources: Mapping[str, Retriever]
) -> list[SubTask]:
    """Trivial planner: one sub-task over *all* registered sources, no decomposition."""
    return [SubTask(goal=query.text, sources=tuple(sources))]


def identity_formulator(task: SubTask, source: str) -> list[LowLevelQuery]:
    """Identity formulator: the sub-goal verbatim as one query (no rewrite/HyDE)."""
    return [LowLevelQuery(source=source, query=task.goal)]


def passthrough_evaluator(task: SubTask, results: Sequence[Result]) -> Judgement:
    """Pass-through critic: keep everything, declare sufficient, never re-query."""
    return Judgement(relevant=list(results), sufficient=True, refinement=None)


def score_reranker(results: Sequence[Result]) -> Sequence[Result]:
    """Cross-source merge (v1): one surface per artifact, ordered by descending score.

    Delegates to :func:`ir.base.best_per_artifact` (ir is the SSOT for hit
    operations): an artifact retrieved by several queries / sources / rounds —
    common once the back-edge re-queries — survives once, at its highest score, so
    the merged list carries no duplicate ``artifact_id``. Also the evaluator's
    pre-selection rank, so :func:`ir.select` never sees duplicates either.

    Note: a plain score sort assumes comparable score scales across sources
    (true when they share an embedder + mode). A rank-based (RRF) cross-source
    merge for heterogeneous backends is a documented follow-up.
    """
    return best_per_artifact(results)


def identity_citer(results: Sequence[Result]) -> Sequence[Result]:
    """No-op citer (verification needs a generated claim — that lives in srag)."""
    return results


# --------------------------------------------------------------------------- #
# The orchestrator — a fixed control loop, fully parametrized by roles
# --------------------------------------------------------------------------- #


@dataclass
class SingleContextAgent:
    """One ReAct-style loop, sequential sub-tasks (ir_09 §7 — the cheap default).

    The loop is fixed; every *decision* is an injected role. Promotion to a
    multi-agent orchestrator (ir_09 §7) swaps this class while keeping the same
    role contracts. The **back-edge** is the single line ``current =
    judged.refinement`` in :meth:`_run_task` — that is what makes this an agent.
    """

    sources: Mapping[str, Retriever]
    planner: Planner = single_subtask_planner
    formulator: Formulator = identity_formulator
    evaluator: Evaluator = passthrough_evaluator
    reranker: Reranker = score_reranker
    citer: Citer = identity_citer
    budget: Budget = field(default_factory=Budget)

    def __call__(self, query: str | Query) -> list[Result]:
        """Run the agent for *query*; returns the final ranked, cited results."""
        q = query if isinstance(query, Query) else Query(text=query)
        accumulated: list[Result] = []
        for task in self.planner(q, self.sources):
            accumulated.extend(self._run_task(task))
        ranked = self.reranker(accumulated)
        return list(self.citer(ranked))

    def _run_task(self, task: SubTask) -> list[Result]:
        found: list[Result] = []
        current = task
        for _ in range(max(1, self.budget.max_rounds)):
            for source in current.sources[: self.budget.max_sources_per_task]:
                retriever = self.sources.get(source)
                if retriever is None:
                    continue
                for llq in self.formulator(current, source):
                    found.extend(retriever(llq.query, **dict(llq.params)))
            judged = self.evaluator(current, found[: self.budget.max_results_per_task])
            found = list(judged.relevant)
            if judged.sufficient or judged.refinement is None:
                break
            current = judged.refinement  # the back-edge: re-query
        return found


def make_search_agent(
    sources: Mapping[str, Retriever],
    *,
    planner: Planner | None = None,
    formulator: Formulator | None = None,
    evaluator: Evaluator | None = None,
    reranker: Reranker | None = None,
    citer: Citer | None = None,
    budget: Budget | None = None,
) -> SingleContextAgent:
    """Build a :class:`SingleContextAgent` over *sources* with smart defaults.

    ``sources`` is a ``Mapping[name, Retriever]`` — e.g.
    ``{"skills": ir.as_retriever("skills")}``. Every role defaults to its no-LLM
    thin-slice implementation, so ``make_search_agent(sources)("query")`` just
    works; inject an LLM ``formulator`` / ``evaluator`` to turn on rewriting and
    the back-edge.
    """
    return SingleContextAgent(
        sources=sources,
        planner=planner or single_subtask_planner,
        formulator=formulator or identity_formulator,
        evaluator=evaluator or passthrough_evaluator,
        reranker=reranker or score_reranker,
        citer=citer or identity_citer,
        budget=budget or Budget(),
    )


def ir_sources(*names: str, **search_defaults: Any) -> dict[str, Retriever]:
    """A source registry ``{name: Retriever}`` backed by named ``ir`` corpora.

    Each name is bound to ``ir.as_retriever(name, **search_defaults)``. A thin
    convenience; once ir ships ``registry.retrievers()`` (a lazy view), prefer
    that. ``search_defaults`` (e.g. ``mode="hybrid"``) apply to every source.
    """
    import ir

    return {name: ir.as_retriever(name, **search_defaults) for name in names}
