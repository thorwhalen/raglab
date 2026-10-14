"""The Composable Search Agent — roles wired by a control loop (ir_09).

`raglab` is the **orchestration layer on top of `ir`** (the retrieval substrate).
This module is the v1 foundation: the immutable value types, the role *Protocols*
(open-closed strategy seams), and a `SingleContextAgent` whose fixed control loop
is fully parametrized by injected roles. Concrete tools live at the leaves — an
`ir` corpus becomes one `Retriever` via :func:`ir.as_retriever`.

The agent is **multi-source by default**: the loop stamps each hit's
provenance (``hit.source``), and the fan-in :class:`Reranker` —
:func:`rrf_reranker` — merges heterogeneous sources by *rank*, never by raw
score (scores from different corpora / embedders / modes are incommensurable;
ir_07/ir_08). Raw magnitudes order and dedup hits *within* one source, and the
loop's pool always carries them; fused (ordinal) scores appear only at the
fan-in boundary.

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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ir owns the retrieval substrate: the Result type, the Retriever leaf
# contract, and the hit operations (dedup, cross-source fusion) live there
# (one-way dependency, ir is the SSOT).
from ir import Retriever, SearchHit, fuse_hits, tag_source
from ir.base import best_per_artifact
from ir.retrieve import DFLT_RRF_K, Identity

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
    "PoolCap",
    "rank_fair_cap",
    "arrival_order_cap",
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


@runtime_checkable
class PoolCap(Protocol):
    """Bound the pool the evaluator sees to at most ``limit`` results.

    A governor decision (which hits survive the budget), injected like every other
    role so a strategy can be swapped without touching the loop. It must return
    hits **unchanged** — never re-scored — because the pool keeps raw per-source
    magnitudes until the fan-in reranker.
    """

    def __call__(self, results: Sequence[Result], limit: int) -> list[Result]: ...


def _rank_key(hit: Result) -> float:
    """A hit's raw score for ordering *within its source*; a missing or NaN score is last."""
    score = hit.score
    if isinstance(score, (int, float)) and not math.isnan(score):
        return float(score)
    return -math.inf


def rank_fair_cap(results: Sequence[Result], limit: int) -> list[Result]:
    """Keep at most ``limit`` hits, taking the best of each source in turn (the default).

    Groups the pool by ``hit.source`` (in first-arrival order; untagged hits are one
    group), collapses each group to one hit per artifact (its best surface, as
    :func:`ir.base.best_per_artifact` does), orders the group best-first by its own raw
    score -- sound, since one source is one score scale -- and deals hits round-robin
    across the groups. So a late source's strong hits are never cut before the evaluator
    sees them, a duplicate never takes a slot, and no score ever crosses a source
    boundary or is rewritten (unlike pre-ranking with the RRF reranker, whose fused
    ordinal scores must never re-enter the pool). The output is in that dealt order
    whatever the pool's size. A missing or NaN score sorts last in its source.
    """
    if limit <= 0:
        return []
    groups: dict[str | None, dict[str, Result]] = {}
    for hit in results:
        best = groups.setdefault(hit.source, {})
        cur = best.get(hit.artifact_id)
        if cur is None or _rank_key(hit) > _rank_key(cur):
            best[hit.artifact_id] = hit
    queues = [
        sorted(best.values(), key=_rank_key, reverse=True) for best in groups.values()
    ]
    capped: list[Result] = []
    for depth in range(max((len(q) for q in queues), default=0)):
        for queue in queues:
            if depth < len(queue):
                capped.append(queue[depth])
                if len(capped) == limit:
                    return capped
    return capped


def arrival_order_cap(results: Sequence[Result], limit: int) -> list[Result]:
    """Keep the first ``limit`` hits in arrival order (source order x query order).

    The pre-governor behaviour: a late source's strong hits can be cut before the
    evaluator sees them. Kept as an explicit opt-in (``pool_cap=arrival_order_cap``).
    """
    return list(results[: max(0, limit)])


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
    """Magnitude merge: one surface per artifact, ordered by descending raw score.

    Delegates to :func:`ir.base.best_per_artifact` (ir is the SSOT for hit
    operations): an artifact retrieved by several queries / rounds — common once
    the back-edge re-queries — survives once, at its highest score. Identity is
    ``(source, artifact_id)``, so two sources' same-id artifacts never collapse.

    A plain score sort compares raw scores **across** sources, which is only
    sound when every source shares one score scale (same embedder + mode) — it
    is the explicit homogeneous-sources opt-in. The default fan-in is
    :func:`rrf_reranker`, which never compares raw scores across sources.
    """
    return best_per_artifact(results)


def _rrf_rerank(
    results: Sequence[Result],
    *,
    rrf_k: int,
    weights: Mapping[str, float] | None,
    identity: Identity,
) -> Sequence[Result]:
    """Group by ``hit.source`` and rank-fuse via :func:`ir.fuse_hits`."""
    # None is preserved as the untagged pseudo-source key: its hits fuse as
    # one rank group and stay unattributed (never an empty-string stamp).
    groups: dict[str | None, list[Result]] = {}
    for h in results:
        groups.setdefault(h.source, []).append(h)
    if len(groups) <= 1:
        # One scale: the magnitude merge, with hits passed through untouched.
        return best_per_artifact(results)
    return fuse_hits(groups, rrf_k=rrf_k, weights=weights, identity=identity)


def rrf_reranker(results: Sequence[Result]) -> Sequence[Result]:
    """Cross-source merge (the default fan-in): fuse by rank, never by raw score.

    Groups the accumulated pool by ``hit.source`` and delegates the merge to
    :func:`ir.fuse_hits` (ir is the SSOT for hit operations): within each
    source raw scores order and dedup that source's hits — one scale, sound —
    and across sources only **ranks** interact (Reciprocal Rank Fusion), so
    heterogeneous embedders / modes can never mis-order the merge, and
    colliding ``artifact_id``\\ s from different sources stay distinct results
    (identity is ``(source, artifact_id)``).

    A single-source pool (or an untagged one — hits with no ``source``) keeps
    its raw scores and exactly :func:`score_reranker`'s ordering; fused,
    rank-derived scores only appear when there is genuinely something to fuse.
    Each fused hit keeps its pre-fusion magnitude as
    ``metadata["source_score"]``. For per-source weights, another ``rrf_k``,
    or opt-in cross-source duplicate merging, use :func:`make_rrf_reranker`.
    """
    return _rrf_rerank(results, rrf_k=DFLT_RRF_K, weights=None, identity=None)


def make_rrf_reranker(
    *,
    rrf_k: int = DFLT_RRF_K,
    weights: Mapping[str, float] | None = None,
    identity: Identity = None,
) -> Reranker:
    """A parametrized :func:`rrf_reranker` (per-source trust weights, ``rrf_k``).

    Args:
        rrf_k: the RRF rank constant (standard default 60).
        weights: optional per-source trust dial, by source name (default 1.0
            each) — biases the merge without ever comparing raw scores.
        identity: opt-in cross-source duplicate detection (e.g. ``"pointer"``)
            — see :data:`ir.retrieve.Identity`. Default: never merge across
            sources.
    """

    def reranker(results: Sequence[Result]) -> Sequence[Result]:
        return _rrf_rerank(results, rrf_k=rrf_k, weights=weights, identity=identity)

    return reranker


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
    reranker: Reranker = rrf_reranker
    citer: Citer = identity_citer
    budget: Budget = field(default_factory=Budget)
    pool_cap: PoolCap = rank_fair_cap

    def __call__(self, query: str | Query) -> list[Result]:
        """Run the agent for *query*; returns the final ranked, cited results."""
        q = query if isinstance(query, Query) else Query(text=query)
        accumulated: list[Result] = []
        for task in self.planner(q, self.sources):
            accumulated.extend(self._run_task(task))
        ranked = self.reranker(accumulated)
        return list(self.citer(ranked))

    def _run_task(self, task: SubTask) -> list[Result]:
        # `kept` is what the evaluator judged relevant last round: governor-approved,
        # carried into the next round untouched. Only a round's *new* hits compete for
        # the rest of the budget (`pool_cap`), so a re-query can never evict a hit the
        # evaluator already kept, nor spend a slot re-offering it.
        kept: list[Result] = []
        current = task
        limit = self.budget.max_results_per_task
        for _ in range(max(1, self.budget.max_rounds)):
            found: list[Result] = []
            for source in current.sources[: self.budget.max_sources_per_task]:
                retriever = self.sources.get(source)
                if retriever is None:
                    continue
                for llq in self.formulator(current, source):
                    # Stamp the registry key on hits the retriever did not
                    # self-attribute (ir-backed retrievers stamp the corpus
                    # name themselves, and their tags win), so any custom
                    # Retriever still yields attributable hits — the fan-in
                    # reranker merges by source.
                    found.extend(
                        tag_source(retriever(llq.query, **dict(llq.params)), source)
                    )
            seen = {(h.source, h.artifact_id) for h in kept}
            fresh = [h for h in found if (h.source, h.artifact_id) not in seen]
            pool = kept + self.pool_cap(fresh, max(0, limit - len(kept)))
            judged = self.evaluator(current, pool)
            kept = list(judged.relevant)
            if judged.sufficient or judged.refinement is None:
                break
            current = judged.refinement  # the back-edge: re-query
        return kept


def make_search_agent(
    sources: Mapping[str, Retriever],
    *,
    planner: Planner | None = None,
    formulator: Formulator | None = None,
    evaluator: Evaluator | None = None,
    reranker: Reranker | None = None,
    citer: Citer | None = None,
    budget: Budget | None = None,
    pool_cap: PoolCap | None = None,
) -> SingleContextAgent:
    """Build a :class:`SingleContextAgent` over *sources* with smart defaults.

    ``sources`` is a ``Mapping[name, Retriever]`` — e.g.
    ``{"skills": ir.as_retriever("skills")}``, :func:`ir_sources`, or the lazy
    ``ir.retrievers()`` view. Every role defaults to its no-LLM thin-slice
    implementation, so ``make_search_agent(sources)("query")`` just works;
    inject an LLM ``formulator`` / ``evaluator`` to turn on rewriting and the
    back-edge.

    A custom ``Retriever`` must return :class:`ir.SearchHit` instances (the
    ``Result`` alias): the loop stamps provenance (``hit.source``) on its
    output, so duck-typed hit objects raise at the tagging step.
    """
    return SingleContextAgent(
        sources=sources,
        planner=planner or single_subtask_planner,
        formulator=formulator or identity_formulator,
        evaluator=evaluator or passthrough_evaluator,
        reranker=reranker or rrf_reranker,
        citer=citer or identity_citer,
        budget=budget or Budget(),
        pool_cap=pool_cap or rank_fair_cap,
    )


def ir_sources(*names: str, **search_defaults: Any) -> dict[str, Retriever]:
    """A source registry ``{name: Retriever}`` backed by named ``ir`` corpora.

    Each name is bound to ``ir.as_retriever(name, **search_defaults)``, opened
    eagerly. For the *lazy* live view over everything registered (a corpus
    opens only when its key is first used), use ``ir.retrievers()`` instead —
    the agent accepts either, or any ``Mapping[name, Retriever]``.
    ``search_defaults`` (e.g. ``mode="hybrid"``) apply to every source.
    """
    import ir

    return {name: ir.as_retriever(name, **search_defaults) for name in names}
