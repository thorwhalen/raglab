"""Label a corpus against a topic, facet by facet, with a graded judge (raglab#8, part B).

:func:`label_corpus` composes existing pieces and adds no orchestrator:

- a planner that emits one :class:`~raglab.agent.SubTask` per **facet** of the
  topic, for broad recall;
- a formulator (identity by default; :func:`raglab.make_llm_formulator` for
  query expansion);
- the graded rubric evaluator (:func:`raglab.make_graded_rubric_evaluator`),
  which judges **every** retrieved candidate on its disclosed content;
- a **coverage** check per facet (enough candidates at ``coverage_level``),
  which, when a facet looks thin and a ``reformulate`` policy is given, emits a
  refinement sub-task: the back-edge of :class:`~raglab.agent.SingleContextAgent`.

The deliverable is a frozen :class:`CorpusLabeling`: each judged artifact's best
:class:`~raglab.agent.GradedLabel` (its ``facet`` set) and each facet's coverage.
Nothing here touches ``ir``; the corpus is read through a retriever
(:func:`ir.as_retriever` by default).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import ir

from ..agent import (
    Budget,
    Formulator,
    GradedLabel,
    Judgement,
    Query,
    Result,
    Retriever,
    SingleContextAgent,
    SubTask,
    identity_formulator,
)
from ..judging import DEFAULT_LEVELS, make_graded_rubric_evaluator

__all__ = ["CorpusLabeling", "FacetCoverage", "label_corpus"]

#: A gap-driven reformulation: ``(facet goal, labels so far) -> a new query, or None``.
Reformulate = Callable[[str, Mapping[str, GradedLabel]], "str | None"]

#: The registry key the corpus's retriever is registered under.
SOURCE = "corpus"

#: Candidates retrieved per query by the default retriever: labelling wants recall.
DFLT_K = 50


@dataclass(frozen=True)
class FacetCoverage:
    """How well one facet was covered: candidates judged, those at the coverage level, rounds run."""

    facet: str
    judged: int
    covered: int
    rounds: int
    sufficient: bool


@dataclass(frozen=True)
class CorpusLabeling:
    """The graded labels of a corpus for one topic, and each facet's coverage."""

    labels: Mapping[str, GradedLabel] = field(default_factory=dict)
    coverage: Mapping[str, FacetCoverage] = field(default_factory=dict)
    levels: tuple[str, ...] = DEFAULT_LEVELS

    def at_least(self, level: str) -> list[str]:
        """The artifact ids labelled ``level`` or better, on the levels they were labelled on."""
        cut = self.levels.index(level)
        return [aid for aid, g in self.labels.items() if self.levels.index(g.level) <= cut]


def _facet_goals(facets: Mapping[str, str] | Iterable[str]) -> dict[str, str]:
    goals = dict(facets) if isinstance(facets, Mapping) else {f: f for f in facets}
    if not goals:
        raise ValueError("label_corpus needs at least one facet")
    if len(set(goals.values())) < len(goals):
        raise ValueError(f"two facets share a goal: {goals}; give each facet its own goal")
    return goals


def label_corpus(
    corpus: Any,
    *,
    facets: Mapping[str, str] | Iterable[str],
    retriever: Retriever | None = None,
    formulator: Formulator = identity_formulator,
    judge: Callable[..., Any] | None = None,
    reformulate: Reformulate | None = None,
    levels: tuple[str, ...] = DEFAULT_LEVELS,
    coverage_level: str = "strong",
    min_covered: int = 1,
    budget: Budget | None = None,
    k: int = DFLT_K,
    **evaluator_kwargs: Any,
) -> CorpusLabeling:
    """Label ``corpus`` for each of ``facets`` with a graded judge; see the module.

    Args:
        corpus: an ``ir`` corpus or corpus name, read through
            ``ir.as_retriever(corpus, k=k)`` unless ``retriever`` is given.
        facets: the topic's facets, as ``{name: goal}`` or names (a name is its
            own goal). One sub-task each.
        retriever: inject a :data:`~raglab.agent.Retriever` instead (a test
            double, a federated source); ``corpus`` is then not read.
        formulator: the Formulator role (identity, or
            :func:`raglab.make_llm_formulator`).
        judge: the graded judge (built lazily on :mod:`oa` when omitted).
        reformulate: gap-driven reformulation; when a facet has fewer than
            ``min_covered`` labels at ``coverage_level`` it gets a new query, which
            becomes the facet's next sub-task. ``None`` (default): one round each.
        budget: the loop bounds (``max_rounds`` caps the reformulations).
        **evaluator_kwargs: forwarded to
            :func:`~raglab.judging.make_graded_rubric_evaluator` (``rubric``,
            ``calibration_offset``, ``disclose_kwargs``, ``rng``, ...).
    """
    goals = _facet_goals(facets)
    levels = tuple(levels)
    named = {"coverage_level": coverage_level, **{
        key: evaluator_kwargs.get(key, default)
        for key, default in (("unsupported_cap", "tangential"), ("relevant_level", "uses-tools"))
    }}
    for key, value in named.items():
        if value not in levels:
            raise ValueError(
                f"{key}={value!r} is not one of {list(levels)}: with custom levels, "
                "pass coverage_level, unsupported_cap and relevant_level from them"
            )
    cut = levels.index(coverage_level)
    source = retriever if retriever is not None else ir.as_retriever(corpus, k=k)
    facet_of: dict[str, str] = {goal: name for name, goal in goals.items()}
    seen: dict[str, dict[str, GradedLabel]] = {name: {} for name in goals}
    rounds: dict[str, int] = dict.fromkeys(goals, 0)

    def covered(name: str) -> int:
        return sum(levels.index(g.level) <= cut for g in seen[name].values())

    max_rounds = (budget if budget is not None else Budget()).max_rounds

    def refine(task: SubTask, name: str) -> SubTask | None:
        if reformulate is None or covered(name) >= min_covered or rounds[name] >= max_rounds:
            return None  # covered, no policy, or no round left to run a new query in
        query = reformulate(goals[name], dict(seen[name]))
        if not query or query in facet_of:
            return None  # nothing new to ask: stop rather than loop
        facet_of[query] = name
        return SubTask(goal=query, sources=task.sources)

    graded_evaluator = make_graded_rubric_evaluator(
        judge=judge, levels=levels, **evaluator_kwargs
    )

    def evaluator(task: SubTask, results: list[Result]) -> Judgement:
        name = facet_of[task.goal]
        rounds[name] += 1
        judged = graded_evaluator(_as_facet_goal(task, goals[name]), results)
        for aid, label in judged.graded.items():
            label = replace(label, facet=name)
            best = seen[name].get(aid)
            if best is None or levels.index(label.level) < levels.index(best.level):
                seen[name][aid] = label
        refinement = refine(task, name)
        return replace(judged, sufficient=refinement is None, refinement=refinement)

    def _as_facet_goal(task: SubTask, goal: str) -> SubTask:
        # The judge grades against the facet's goal, whatever query retrieved the pool.
        return task if task.goal == goal else replace(task, goal=goal)

    def planner(query: Query, sources: Mapping[str, Retriever]) -> list[SubTask]:
        return [SubTask(goal=goal, sources=(SOURCE,)) for goal in goals.values()]

    agent = SingleContextAgent(
        sources={SOURCE: source},
        planner=planner,
        formulator=formulator,
        evaluator=evaluator,
        budget=budget if budget is not None else Budget(),
    )
    agent(Query(text=" / ".join(goals.values())))

    labels: dict[str, GradedLabel] = {}
    for per_facet in seen.values():
        for aid, label in per_facet.items():
            if aid not in labels or levels.index(label.level) < levels.index(labels[aid].level):
                labels[aid] = label
    coverage = {
        name: FacetCoverage(
            facet=name,
            judged=len(seen[name]),
            covered=covered(name),
            rounds=rounds[name],
            sufficient=covered(name) >= min_covered,
        )
        for name in goals
    }
    return CorpusLabeling(labels=labels, coverage=coverage, levels=levels)
