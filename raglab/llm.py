"""LLM-backed roles — the Formulator and Evaluator (ir_09 §8 steps 1 & 3).

`raglab`'s no-LLM thin slice (:mod:`raglab.agent`) runs offline: an identity
formulator and a pass-through evaluator. This module supplies the two LLM roles
that turn that slice into a real agent — query understanding and the back-edge:

- :func:`make_llm_formulator` — query rewrite / expand / HyDE. A thin adapter
  over ir's :func:`ir.make_llm_formulator` (the ``formulate=`` seam, ir_09 §3):
  ir owns the lazy-:mod:`oa` rewriter and the identity fallback; raglab only
  adapts the shape ``str -> str | [str]`` to the role contract
  ``(SubTask, source) -> [LowLevelQuery]``.
- :func:`make_llm_evaluator` — sufficiency + refinement. ``ir.select`` owns the
  *relevance* decision (the calibrated committed subset, ir_09 §3); the LLM owns
  *sufficiency* — informed by ir's model-free :attr:`ir.Selection.sufficient`
  hint, it judges whether the committed subset actually satisfies the goal and,
  when it does not, emits a ``refinement`` SubTask. That refinement is the
  **back-edge** that makes the loop an agent rather than a DAG.

Two load-bearing boundaries (ir ↔ raglab), guarded here:

1. A Formulator returns **queries, never SubTasks** — decomposition is the
   Planner's job. :func:`make_llm_formulator` only ever yields
   :class:`~raglab.agent.LowLevelQuery`\\ s.
2. The **back-edge lives in raglab**, never in ir: ir derives a ``sufficient``
   *signal* from its own selection; raglab's Evaluator is what reads it, decides,
   and re-queries.

Both builders mirror :func:`ir.select.make_llm_selector`: an injectable callable
(a test double, or your own router), built lazily on :mod:`oa` only when omitted
(so ``import raglab`` stays offline), with a **safe fallback** — a formulator must
never make retrieval worse than the raw query, and an evaluator must never
fabricate an endless loop (on any failure it returns no refinement, which is the
loop's break condition).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import ir

from .agent import (
    Evaluator,
    Formulator,
    Judgement,
    LowLevelQuery,
    Reranker,
    Result,
    SubTask,
    rrf_reranker,
)

__all__ = ["make_llm_formulator", "make_llm_evaluator", "EVALUATION_PROMPT"]

#: An ir-style query formulator: a query string -> one query or several.
QueryFormulator = Callable[[str], "str | Sequence[str]"]

#: An evaluator judge: ``(goal, results) -> (sufficient, refinement)``.
#: ``sufficient`` ends the loop; a non-empty ``refinement`` query becomes the
#: next sub-goal (the back-edge). A raw text reply is also accepted and parsed.
Judge = Callable[..., "tuple[bool, str | None] | str"]

#: Truncate each rendered result's text to this many chars in the judge prompt.
DFLT_MAX_RESULT_CHARS = 500


# --------------------------------------------------------------------------- #
# LLM Formulator — adapt ir's query-level formulator to the role contract
# --------------------------------------------------------------------------- #


def make_llm_formulator(
    *,
    formulate: QueryFormulator | None = None,
    params: Mapping[str, Any] | None = None,
    **make_kwargs: Any,
) -> Formulator:
    """An LLM-backed raglab :class:`~raglab.agent.Formulator`.

    Adapts an ir-style query formulator (``str -> str | [str]``) to the role
    contract ``(SubTask, source) -> [LowLevelQuery]``: the sub-goal text is
    rewritten / expanded into one or more search queries, each wrapped as a
    :class:`~raglab.agent.LowLevelQuery` against ``source``.

    Args:
        formulate: an injectable ir-style formulator (a test double, or one you
            built). When omitted it is built once via
            :func:`ir.make_llm_formulator` (``**make_kwargs`` are forwarded —
            e.g. ``n``, ``prompt``, ``rewriter``); that call is offline (``oa`` is
            imported lazily only when the formulator is *invoked*), so
            ``import raglab`` stays offline.
        params: per-call retriever overrides (e.g. ``{"mode": "hybrid", "k": 5}``)
            attached to every emitted :class:`~raglab.agent.LowLevelQuery`.

    The boundary: this returns **queries, never SubTasks**. ir's formulator
    already falls back to identity on any failure, so the emitted list is never
    empty — a formulator must never make retrieval worse than the raw sub-goal.
    """
    extra = dict(params or {})
    fn: QueryFormulator = (
        formulate if formulate is not None else ir.make_llm_formulator(**make_kwargs)
    )

    def formulator(task: SubTask, source: str) -> list[LowLevelQuery]:
        try:
            out = fn(task.goal)
        except Exception:
            out = None  # a formulator must never make retrieval worse: fall back
        queries = [out] if isinstance(out, str) else list(out or [])
        queries = [q for q in queries if isinstance(q, str) and q.strip()] or [
            task.goal
        ]
        return [
            LowLevelQuery(source=source, query=q, params=dict(extra)) for q in queries
        ]

    return formulator


# --------------------------------------------------------------------------- #
# LLM Evaluator — ir.select owns relevance; the LLM owns sufficiency + the
# back-edge (ir_09 §3/§4)
# --------------------------------------------------------------------------- #

#: Default prompt for :func:`make_llm_evaluator` — judge sufficiency, else emit a
#: single improved query (the refinement that drives the back-edge).
EVALUATION_PROMPT = """\
A search agent is pursuing this goal:

{goal}

A calibrated selector reviewed the retrieved candidates and committed to the
results below (it abstained if this list is empty):

{results}

Decide whether these results are SUFFICIENT to satisfy the goal.
- If they are, reply with exactly: SUFFICIENT
- If they are not, reply with: INSUFFICIENT
  then, on the next line, a single improved search query that would retrieve what
  is still missing. Keep it a terse search phrase — no prose, no numbering.
"""


def make_llm_evaluator(
    *,
    judge: Judge | None = None,
    select_strategy: str = "conservative",
    select_kwargs: Mapping[str, Any] | None = None,
    prerank: Reranker | None = None,
    prompt: str = EVALUATION_PROMPT,
    max_result_chars: int = DFLT_MAX_RESULT_CHARS,
    **prompt_function_kwargs: Any,
) -> Evaluator:
    """An LLM-backed raglab :class:`~raglab.agent.Evaluator` (turns on the back-edge).

    Relevance is ir's: each round the accumulated results are passed through
    :func:`ir.select` and the committed subset becomes the ``Judgement.relevant``
    (so the LLM stays in its lane — LLM relevance is known-fragile, ir_01 §3).
    Sufficiency is the LLM's: it reads the committed subset (informed by ir's
    model-free :attr:`~ir.Selection.sufficient` hint via abstention) and decides
    whether the goal is satisfied. When it is not, the judge's improved query
    becomes a ``refinement`` :class:`~raglab.agent.SubTask` over the same sources
    — the **back-edge**.

    Args:
        judge: an injectable ``(goal, results) -> (sufficient, refinement)``
            callable (a test double, or your own router); a raw text reply is also
            accepted and parsed. When omitted it is built lazily on :mod:`oa`.
        select_strategy: ir selection strategy for the relevance decision
            (default ``"conservative"`` — distractor-robust).
        select_kwargs: extra args forwarded to :func:`ir.select`
            (e.g. ``max_k``, ``rel``, ``min_score``).
        prerank: the per-round merge that puts the accumulated pool best-first
            (and deduped) before ``ir.select``. Defaults to
            :func:`~raglab.agent.rrf_reranker` — a round's pool can already mix
            sources (a SubTask spans several), so the same rank-based,
            scale-safe merge as the fan-in applies; single-source rounds keep
            raw scores. Inject :func:`~raglab.agent.score_reranker` for
            sources known to share one score scale.
        max_result_chars: per-result text truncation in the judge prompt.

    Safe fallback: any judge error returns ``refinement=None`` — the control
    loop's break condition — so a failing judge can never fabricate an endless
    loop. Sufficiency without a refinement query is likewise treated as a stop.
    """
    sel_kw = dict(select_kwargs or {})
    prerank = prerank or rrf_reranker

    def _ask_judge(goal: str, rendered: str) -> tuple[bool, str | None]:
        fn = (
            judge
            if judge is not None
            else _default_llm_judge(prompt, **prompt_function_kwargs)
        )
        return _normalize_verdict(fn(goal=goal, results=rendered))

    def evaluator(task: SubTask, results: Sequence[Result]) -> Judgement:
        # ``ir.select`` documents a best-first precondition; the loop accumulates
        # hits across rounds/sources in arbitrary order, so merge them first —
        # the same (scale-safe) merge the final fan-in reranker uses.
        ranked = prerank(results)
        selection = ir.select(list(ranked), strategy=select_strategy, **sel_kw)
        relevant = list(selection.selected)
        rendered = (
            _render_results(relevant, max_result_chars)
            if relevant
            else "(none — the selector abstained)"
        )
        try:
            sufficient, refinement = _ask_judge(task.goal, rendered)
        except Exception:
            # Trust ir's model-free signal for the report, but never re-query on a
            # judge failure: refinement=None is the loop's break condition.
            return Judgement(
                relevant=relevant, sufficient=selection.sufficient, refinement=None
            )
        if sufficient or not refinement:
            return Judgement(relevant=relevant, sufficient=True, refinement=None)
        return Judgement(
            relevant=relevant,
            sufficient=False,
            refinement=SubTask(goal=refinement, sources=task.sources),
        )

    return evaluator


def _render_results(results: Sequence[Result], max_chars: int) -> str:
    """Render committed hits as ``- id: text`` lines for the judge prompt."""
    return "\n".join(f"- {h.artifact_id}: {str(h.text)[:max_chars]}" for h in results)


def _normalize_verdict(out: "tuple[bool, str | None] | str") -> tuple[bool, str | None]:
    """Coerce a judge reply (a ``(sufficient, refinement)`` tuple, or raw text)."""
    if isinstance(out, tuple):
        sufficient, refinement = out
        refinement = str(refinement).strip() if refinement else None
        return bool(sufficient), (refinement or None)
    return _parse_verdict(str(out or ""))


def _parse_verdict(text: str) -> tuple[bool, str | None]:
    """Parse a SUFFICIENT / INSUFFICIENT + refinement-query text reply."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0].upper().startswith("SUFFICIENT"):
        return True, None
    refinement = lines[1] if len(lines) > 1 else None
    return False, (refinement or None)


def _default_llm_judge(prompt: str, **prompt_function_kwargs: Any) -> Judge:
    """Build the default sufficiency judge on :mod:`oa` (lazy import)."""
    import oa

    fn = oa.prompt_function(
        prompt,
        egress=_parse_verdict,
        name="evaluate_sufficiency",
        **prompt_function_kwargs,
    )

    def judge(*, goal: str, results: str) -> tuple[bool, str | None]:
        return fn(goal=goal, results=results)

    return judge
