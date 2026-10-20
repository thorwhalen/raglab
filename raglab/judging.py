"""Graded rubric judging — every candidate gets a level, three axes and evidence (raglab#8).

A binary relevant/not cannot say "mentions the word but is not about it". The
evaluator built by :func:`make_graded_rubric_evaluator` asks an injected judge,
for **each** candidate in the round's pool, for the three Self-RAG axes
(``is_about``, ``is_supported``, ``is_useful``), a level on an ordered scale and
a verbatim ``evidence_quote`` from the candidate's disclosed content. The judge
proposes; deterministic gates here dispose:

- a level above ``unsupported_cap`` needs ``is_supported`` — topical vocabulary
  without supporting content is capped (the direct fix for on-word distractors);
- with ``require_evidence_quote``, it also needs a quote found verbatim in the
  content the judge was shown (whitespace-insensitive);
- ``calibration_offset`` (levels, learned from a gold slice) shifts every level
  down (positive: the judge over-grades) or up;
- the level options are shown in a fresh random order on every call, against
  position bias.

The result is a :class:`~raglab.agent.Judgement` whose additive ``graded`` map
carries every label, whose ``relevant`` keeps the candidates at or above
``relevant_level``, and whose back-edge comes from an injected ``refine`` policy
(none by default: the loop stops). As in :mod:`raglab.llm`, the default judge is
built lazily on :mod:`oa`, so ``import raglab`` stays offline, and a judge that
fails on a candidate leaves it ungraded rather than guessing.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ir
from ir.base import best_per_artifact

from .agent import Evaluator, GradedLabel, Judgement, Result, SubTask

__all__ = [
    "DEFAULT_LEVELS",
    "DEFAULT_TOPICAL_RUBRIC",
    "GRADING_PROMPT",
    "make_graded_rubric_evaluator",
    "apply_gates",
]

#: Ordered levels, best first.
DEFAULT_LEVELS: tuple[str, ...] = ("core", "strong", "uses-tools", "tangential", "none")

#: What the default levels mean, for a "is this package about TOPIC" question.
DEFAULT_TOPICAL_RUBRIC = """\
- core: the candidate's main purpose IS the topic.
- strong: a substantial, central part of the candidate is about the topic.
- uses-tools: the candidate builds on tools or libraries of the topic (its dependencies or imports establish it) without being about it.
- tangential: the topic is mentioned, or its vocabulary appears, but the content does not establish it.
- none: unrelated.
"""

#: The judge's prompt; ``{levels}`` is shown in a fresh random order each call.
GRADING_PROMPT = """\
Goal: {goal}

Rubric (levels, in no particular order):
{levels}

Definitions:
{rubric}

Candidate {artifact_id}:
---
{content}
---

Answer with one JSON object and nothing else, with keys:
"is_about" (bool: is the candidate topically about the goal?),
"is_supported" (bool: does its content - code, dependencies, imports, full text - actually establish it, beyond a mention of the words?),
"is_useful" (bool: would it help someone pursuing the goal?),
"evidence_quote" (a short span copied VERBATIM from the candidate's content that supports your level),
"level" (one of the rubric's levels).
"""

#: A graded judge: keyword arguments ``goal, artifact_id, content, levels, rubric``
#: -> a mapping with the label's keys, or the raw JSON text of one.
GradedJudge = Callable[..., "Mapping[str, Any] | str"]

#: A refinement policy: ``(task, graded) -> the next SubTask, or None to stop``.
Refine = Callable[[SubTask, Mapping[str, GradedLabel]], "SubTask | None"]

#: Truncate each candidate's disclosed content to this many chars in the prompt.
DFLT_MAX_CONTENT_CHARS = 4000


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def apply_gates(
    proposed: GradedLabel,
    content: str,
    *,
    levels: Sequence[str] = DEFAULT_LEVELS,
    unsupported_cap: str = "tangential",
    require_evidence_quote: bool = True,
    calibration_offset: float | None = None,
) -> GradedLabel:
    """The level a judge's proposal is allowed, after calibration and the gates.

    >>> label = GradedLabel('core', True, False, True, 'a DAG of tasks')
    >>> apply_gates(label, 'Runs a DAG of tasks asynchronously.').level
    'tangential'
    >>> ok = GradedLabel('core', True, True, True, 'graph  library')
    >>> apply_gates(ok, 'A graph library.').level, apply_gates(ok, 'Nothing here.').level
    ('core', 'tangential')
    >>> apply_gates(ok, 'A graph library.', calibration_offset=1.4).level
    'strong'
    """
    cap = levels.index(unsupported_cap)
    rank = levels.index(proposed.level)
    if calibration_offset:
        rank = min(max(round(rank + calibration_offset), 0), len(levels) - 1)
    quote = _normalize_space(proposed.evidence_quote)
    quoted = bool(quote) and quote in _normalize_space(content)
    if rank < cap and (
        not proposed.is_supported or (require_evidence_quote and not quoted)
    ):
        rank = cap
    return GradedLabel(
        level=levels[rank],
        is_about=proposed.is_about,
        is_supported=proposed.is_supported,
        is_useful=proposed.is_useful,
        evidence_quote=proposed.evidence_quote if quoted else "",
        facet=proposed.facet,
    )


def _parse_label(out: Mapping[str, Any] | str, levels: Sequence[str]) -> GradedLabel:
    """A judge's reply as a :class:`GradedLabel`; raises ``ValueError`` when unusable."""
    if isinstance(out, str):
        match = re.search(r"\{.*\}", out, flags=re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in the judge's reply: {out[:200]!r}")
        out = json.loads(match.group(0))
    level = str(out.get("level", "")).strip().lower()
    if level not in levels:
        raise ValueError(f"level {level!r} is not one of {list(levels)}")

    def flag(key: str) -> bool:
        value = out.get(key)
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes")
        return bool(value)

    return GradedLabel(
        level=level,
        is_about=flag("is_about"),
        is_supported=flag("is_supported"),
        is_useful=flag("is_useful"),
        evidence_quote=str(out.get("evidence_quote") or ""),
    )


def _contents(
    hits: Sequence[Result], level: str, disclose_kwargs: Mapping[str, Any]
) -> list[str]:
    """Each hit's disclosed content at ``level``, its snippet when nothing loads."""
    selection = ir.Selection(
        selected=list(hits), candidates=list(hits), abstained=False, reason="graded"
    )
    disclosed = ir.disclose(selection, level=level, **dict(disclose_kwargs))
    return [d.body or h.text for h, d in zip(hits, disclosed)]


def make_graded_rubric_evaluator(
    *,
    levels: Sequence[str] = DEFAULT_LEVELS,
    rubric: str = DEFAULT_TOPICAL_RUBRIC,
    judge: GradedJudge | None = None,
    disclose_level: str = "body",
    disclose_kwargs: Mapping[str, Any] | None = None,
    require_evidence_quote: bool = True,
    unsupported_cap: str = "tangential",
    calibration_offset: float | None = None,
    relevant_level: str = "uses-tools",
    refine: Refine | None = None,
    rng: random.Random | None = None,
    prompt: str = GRADING_PROMPT,
    max_content_chars: int = DFLT_MAX_CONTENT_CHARS,
    **prompt_function_kwargs: Any,
) -> Evaluator:
    """An :class:`~raglab.agent.Evaluator` that grades every candidate on a rubric.

    Args:
        levels: the ordered levels, best first.
        rubric: what each level means, shown to the judge.
        judge: an injectable graded judge (a test double, or your own router);
            built lazily on :mod:`oa` when omitted.
        disclose_level, disclose_kwargs: what the judge reads — each candidate's
            content through :func:`ir.disclose` at that level (``store=`` /
            ``loader=`` go in ``disclose_kwargs``); a candidate with nothing to
            load is judged on its snippet.
        require_evidence_quote, unsupported_cap, calibration_offset: the gates,
            see :func:`apply_gates`.
        relevant_level: the lowest level kept in ``Judgement.relevant``.
        refine: the back-edge policy ``(task, graded) -> SubTask | None``; by
            default none, so the loop stops after one round.
        rng: the source of the per-call level order (seed it for reproducible
            prompts).

    Every candidate of the round's pool is judged once per round (one judge call
    each, deduped per artifact): this evaluator is for labelling a pool, not for
    cheap filtering.
    """
    levels = tuple(levels)
    for name, value in (("unsupported_cap", unsupported_cap), ("relevant_level", relevant_level)):
        if value not in levels:
            raise ValueError(f"{name}={value!r} is not one of {list(levels)}")
    keep = levels.index(relevant_level)
    rng = rng if rng is not None else random.Random()
    disclose_kwargs = dict(disclose_kwargs or {})
    state: dict[str, GradedJudge] = {}

    def ask(**kwargs: Any) -> Mapping[str, Any] | str:
        if judge is not None:
            return judge(**kwargs)
        if "judge" not in state:
            state["judge"] = _default_graded_judge(prompt, **prompt_function_kwargs)
        return state["judge"](**kwargs)

    def evaluator(task: SubTask, results: Sequence[Result]) -> Judgement:
        hits = best_per_artifact(results)
        contents = _contents(hits, disclose_level, disclose_kwargs)
        graded: dict[str, GradedLabel] = {}
        for hit, content in zip(hits, contents):
            content = str(content)[:max_content_chars]
            shown = list(levels)
            rng.shuffle(shown)
            try:
                proposed = _parse_label(
                    ask(
                        goal=task.goal,
                        artifact_id=hit.artifact_id,
                        content=content,
                        levels="\n".join(f"- {lv}" for lv in shown),
                        rubric=rubric,
                    ),
                    levels,
                )
            except Exception:
                continue  # left ungraded: never guess a level
            graded[hit.artifact_id] = apply_gates(
                proposed,
                content,
                levels=levels,
                unsupported_cap=unsupported_cap,
                require_evidence_quote=require_evidence_quote,
                calibration_offset=calibration_offset,
            )
        relevant = [
            h
            for h in hits
            if h.artifact_id in graded and levels.index(graded[h.artifact_id].level) <= keep
        ]
        refinement = refine(task, graded) if refine is not None else None
        return Judgement(
            relevant=relevant,
            sufficient=refinement is None,
            refinement=refinement,
            graded=graded,
        )

    return evaluator


def _default_graded_judge(prompt: str, **prompt_function_kwargs: Any) -> GradedJudge:
    """Build the default graded judge on :mod:`oa` (lazy import)."""
    import oa

    return oa.prompt_function(prompt, name="grade_candidate", **prompt_function_kwargs)
