# raglab.llm

LLM-backed roles — the Formulator and Evaluator (ir_09 §8 steps 1 & 3).

`raglab`’s no-LLM thin slice ([`raglab.agent`](raglab.agent.html.md#module-raglab.agent)) runs offline: an identity
formulator and a pass-through evaluator. This module supplies the two LLM roles
that turn that slice into a real agent — query understanding and the back-edge:

- [`make_llm_formulator()`](#raglab.llm.make_llm_formulator) — query rewrite / expand / HyDE. A thin adapter
  over ir’s `ir.make_llm_formulator()` (the `formulate=` seam, ir_09 §3):
  ir owns the lazy-`oa` rewriter and the identity fallback; raglab only
  adapts the shape `str -> str | [str]` to the role contract
  `(SubTask, source) -> [LowLevelQuery]`.
- [`make_llm_evaluator()`](#raglab.llm.make_llm_evaluator) — sufficiency + refinement. `ir.select` owns the
  *relevance* decision (the calibrated committed subset, ir_09 §3); the LLM owns
  *sufficiency* — informed by ir’s model-free `ir.Selection.sufficient`
  hint, it judges whether the committed subset actually satisfies the goal and,
  when it does not, emits a `refinement` SubTask. That refinement is the
  **back-edge** that makes the loop an agent rather than a DAG.

Two load-bearing boundaries (ir ↔ raglab), guarded here:

1. A Formulator returns **queries, never SubTasks** — decomposition is the
   Planner’s job. [`make_llm_formulator()`](#raglab.llm.make_llm_formulator) only ever yields
   [`LowLevelQuery`](raglab.agent.html.md#raglab.agent.LowLevelQuery)s.
2. The **back-edge lives in raglab**, never in ir: ir derives a `sufficient`
   *signal* from its own selection; raglab’s Evaluator is what reads it, decides,
   and re-queries.

Both builders mirror `ir.select.make_llm_selector()`: an injectable callable
(a test double, or your own router), built lazily on `oa` only when omitted
(so `import raglab` stays offline), with a **safe fallback** — a formulator must
never make retrieval worse than the raw query, and an evaluator must never
fabricate an endless loop (on any failure it returns no refinement, which is the
loop’s break condition).

### Module Attributes

| [`EVALUATION_PROMPT`](#raglab.llm.EVALUATION_PROMPT)   | Default prompt for [`make_llm_evaluator()`](#raglab.llm.make_llm_evaluator) — judge sufficiency, else emit a single improved query (the refinement that drives the back-edge).   |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### Functions

| [`make_llm_formulator`](#raglab.llm.make_llm_formulator)(\*[, formulate, params])   | An LLM-backed raglab [`Formulator`](raglab.agent.html.md#raglab.agent.Formulator).                        |
|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [`make_llm_evaluator`](#raglab.llm.make_llm_evaluator)(\*[, judge, ...])           | An LLM-backed raglab [`Evaluator`](raglab.agent.html.md#raglab.agent.Evaluator) (turns on the back-edge). |

### raglab.llm.EVALUATION_PROMPT *= 'A search agent is pursuing this goal:\\n\\n{goal}\\n\\nA calibrated selector reviewed the retrieved candidates and committed to the\\nresults below (it abstained if this list is empty):\\n\\n{results}\\n\\nDecide whether these results are SUFFICIENT to satisfy the goal.\\n- If they are, reply with exactly: SUFFICIENT\\n- If they are not, reply with: INSUFFICIENT\\n  then, on the next line, a single improved search query that would retrieve what\\n  is still missing. Keep it a terse search phrase — no prose, no numbering.\\n'*

Default prompt for [`make_llm_evaluator()`](#raglab.llm.make_llm_evaluator) — judge sufficiency, else emit a
single improved query (the refinement that drives the back-edge).

### raglab.llm.make_llm_evaluator(, judge=None, select_strategy='conservative', select_kwargs=None, prerank=None, prompt='A search agent is pursuing this goal:\\n\\n{goal}\\n\\nA calibrated selector reviewed the retrieved candidates and committed to the\\nresults below (it abstained if this list is empty):\\n\\n{results}\\n\\nDecide whether these results are SUFFICIENT to satisfy the goal.\\n- If they are, reply with exactly: SUFFICIENT\\n- If they are not, reply with: INSUFFICIENT\\n  then, on the next line, a single improved search query that would retrieve what\\n  is still missing. Keep it a terse search phrase — no prose, no numbering.\\n', max_result_chars=500, \*\*prompt_function_kwargs)

An LLM-backed raglab [`Evaluator`](raglab.agent.html.md#raglab.agent.Evaluator) (turns on the back-edge).

Relevance is ir’s: each round the accumulated results are passed through
`ir.select()` and the committed subset becomes the `Judgement.relevant`
(so the LLM stays in its lane — LLM relevance is known-fragile, ir_01 §3).
Sufficiency is the LLM’s: it reads the committed subset (informed by ir’s
model-free `sufficient` hint via abstention) and decides
whether the goal is satisfied. When it is not, the judge’s improved query
becomes a `refinement` [`SubTask`](raglab.agent.html.md#raglab.agent.SubTask) over the same sources
— the **back-edge**.

* **Parameters:**
  * **judge** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`bool`](https://docs.python.org/3/builtins/functions.html#bool), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)] | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – an injectable `(goal, results) -> (sufficient, refinement)`
    callable (a test double, or your own router); a raw text reply is also
    accepted and parsed. When omitted it is built lazily on `oa`.
  * **select_strategy** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – ir selection strategy for the relevance decision
    (default `"conservative"` — distractor-robust).
  * **select_kwargs** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – extra args forwarded to `ir.select()`
    (e.g. `max_k`, `rel`). `min_score` is allowed only while the
    round’s pool keeps raw scores (single-source, or
    `prerank=score_reranker`): an absolute floor is per-(corpus,
    mode, embedder), so a round whose pool was rank-fused across
    sources **raises** rather than silently mass-abstaining on
    ordinal RRF scores.
  * **prerank** ([`Reranker`](raglab.agent.html.md#raglab.agent.Reranker) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the per-round merge that puts the accumulated pool best-first
    (and deduped) before `ir.select`. Defaults to
    [`rrf_reranker()`](raglab.agent.html.md#raglab.agent.rrf_reranker) — a round’s pool can already mix
    sources (a SubTask spans several), so the same rank-based,
    scale-safe merge as the fan-in applies; single-source rounds keep
    raw scores. Inject [`score_reranker()`](raglab.agent.html.md#raglab.agent.score_reranker) for
    sources known to share one score scale.
  * **max_result_chars** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – per-result text truncation in the judge prompt.
* **Return type:**
  [`Evaluator`](raglab.agent.html.md#raglab.agent.Evaluator)

Safe fallback: any judge error returns `refinement=None` — the control
loop’s break condition — so a failing judge can never fabricate an endless
loop. Sufficiency without a refinement query is likewise treated as a stop.

### raglab.llm.make_llm_formulator(, formulate=None, params=None, \*\*make_kwargs)

An LLM-backed raglab [`Formulator`](raglab.agent.html.md#raglab.agent.Formulator).

Adapts an ir-style query formulator (`str -> str | [str]`) to the role
contract `(SubTask, source) -> [LowLevelQuery]`: the sub-goal text is
rewritten / expanded into one or more search queries, each wrapped as a
[`LowLevelQuery`](raglab.agent.html.md#raglab.agent.LowLevelQuery) against `source`.

* **Parameters:**
  * **formulate** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – an injectable ir-style formulator (a test double, or one you
    built). When omitted it is built once via
    `ir.make_llm_formulator()` (`**make_kwargs` are forwarded —
    e.g. `n`, `prompt`, `rewriter`); that call is offline (`oa` is
    imported lazily only when the formulator is *invoked*), so
    `import raglab` stays offline.
  * **params** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – per-call retriever overrides (e.g. `{"mode": "hybrid", "k": 5}`)
    attached to every emitted [`LowLevelQuery`](raglab.agent.html.md#raglab.agent.LowLevelQuery).
* **Return type:**
  [`Formulator`](raglab.agent.html.md#raglab.agent.Formulator)

The boundary: this returns **queries, never SubTasks**. ir’s formulator
already falls back to identity on any failure, so the emitted list is never
empty — a formulator must never make retrieval worse than the raw sub-goal.
