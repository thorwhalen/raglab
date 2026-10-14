# raglab

`raglab` — the agentic-search / RAG orchestration layer on top of `ir`.

`raglab` turns a retrieval substrate into a *Composable Search Agent* (the ir_09
architecture): a small set of injected **roles** — Planner, Formulator,
Retriever, Evaluator, Reranker, Citer — wired by a control loop whose back-edge
(evaluator → reformulate) is what makes it an agent rather than a DAG. Concrete
tools live at the leaves: an `ir` corpus becomes one `Retriever` via
`ir.as_retriever`.

Quick start (the no-LLM thin slice — runs offline):

```default
import ir
import raglab

# register ir corpora as the agent's sources, then search across them:
sources = raglab.ir_sources("skills", "reports", mode="hybrid")
agent = raglab.make_search_agent(sources)
results = agent("how do I deploy the app")      # ranked ir.SearchHits
```

Inject an LLM `formulator` (query rewrite/HyDE) and `evaluator` (sufficiency

+ refinement) to turn on query understanding and the back-edge. Dependency
  direction is one-way: `raglab` imports `ir`; `ir` never imports `raglab`.

> Fresh start (v0.2.0+). This repo took over the `raglab` PyPI name; the older
> backend now lives at `raglab_bak`. Development is just beginning.

### Functions

| [`rank_fair_cap`](#raglab.rank_fair_cap)(results, limit)                     | Keep at most `limit` hits, taking the best of each source in turn (the default).                                                 |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [`arrival_order_cap`](#raglab.arrival_order_cap)(results, limit)                 | Keep the first `limit` hits in arrival order (source order x query order).                                                       |
| [`make_search_agent`](#raglab.make_search_agent)(sources, \*[, planner, ...])    | Build a [`SingleContextAgent`](#raglab.SingleContextAgent) over *sources* with smart defaults.                  |
| [`ir_sources`](#raglab.ir_sources)(\*names, \*\*search_defaults)          | A source registry `{name: Retriever}` backed by named `ir` corpora.                                                              |
| [`single_subtask_planner`](#raglab.single_subtask_planner)(query, sources)            | Trivial planner: one sub-task over *all* registered sources, no decomposition.                                                   |
| [`identity_formulator`](#raglab.identity_formulator)(task, source)                 | Identity formulator: the sub-goal verbatim as one query (no rewrite/HyDE).                                                       |
| [`passthrough_evaluator`](#raglab.passthrough_evaluator)(task, results)              | Pass-through critic: keep everything, declare sufficient, never re-query.                                                        |
| [`score_reranker`](#raglab.score_reranker)(results)                           | Magnitude merge: one surface per artifact, ordered by descending raw score.                                                      |
| [`rrf_reranker`](#raglab.rrf_reranker)(results)                             | Cross-source merge (the default fan-in): fuse by rank, never by raw score.                                                       |
| [`make_rrf_reranker`](#raglab.make_rrf_reranker)(\*[, rrf_k, weights, identity]) | A parametrized [`rrf_reranker()`](#raglab.rrf_reranker) (per-source trust weights, `rrf_k`).              |
| [`identity_citer`](#raglab.identity_citer)(results)                           | No-op citer (verification needs a generated claim — that lives in srag).                                                         |
| [`make_llm_formulator`](#raglab.make_llm_formulator)(\*[, formulate, params])      | An LLM-backed raglab [`Formulator`](raglab.agent.html.md#raglab.agent.Formulator).                        |
| [`make_llm_evaluator`](#raglab.make_llm_evaluator)(\*[, judge, ...])              | An LLM-backed raglab [`Evaluator`](raglab.agent.html.md#raglab.agent.Evaluator) (turns on the back-edge). |

### Classes

| [`Query`](#raglab.Query)(text[, constraints])                      | A user intent: free text plus optional structured constraints.             |
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`SubTask`](#raglab.SubTask)(goal, sources)                          | A planner's unit of work: a sub-goal bound to a set of registered sources. |
| [`LowLevelQuery`](#raglab.LowLevelQuery)(source, query[, params])          | One concrete query against one source.                                     |
| [`Judgement`](#raglab.Judgement)(relevant, sufficient[, refinement])   | An evaluator's verdict over a round's results.                             |
| [`Result`](#raglab.Result)                                          |                                                                            |
| [`Planner`](#raglab.Planner)(\*args, \*\*kwargs)                     | Decompose a query into sub-tasks and select sources for each.              |
| [`Formulator`](#raglab.Formulator)(\*args, \*\*kwargs)                  | Turn a sub-task + one source into concrete low-level queries.              |
| [`Evaluator`](#raglab.Evaluator)(\*args, \*\*kwargs)                   | Judge relevance + sufficiency; optionally emit a refinement (back-edge).   |
| [`Reranker`](#raglab.Reranker)(\*args, \*\*kwargs)                    | Produce the final ordering over the (cross-source) merged results.         |
| [`Citer`](#raglab.Citer)(\*args, \*\*kwargs)                       | Confirm/annotate that each result supports its use (identity by default).  |
| [`Budget`](#raglab.Budget)([max_rounds, max_sources_per_task, ...]) | Loop bounds: the safety net under the (harder) sufficiency decision.       |
| [`PoolCap`](#raglab.PoolCap)(\*args, \*\*kwargs)                     | Bound the pool the evaluator sees to at most `limit` results.              |
| [`SingleContextAgent`](#raglab.SingleContextAgent)(sources[, planner, ...])     | One ReAct-style loop, sequential sub-tasks (ir_09 §7 — the cheap default). |

### *class* raglab.Budget(max_rounds=3, max_sources_per_task=4, max_results_per_task=50)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Loop bounds: the safety net under the (harder) sufficiency decision.

### *class* raglab.Citer(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Confirm/annotate that each result supports its use (identity by default).

### *class* raglab.Evaluator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Judge relevance + sufficiency; optionally emit a refinement (back-edge).

### *class* raglab.Formulator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Turn a sub-task + one source into concrete low-level queries.

### *class* raglab.Judgement(relevant, sufficient, refinement=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An evaluator’s verdict over a round’s results.

`relevant` is the kept subset; `sufficient` says whether to stop; a
non-`None` `refinement` is the **back-edge** — the next sub-task to
re-query with. The pass-through default returns `sufficient=True` and no
refinement (no loop).

### *class* raglab.LowLevelQuery(source, query, params=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One concrete query against one source.

`query` is the text handed to the source’s `Retriever`; `params`
are per-call retriever overrides (e.g. `mode` / `filter` / `k` for an
ir corpus). A formulator turns a [`SubTask`](#raglab.SubTask) into these.

### *class* raglab.Planner(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Decompose a query into sub-tasks and select sources for each.

### *class* raglab.PoolCap(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Bound the pool the evaluator sees to at most `limit` results.

A governor decision (which hits survive the budget), injected like every other
role so a strategy can be swapped without touching the loop. It must return
hits **unchanged** — never re-scored — because the pool keeps raw per-source
magnitudes until the fan-in reranker.

### *class* raglab.Query(text, constraints=<factory>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A user intent: free text plus optional structured constraints.

### *class* raglab.Reranker(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Produce the final ordering over the (cross-source) merged results.

### raglab.Result

alias of `SearchHit`

### *class* raglab.SingleContextAgent(sources, planner=<function single_subtask_planner>, formulator=<function identity_formulator>, evaluator=<function passthrough_evaluator>, reranker=<function rrf_reranker>, citer=<function identity_citer>, budget=<factory>, pool_cap=<function rank_fair_cap>)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One ReAct-style loop, sequential sub-tasks (ir_09 §7 — the cheap default).

The loop is fixed; every *decision* is an injected role. Promotion to a
multi-agent orchestrator (ir_09 §7) swaps this class while keeping the same
role contracts. The **back-edge** is the single line `current =
judged.refinement` in `_run_task()` — that is what makes this an agent.

#### citer()

No-op citer (verification needs a generated claim — that lives in srag).

* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`SearchHit`]

#### evaluator(results)

Pass-through critic: keep everything, declare sufficient, never re-query.

* **Return type:**
  [`Judgement`](raglab.agent.html.md#raglab.agent.Judgement)

#### formulator(source)

Identity formulator: the sub-goal verbatim as one query (no rewrite/HyDE).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`LowLevelQuery`](raglab.agent.html.md#raglab.agent.LowLevelQuery)]

#### planner(sources)

Trivial planner: one sub-task over *all* registered sources, no decomposition.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SubTask`](raglab.agent.html.md#raglab.agent.SubTask)]

#### pool_cap(limit)

Keep at most `limit` hits, taking the best of each source in turn (the default).

Groups the pool by `hit.source` (in first-arrival order; untagged hits are one
group), collapses each group to one hit per artifact (its best surface, as
`ir.base.best_per_artifact()` does), orders the group best-first by its own raw
score – sound, since one source is one score scale – and deals hits round-robin
across the groups. So a late source’s strong hits are never cut before the evaluator
sees them, a duplicate never takes a slot, and no score ever crosses a source
boundary or is rewritten (unlike pre-ranking with the RRF reranker, whose fused
ordinal scores must never re-enter the pool). The output is in that dealt order
whatever the pool’s size. A missing or NaN score sorts last in its source.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`SearchHit`]

#### reranker()

Cross-source merge (the default fan-in): fuse by rank, never by raw score.

Groups the accumulated pool by `hit.source` and delegates the merge to
`ir.fuse_hits()` (ir is the SSOT for hit operations): within each
source raw scores order and dedup that source’s hits — one scale, sound —
and across sources only **ranks** interact (Reciprocal Rank Fusion), so
heterogeneous embedders / modes can never mis-order the merge, and
colliding `artifact_id`s from different sources stay distinct results
(identity is `(source, artifact_id)`).

A single-source pool (or an untagged one — hits with no `source`) keeps
its raw scores and exactly [`score_reranker()`](#raglab.score_reranker)’s ordering; fused,
rank-derived scores only appear when there is genuinely something to fuse.
Each fused hit keeps its pre-fusion magnitude as
`metadata["source_score"]`. For per-source weights, another `rrf_k`,
or opt-in cross-source duplicate merging, use [`make_rrf_reranker()`](#raglab.make_rrf_reranker).

* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`SearchHit`]

### *class* raglab.SubTask(goal, sources)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A planner’s unit of work: a sub-goal bound to a set of registered sources.

### raglab.arrival_order_cap(results, limit)

Keep the first `limit` hits in arrival order (source order x query order).

The pre-governor behaviour: a late source’s strong hits can be cut before the
evaluator sees them. Kept as an explicit opt-in (`pool_cap=arrival_order_cap`).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`SearchHit`]

### raglab.identity_citer(results)

No-op citer (verification needs a generated claim — that lives in srag).

* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`SearchHit`]

### raglab.identity_formulator(task, source)

Identity formulator: the sub-goal verbatim as one query (no rewrite/HyDE).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`LowLevelQuery`](raglab.agent.html.md#raglab.agent.LowLevelQuery)]

### raglab.ir_sources(\*names, \*\*search_defaults)

A source registry `{name: Retriever}` backed by named `ir` corpora.

Each name is bound to `ir.as_retriever(name, **search_defaults)`, opened
eagerly. For the *lazy* live view over everything registered (a corpus
opens only when its key is first used), use `ir.retrievers()` instead —
the agent accepts either, or any `Mapping[name, Retriever]`.
`search_defaults` (e.g. `mode="hybrid"`) apply to every source.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`SearchHit`]]]

### raglab.make_llm_evaluator(, judge=None, select_strategy='conservative', select_kwargs=None, prerank=None, prompt='A search agent is pursuing this goal:\\\\n\\\\n{goal}\\\\n\\\\nA calibrated selector reviewed the retrieved candidates and committed to the\\\\nresults below (it abstained if this list is empty):\\\\n\\\\n{results}\\\\n\\\\nDecide whether these results are SUFFICIENT to satisfy the goal.\\\\n- If they are, reply with exactly: SUFFICIENT\\\\n- If they are not, reply with: INSUFFICIENT\\\\n  then, on the next line, a single improved search query that would retrieve what\\\\n  is still missing. Keep it a terse search phrase — no prose, no numbering.\\\\n', max_result_chars=500, \*\*prompt_function_kwargs)

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

### raglab.make_llm_formulator(, formulate=None, params=None, \*\*make_kwargs)

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

### raglab.make_rrf_reranker(, rrf_k=60, weights=None, identity=None)

A parametrized [`rrf_reranker()`](#raglab.rrf_reranker) (per-source trust weights, `rrf_k`).

* **Parameters:**
  * **rrf_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – the RRF rank constant (standard default 60).
  * **weights** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional per-source trust dial, by source name (default 1.0
    each) — biases the merge without ever comparing raw scores.
  * **identity** (`Union`[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`SearchHit`], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – opt-in cross-source duplicate detection (e.g. `"pointer"`)
    — see `ir.retrieve.Identity`. Default: never merge across
    sources.
* **Return type:**
  [`Reranker`](raglab.agent.html.md#raglab.agent.Reranker)

### raglab.make_search_agent(sources, , planner=None, formulator=None, evaluator=None, reranker=None, citer=None, budget=None, pool_cap=None)

Build a [`SingleContextAgent`](#raglab.SingleContextAgent) over *sources* with smart defaults.

`sources` is a `Mapping[name, Retriever]` — e.g.
`{"skills": ir.as_retriever("skills")}`, [`ir_sources()`](#raglab.ir_sources), or the lazy
`ir.retrievers()` view. Every role defaults to its no-LLM thin-slice
implementation, so `make_search_agent(sources)("query")` just works;
inject an LLM `formulator` / `evaluator` to turn on rewriting and the
back-edge.

A custom `Retriever` must return `ir.SearchHit` instances (the
`Result` alias): the loop stamps provenance (`hit.source`) on its
output, so duck-typed hit objects raise at the tagging step.

* **Return type:**
  [`SingleContextAgent`](raglab.agent.html.md#raglab.agent.SingleContextAgent)

### raglab.passthrough_evaluator(task, results)

Pass-through critic: keep everything, declare sufficient, never re-query.

* **Return type:**
  [`Judgement`](raglab.agent.html.md#raglab.agent.Judgement)

### raglab.rank_fair_cap(results, limit)

Keep at most `limit` hits, taking the best of each source in turn (the default).

Groups the pool by `hit.source` (in first-arrival order; untagged hits are one
group), collapses each group to one hit per artifact (its best surface, as
`ir.base.best_per_artifact()` does), orders the group best-first by its own raw
score – sound, since one source is one score scale – and deals hits round-robin
across the groups. So a late source’s strong hits are never cut before the evaluator
sees them, a duplicate never takes a slot, and no score ever crosses a source
boundary or is rewritten (unlike pre-ranking with the RRF reranker, whose fused
ordinal scores must never re-enter the pool). The output is in that dealt order
whatever the pool’s size. A missing or NaN score sorts last in its source.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`SearchHit`]

### raglab.rrf_reranker(results)

Cross-source merge (the default fan-in): fuse by rank, never by raw score.

Groups the accumulated pool by `hit.source` and delegates the merge to
`ir.fuse_hits()` (ir is the SSOT for hit operations): within each
source raw scores order and dedup that source’s hits — one scale, sound —
and across sources only **ranks** interact (Reciprocal Rank Fusion), so
heterogeneous embedders / modes can never mis-order the merge, and
colliding `artifact_id`s from different sources stay distinct results
(identity is `(source, artifact_id)`).

A single-source pool (or an untagged one — hits with no `source`) keeps
its raw scores and exactly [`score_reranker()`](#raglab.score_reranker)’s ordering; fused,
rank-derived scores only appear when there is genuinely something to fuse.
Each fused hit keeps its pre-fusion magnitude as
`metadata["source_score"]`. For per-source weights, another `rrf_k`,
or opt-in cross-source duplicate merging, use [`make_rrf_reranker()`](#raglab.make_rrf_reranker).

* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`SearchHit`]

### raglab.score_reranker(results)

Magnitude merge: one surface per artifact, ordered by descending raw score.

Delegates to `ir.base.best_per_artifact()` (ir is the SSOT for hit
operations): an artifact retrieved by several queries / rounds — common once
the back-edge re-queries — survives once, at its highest score. Identity is
`(source, artifact_id)`, so two sources’ same-id artifacts never collapse.

A plain score sort compares raw scores **across** sources, which is only
sound when every source shares one score scale (same embedder + mode) — it
is the explicit homogeneous-sources opt-in. The default fan-in is
[`rrf_reranker()`](#raglab.rrf_reranker), which never compares raw scores across sources.

* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`SearchHit`]

### raglab.single_subtask_planner(query, sources)

Trivial planner: one sub-task over *all* registered sources, no decomposition.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SubTask`](raglab.agent.html.md#raglab.agent.SubTask)]

### Modules

| [`agent`](raglab.agent.html.md#module-raglab.agent)   | The Composable Search Agent — roles wired by a control loop (ir_09).    |
|------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`llm`](raglab.llm.html.md#module-raglab.llm)       | LLM-backed roles — the Formulator and Evaluator (ir_09 §8 steps 1 & 3). |
