# raglab.agent

The Composable Search Agent — roles wired by a control loop (ir_09).

`raglab` is the \*\*orchestration layer on top of `ir`** (the retrieval substrate).
This module is the v1 foundation: the immutable value types, the role *Protocols*
(open-closed strategy seams), and a `SingleContextAgent` whose fixed control loop
is fully parametrized by injected roles. Concrete tools live at the leaves — an
`ir` corpus becomes one `Retriever` via `ir.as_retriever()`.

The agent is **multi-source by default**: the loop stamps each hit’s
provenance (`hit.source`), and the fan-in [`Reranker`](#raglab.agent.Reranker) —
[`rrf_reranker()`](#raglab.agent.rrf_reranker) — merges heterogeneous sources by *rank*, never by raw
score (scores from different corpora / embedders / modes are incommensurable;
ir_07/ir_08). Raw magnitudes order and dedup hits *within* one source, and the
loop’s pool always carries them; fused (ordinal) scores appear only at the
fan-in boundary.

The shape follows ir_09 §3/§6: a small set of named roles —
`Planner / Formulator / Retriever / Evaluator / Reranker / Citer` — and a loop
whose defining feature is the **back-edge** (evaluator → reformulate) that makes
it an *agent* rather than a DAG. v1 ships the loop with smart defaults (a trivial
planner + a pass-through evaluator), so the thin slice runs end-to-end with no
LLM; turning on the back-edge is just injecting an LLM `Evaluator` that returns
a `refinement` (ir_09 §8 step 3).

Progressive disclosure: [`make_search_agent()`](#raglab.agent.make_search_agent) gives sensible defaults for
every role, so the simple path is `make_search_agent(sources)("query")`.

Dependency direction is one-way: `raglab` imports `ir`; `ir` never imports
`raglab`. The `Result` type and the `Retriever` contract are ir’s (SSOT).

### Module Attributes

| [`Result`](#raglab.agent.Result)   | A retrieved item — ir's `SearchHit` (ir_09's `Result`): a *pointer + snippet* (`text`) with a `score` and `metadata`.   |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|

### Functions

| [`rank_fair_cap`](#raglab.agent.rank_fair_cap)(results, limit)                     | Keep at most `limit` hits, taking the best of each source in turn (the default).                                    |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`arrival_order_cap`](#raglab.agent.arrival_order_cap)(results, limit)                 | Keep the first `limit` hits in arrival order (source order x query order).                                          |
| [`make_search_agent`](#raglab.agent.make_search_agent)(sources, \*[, planner, ...])    | Build a [`SingleContextAgent`](#raglab.agent.SingleContextAgent) over *sources* with smart defaults.     |
| [`ir_sources`](#raglab.agent.ir_sources)(\*names, \*\*search_defaults)          | A source registry `{name: Retriever}` backed by named `ir` corpora.                                                 |
| [`single_subtask_planner`](#raglab.agent.single_subtask_planner)(query, sources)            | Trivial planner: one sub-task over *all* registered sources, no decomposition.                                      |
| [`identity_formulator`](#raglab.agent.identity_formulator)(task, source)                 | Identity formulator: the sub-goal verbatim as one query (no rewrite/HyDE).                                          |
| [`passthrough_evaluator`](#raglab.agent.passthrough_evaluator)(task, results)              | Pass-through critic: keep everything, declare sufficient, never re-query.                                           |
| [`score_reranker`](#raglab.agent.score_reranker)(results)                           | Magnitude merge: one surface per artifact, ordered by descending raw score.                                         |
| [`rrf_reranker`](#raglab.agent.rrf_reranker)(results)                             | Cross-source merge (the default fan-in): fuse by rank, never by raw score.                                          |
| [`make_rrf_reranker`](#raglab.agent.make_rrf_reranker)(\*[, rrf_k, weights, identity]) | A parametrized [`rrf_reranker()`](#raglab.agent.rrf_reranker) (per-source trust weights, `rrf_k`). |
| [`identity_citer`](#raglab.agent.identity_citer)(results)                           | No-op citer (verification needs a generated claim — that lives in srag).                                            |

### Classes

| [`Query`](#raglab.agent.Query)(text[, constraints])                      | A user intent: free text plus optional structured constraints.                                                        |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [`SubTask`](#raglab.agent.SubTask)(goal, sources)                          | A planner's unit of work: a sub-goal bound to a set of registered sources.                                            |
| [`LowLevelQuery`](#raglab.agent.LowLevelQuery)(source, query[, params])          | One concrete query against one source.                                                                                |
| [`Judgement`](#raglab.agent.Judgement)(relevant, sufficient[, ...])          | An evaluator's verdict over a round's results.                                                                        |
| [`GradedLabel`](#raglab.agent.GradedLabel)(level, is_about, is_supported, ...) | A graded verdict on one candidate (raglab#8): a level, three axes, the evidence.                                      |
| [`Result`](#raglab.agent.Result)                                          | A retrieved item — ir's `SearchHit` (ir_09's `Result`): a *pointer + snippet* (`text`) with a `score` and `metadata`. |
| [`Planner`](#raglab.agent.Planner)(\*args, \*\*kwargs)                     | Decompose a query into sub-tasks and select sources for each.                                                         |
| [`Formulator`](#raglab.agent.Formulator)(\*args, \*\*kwargs)                  | Turn a sub-task + one source into concrete low-level queries.                                                         |
| [`Evaluator`](#raglab.agent.Evaluator)(\*args, \*\*kwargs)                   | Judge relevance + sufficiency; optionally emit a refinement (back-edge).                                              |
| [`Reranker`](#raglab.agent.Reranker)(\*args, \*\*kwargs)                    | Produce the final ordering over the (cross-source) merged results.                                                    |
| [`Citer`](#raglab.agent.Citer)(\*args, \*\*kwargs)                       | Confirm/annotate that each result supports its use (identity by default).                                             |
| [`Budget`](#raglab.agent.Budget)([max_rounds, max_sources_per_task, ...]) | Loop bounds: the safety net under the (harder) sufficiency decision.                                                  |
| [`PoolCap`](#raglab.agent.PoolCap)(\*args, \*\*kwargs)                     | Bound the pool the evaluator sees to at most `limit` results.                                                         |
| [`SingleContextAgent`](#raglab.agent.SingleContextAgent)(sources[, planner, ...])     | One ReAct-style loop, sequential sub-tasks (ir_09 §7 — the cheap default).                                            |

### *class* raglab.agent.Budget(max_rounds=3, max_sources_per_task=4, max_results_per_task=50)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Loop bounds: the safety net under the (harder) sufficiency decision.

### *class* raglab.agent.Citer(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Confirm/annotate that each result supports its use (identity by default).

### *class* raglab.agent.Evaluator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Judge relevance + sufficiency; optionally emit a refinement (back-edge).

### *class* raglab.agent.Formulator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Turn a sub-task + one source into concrete low-level queries.

### *class* raglab.agent.GradedLabel(level, is_about, is_supported, is_useful, evidence_quote='', facet=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A graded verdict on one candidate (raglab#8): a level, three axes, the evidence.

`level` is one of the evaluator’s ordered levels (best first). The axes are
Self-RAG’s: `is_about` (topical, ISREL), `is_supported` (the content
actually establishes it, not a lexical mention: ISSUP) and `is_useful`
(ISUSE). `evidence_quote` is a verbatim span of the candidate’s content
backing the level; `facet` names the sub-goal it was judged against.

### *class* raglab.agent.Judgement(relevant, sufficient, refinement=None, graded=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An evaluator’s verdict over a round’s results.

`relevant` is the kept subset; `sufficient` says whether to stop; a
non-`None` `refinement` is the **back-edge** — the next sub-task to
re-query with. The pass-through default returns `sufficient=True` and no
refinement (no loop). `graded` is optional and additive: a graded
evaluator maps each judged `artifact_id` to its [`GradedLabel`](#raglab.agent.GradedLabel);
binary evaluators leave it empty.

### *class* raglab.agent.LowLevelQuery(source, query, params=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One concrete query against one source.

`query` is the text handed to the source’s `Retriever`; `params`
are per-call retriever overrides (e.g. `mode` / `filter` / `k` for an
ir corpus). A formulator turns a [`SubTask`](#raglab.agent.SubTask) into these.

### *class* raglab.agent.Planner(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Decompose a query into sub-tasks and select sources for each.

### *class* raglab.agent.PoolCap(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Bound the pool the evaluator sees to at most `limit` results.

A governor decision (which hits survive the budget), injected like every other
role so a strategy can be swapped without touching the loop. It must return
hits **unchanged** — never re-scored — because the pool keeps raw per-source
magnitudes until the fan-in reranker.

### *class* raglab.agent.Query(text, constraints=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A user intent: free text plus optional structured constraints.

### *class* raglab.agent.Reranker(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Produce the final ordering over the (cross-source) merged results.

### raglab.agent.Result

A retrieved item — ir’s `SearchHit` (ir_09’s `Result`):
a *pointer + snippet* (`text`) with a `score` and `metadata`.

### *class* raglab.agent.SingleContextAgent(sources, planner=<function single_subtask_planner>, formulator=<function identity_formulator>, evaluator=<function passthrough_evaluator>, reranker=<function rrf_reranker>, citer=<function identity_citer>, budget=Budget(), pool_cap=<function rank_fair_cap>, run_log=None)

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
  [`Judgement`](#raglab.agent.Judgement)

#### formulator(source)

Identity formulator: the sub-goal verbatim as one query (no rewrite/HyDE).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`LowLevelQuery`](#raglab.agent.LowLevelQuery)]

#### planner(sources)

Trivial planner: one sub-task over *all* registered sources, no decomposition.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SubTask`](#raglab.agent.SubTask)]

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
its raw scores and exactly [`score_reranker()`](#raglab.agent.score_reranker)’s ordering; fused,
rank-derived scores only appear when there is genuinely something to fuse.
Each fused hit keeps its pre-fusion magnitude as
`metadata["source_score"]`. For per-source weights, another `rrf_k`,
or opt-in cross-source duplicate merging, use [`make_rrf_reranker()`](#raglab.agent.make_rrf_reranker).

* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`SearchHit`]

### *class* raglab.agent.SubTask(goal, sources)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A planner’s unit of work: a sub-goal bound to a set of registered sources.

### raglab.agent.arrival_order_cap(results, limit)

Keep the first `limit` hits in arrival order (source order x query order).

The pre-governor behaviour: a late source’s strong hits can be cut before the
evaluator sees them. Kept as an explicit opt-in (`pool_cap=arrival_order_cap`).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`SearchHit`]

### raglab.agent.identity_citer(results)

No-op citer (verification needs a generated claim — that lives in srag).

* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`SearchHit`]

### raglab.agent.identity_formulator(task, source)

Identity formulator: the sub-goal verbatim as one query (no rewrite/HyDE).

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`LowLevelQuery`](#raglab.agent.LowLevelQuery)]

### raglab.agent.ir_sources(\*names, \*\*search_defaults)

A source registry `{name: Retriever}` backed by named `ir` corpora.

Each name is bound to `ir.as_retriever(name, **search_defaults)`, opened
eagerly. For the *lazy* live view over everything registered (a corpus
opens only when its key is first used), use `ir.retrievers()` instead —
the agent accepts either, or any `Mapping[name, Retriever]`.
`search_defaults` (e.g. `mode="hybrid"`) apply to every source.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`SearchHit`]]]

### raglab.agent.make_rrf_reranker(, rrf_k=60, weights=None, identity=None)

A parametrized [`rrf_reranker()`](#raglab.agent.rrf_reranker) (per-source trust weights, `rrf_k`).

* **Parameters:**
  * **rrf_k** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – the RRF rank constant (standard default 60).
  * **weights** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`float`](https://docs.python.org/3/builtins/functions.html#float)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – optional per-source trust dial, by source name (default 1.0
    each) — biases the merge without ever comparing raw scores.
  * **identity** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`SearchHit`], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – opt-in cross-source duplicate detection (e.g. `"pointer"`)
    — see `ir.retrieve.Identity`. Default: never merge across
    sources.
* **Return type:**
  [`Reranker`](#raglab.agent.Reranker)

### raglab.agent.make_search_agent(sources, , planner=None, formulator=None, evaluator=None, reranker=None, citer=None, budget=None, pool_cap=None, run_log=None)

Build a [`SingleContextAgent`](#raglab.agent.SingleContextAgent) over *sources* with smart defaults.

`sources` is a `Mapping[name, Retriever]` — e.g.
`{"skills": ir.as_retriever("skills")}`, [`ir_sources()`](#raglab.agent.ir_sources), or the lazy
`ir.retrievers()` view. Every role defaults to its no-LLM thin-slice
implementation, so `make_search_agent(sources)("query")` just works;
inject an LLM `formulator` / `evaluator` to turn on rewriting and the
back-edge. Pass a `run_log` (any callable taking a [`RoundRecord`](raglab.html.md#raglab.RoundRecord), e.g.
`list.append`) to record what each round decided.

A custom `Retriever` must return `ir.SearchHit` instances (the
`Result` alias): the loop stamps provenance (`hit.source`) on its
output, so duck-typed hit objects raise at the tagging step.

* **Return type:**
  [`SingleContextAgent`](#raglab.agent.SingleContextAgent)

### raglab.agent.passthrough_evaluator(task, results)

Pass-through critic: keep everything, declare sufficient, never re-query.

* **Return type:**
  [`Judgement`](#raglab.agent.Judgement)

### raglab.agent.rank_fair_cap(results, limit)

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

### raglab.agent.rrf_reranker(results)

Cross-source merge (the default fan-in): fuse by rank, never by raw score.

Groups the accumulated pool by `hit.source` and delegates the merge to
`ir.fuse_hits()` (ir is the SSOT for hit operations): within each
source raw scores order and dedup that source’s hits — one scale, sound —
and across sources only **ranks** interact (Reciprocal Rank Fusion), so
heterogeneous embedders / modes can never mis-order the merge, and
colliding `artifact_id`s from different sources stay distinct results
(identity is `(source, artifact_id)`).

A single-source pool (or an untagged one — hits with no `source`) keeps
its raw scores and exactly [`score_reranker()`](#raglab.agent.score_reranker)’s ordering; fused,
rank-derived scores only appear when there is genuinely something to fuse.
Each fused hit keeps its pre-fusion magnitude as
`metadata["source_score"]`. For per-source weights, another `rrf_k`,
or opt-in cross-source duplicate merging, use [`make_rrf_reranker()`](#raglab.agent.make_rrf_reranker).

* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`SearchHit`]

### raglab.agent.score_reranker(results)

Magnitude merge: one surface per artifact, ordered by descending raw score.

Delegates to `ir.base.best_per_artifact()` (ir is the SSOT for hit
operations): an artifact retrieved by several queries / rounds — common once
the back-edge re-queries — survives once, at its highest score. Identity is
`(source, artifact_id)`, so two sources’ same-id artifacts never collapse.

A plain score sort compares raw scores **across** sources, which is only
sound when every source shares one score scale (same embedder + mode) — it
is the explicit homogeneous-sources opt-in. The default fan-in is
[`rrf_reranker()`](#raglab.agent.rrf_reranker), which never compares raw scores across sources.

* **Return type:**
  [`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[`SearchHit`]

### raglab.agent.single_subtask_planner(query, sources)

Trivial planner: one sub-task over *all* registered sources, no decomposition.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`SubTask`](#raglab.agent.SubTask)]
