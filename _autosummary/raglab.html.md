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
| [`apply_gates`](#raglab.apply_gates)(proposed, content, \*[, levels, ...]) | The level a judge's proposal is allowed, after calibration and the gates.                                                        |
| [`make_graded_rubric_evaluator`](#raglab.make_graded_rubric_evaluator)(\*[, levels, ...])   | An [`Evaluator`](raglab.agent.html.md#raglab.agent.Evaluator) that grades every candidate on a rubric.    |

### Classes

| [`Query`](#raglab.Query)(text[, constraints])                       | A user intent: free text plus optional structured constraints.                                                    |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [`SubTask`](#raglab.SubTask)(goal, sources)                           | A planner's unit of work: a sub-goal bound to a set of registered sources.                                        |
| [`LowLevelQuery`](#raglab.LowLevelQuery)(source, query[, params])           | One concrete query against one source.                                                                            |
| [`Judgement`](#raglab.Judgement)(relevant, sufficient[, ...])           | An evaluator's verdict over a round's results.                                                                    |
| [`GradedLabel`](#raglab.GradedLabel)(level, is_about, is_supported, ...)  | A graded verdict on one candidate (raglab#8): a level, three axes, the evidence.                                  |
| [`Result`](#raglab.Result)                                           |                                                                                                                   |
| [`Planner`](#raglab.Planner)(\*args, \*\*kwargs)                      | Decompose a query into sub-tasks and select sources for each.                                                     |
| [`Formulator`](#raglab.Formulator)(\*args, \*\*kwargs)                   | Turn a sub-task + one source into concrete low-level queries.                                                     |
| [`Evaluator`](#raglab.Evaluator)(\*args, \*\*kwargs)                    | Judge relevance + sufficiency; optionally emit a refinement (back-edge).                                          |
| [`Reranker`](#raglab.Reranker)(\*args, \*\*kwargs)                     | Produce the final ordering over the (cross-source) merged results.                                                |
| [`Citer`](#raglab.Citer)(\*args, \*\*kwargs)                        | Confirm/annotate that each result supports its use (identity by default).                                         |
| [`Budget`](#raglab.Budget)([max_rounds, max_sources_per_task, ...])  | Loop bounds: the safety net under the (harder) sufficiency decision.                                              |
| [`PoolCap`](#raglab.PoolCap)(\*args, \*\*kwargs)                      | Bound the pool the evaluator sees to at most `limit` results.                                                     |
| [`Memory`](#raglab.Memory)(\*args, \*\*kwargs)                       | Cross-run memory: recall hits for a purpose; take [`RoundRecord`](#raglab.RoundRecord) s. |
| [`PurposeStore`](#raglab.PurposeStore)([backend, embedder, ...])           | `MutableMapping[purpose, list[ExtractionUnit]]` over an injected `backend`.                                       |
| [`ExtractionUnit`](#raglab.ExtractionUnit)(purpose, hit, recorded_at[, ...]) | One piece of evidence a run committed to, for one purpose.                                                        |
| [`RoundRecord`](#raglab.RoundRecord)(purpose, round, goal, sources, ...)  | What one round of the loop decided: the run-log's unit (raglab#6).                                                |
| [`RunLog`](#raglab.RunLog)(\*args, \*\*kwargs)                       | A sink for the loop's [`RoundRecord`](#raglab.RoundRecord) s, one call per round.         |
| [`SingleContextAgent`](#raglab.SingleContextAgent)(sources[, planner, ...])      | One ReAct-style loop, sequential sub-tasks (ir_09 §7 — the cheap default).                                        |

### *class* raglab.Budget(max_rounds=3, max_sources_per_task=4, max_results_per_task=50)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Loop bounds: the safety net under the (harder) sufficiency decision.

### *class* raglab.Citer(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Confirm/annotate that each result supports its use (identity by default).

### *class* raglab.Evaluator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Judge relevance + sufficiency; optionally emit a refinement (back-edge).

### *class* raglab.ExtractionUnit(purpose, hit, recorded_at, importance=1.0, confidence=None, level='raw', supersedes=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One piece of evidence a run committed to, for one purpose.

`hit` is the `SearchHit.to_dict()` of the kept hit: provenance
(`source`, `artifact_id`) and its raw magnitude, never a fused score.
`level` is the consolidation level (`"raw"` until CONSOLIDATE exists);
`supersedes` names the artifact id a later unit replaces (for SUPERSEDE).

#### *classmethod* from_dict(d)

Inverse of [`to_dict()`](#raglab.ExtractionUnit.to_dict).

* **Return type:**
  [`ExtractionUnit`](raglab.memory.html.md#raglab.memory.ExtractionUnit)

#### *property* identity *: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None), [str](https://docs.python.org/3/builtins/stdtypes.html#str)]*

ir’s identity of the artifact this points at.

* **Type:**
  `(source, artifact_id)`

#### to_dict()

JSON-clean form.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

#### to_result()

The `SearchHit` this unit was made from.

* **Return type:**
  `SearchHit`

### *class* raglab.Formulator(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Turn a sub-task + one source into concrete low-level queries.

### *class* raglab.GradedLabel(level, is_about, is_supported, is_useful, evidence_quote='', facet=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A graded verdict on one candidate (raglab#8): a level, three axes, the evidence.

`level` is one of the evaluator’s ordered levels (best first). The axes are
Self-RAG’s: `is_about` (topical, ISREL), `is_supported` (the content
actually establishes it, not a lexical mention: ISSUP) and `is_useful`
(ISUSE). `evidence_quote` is a verbatim span of the candidate’s content
backing the level; `facet` names the sub-goal it was judged against.

### *class* raglab.Judgement(relevant, sufficient, refinement=None, graded=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

An evaluator’s verdict over a round’s results.

`relevant` is the kept subset; `sufficient` says whether to stop; a
non-`None` `refinement` is the **back-edge** — the next sub-task to
re-query with. The pass-through default returns `sufficient=True` and no
refinement (no loop). `graded` is optional and additive: a graded
evaluator maps each judged `artifact_id` to its [`GradedLabel`](#raglab.GradedLabel);
binary evaluators leave it empty.

### *class* raglab.LowLevelQuery(source, query, params=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One concrete query against one source.

`query` is the text handed to the source’s `Retriever`; `params`
are per-call retriever overrides (e.g. `mode` / `filter` / `k` for an
ir corpus). A formulator turns a [`SubTask`](#raglab.SubTask) into these.

### *class* raglab.Memory(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Cross-run memory: recall hits for a purpose; take [`RoundRecord`](#raglab.RoundRecord) s.

Opt-in (`SingleContextAgent.memory` defaults to `None`). Before sweeping a
sub-task, the agent offers the recalled hits to the evaluator; if they are
sufficient, it does not sweep. [`raglab.memory.PurposeStore`](raglab.memory.html.md#raglab.memory.PurposeStore) is one.

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

### *class* raglab.PurposeStore(backend=dict(), embedder=None, write_strategy=<function upsert_write>, read_strategy=<function make_relevance_recency_reader.<locals>.read>, min_relevance=0.8, clock=<built-in function time>)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

`MutableMapping[purpose, list[ExtractionUnit]]` over an injected `backend`.

`backend` maps `purpose_id()` to a JSON-clean record
`{"purpose", "embedding", "units"}`: any `MutableMapping` (a dict, a dol
store of JSON files, …). `store[purpose]` recalls from every *related*
purpose (the same one, or one whose embedding is at least `min_relevance`
cosine-similar); `store[purpose] = units` replaces that purpose’s units.
Calling the store with a [`RoundRecord`](#raglab.RoundRecord) journals what the round kept.

#### clock()

time() -> floating-point number

Return the current time in seconds since the Epoch.
Fractions of a second may be present if the system clock provides them.

#### recall(purpose)

The units of every purpose related to `purpose`, ranked by `read_strategy`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`ExtractionUnit`](raglab.memory.html.md#raglab.memory.ExtractionUnit)]

#### recall_hits(purpose)

[`recall()`](#raglab.PurposeStore.recall) as `SearchHit` s, ready to seed the agent’s pool.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`SearchHit`]

#### write(purpose, units)

Merge `units` into `purpose`’s, with the `write_strategy`.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### write_strategy(new)

ADD new artifacts; UPDATE (replace with the newer unit) those already kept.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`ExtractionUnit`](raglab.memory.html.md#raglab.memory.ExtractionUnit)]

### *class* raglab.Query(text, constraints=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A user intent: free text plus optional structured constraints.

### *class* raglab.Reranker(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Produce the final ordering over the (cross-source) merged results.

### raglab.Result

alias of `SearchHit`

### *class* raglab.RoundRecord(purpose, round, goal, sources, sufficient, refinement, committed, graded=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

What one round of the loop decided: the run-log’s unit (raglab#6).

Written at the evaluator’s judgement point, it is the single record shape that
per-run observability (the run-log) and cross-run memory (a purpose store)
both consume. `purpose` is the sub-task’s original goal; `goal` is what this
round actually searched (a refinement, from round 1 on). `committed` holds the
hits the evaluator kept, with their **raw** per-source magnitudes: the pool never
carries fused scores, so neither does the record. Round `-1` is the recall
round of an agent with a `memory`: its pool was recalled, not swept.

#### to_dict()

JSON-clean form: hits via `SearchHit.to_dict`, labels as plain dicts.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

### *class* raglab.RunLog(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

A sink for the loop’s [`RoundRecord`](#raglab.RoundRecord) s, one call per round.

Opt-in (the agent’s `run_log` defaults to `None`: nothing is recorded).
`list.append` is one: `log = []; make_search_agent(sources, run_log=log.append)`.

### *class* raglab.SingleContextAgent(sources, planner=<function single_subtask_planner>, formulator=<function identity_formulator>, evaluator=<function passthrough_evaluator>, reranker=<function rrf_reranker>, citer=<function identity_citer>, budget=Budget(), pool_cap=<function rank_fair_cap>, run_log=None, memory=None)

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

### raglab.apply_gates(proposed, content, , levels=('core', 'strong', 'uses-tools', 'tangential', 'none'), unsupported_cap='tangential', require_evidence_quote=True, calibration_offset=None, min_quote_chars=8)

The level a judge’s proposal is allowed, after calibration and the gates.

The quote must be at least `min_quote_chars` long and found in `content`,
ignoring case and runs of whitespace (judges re-case and re-wrap what they
copy). `calibration_offset` is in levels, rounded half up, so a shift is
the same at every level.

```pycon
>>> label = GradedLabel('core', True, False, True, 'a DAG of tasks')
>>> apply_gates(label, 'Runs a DAG of tasks asynchronously.').level
'tangential'
>>> ok = GradedLabel('core', True, True, True, 'graph  library')
>>> apply_gates(ok, 'A graph library.').level, apply_gates(ok, 'Nothing here.').level
('core', 'tangential')
>>> apply_gates(ok, 'A graph library.', calibration_offset=1.4).level
'strong'
```

* **Return type:**
  [`GradedLabel`](raglab.agent.html.md#raglab.agent.GradedLabel)

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

### raglab.make_graded_rubric_evaluator(, levels=('core', 'strong', 'uses-tools', 'tangential', 'none'), rubric="- core: the candidate's main purpose IS the topic.\\n- strong: a substantial, central part of the candidate is about the topic.\\n- uses-tools: the candidate builds on tools or libraries of the topic (its dependencies or imports establish it) without being about it.\\n- tangential: the topic is mentioned, or its vocabulary appears, but the content does not establish it.\\n- none: unrelated.\\n", judge=None, disclose_level='body', disclose_kwargs=None, require_evidence_quote=True, unsupported_cap='tangential', calibration_offset=None, min_quote_chars=8, relevant_level='uses-tools', refine=None, rng=None, prompt='Goal: {goal}\\n\\nRubric (levels, in no particular order):\\n{levels}\\n\\nDefinitions:\\n{rubric}\\n\\nCandidate {artifact_id}:\\n---\\n{content}\\n---\\n\\nAnswer with one JSON object and nothing else, with keys:\\n"is_about" (bool: is the candidate topically about the goal?),\\n"is_supported" (bool: does its content - code, dependencies, imports, full text - actually establish it, beyond a mention of the words?),\\n"is_useful" (bool: would it help someone pursuing the goal?),\\n"evidence_quote" (a short span copied VERBATIM from the candidate\\'s content that supports your level),\\n"level" (one of the rubric\\'s levels).\\n', max_content_chars=4000, \*\*prompt_function_kwargs)

An [`Evaluator`](raglab.agent.html.md#raglab.agent.Evaluator) that grades every candidate on a rubric.

* **Parameters:**
  * **levels** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the ordered levels, best first.
  * **rubric** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – what each level means, shown to the judge.
  * **judge** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – an injectable graded judge (a test double, or your own router);
    built lazily on `oa` when omitted.
  * **disclose_level** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – what the judge reads — each candidate’s
    content through `ir.disclose()` at that level (`store=` /
    `loader=` go in `disclose_kwargs`); a candidate with nothing to
    load is judged on its snippet.
  * **disclose_kwargs** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – what the judge reads — each candidate’s
    content through `ir.disclose()` at that level (`store=` /
    `loader=` go in `disclose_kwargs`); a candidate with nothing to
    load is judged on its snippet.
  * **require_evidence_quote** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – min_quote_chars: the gates, see [`apply_gates()`](#raglab.apply_gates).
  * **unsupported_cap** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – min_quote_chars: the gates, see [`apply_gates()`](#raglab.apply_gates).
  * **calibration_offset** ([`float`](https://docs.python.org/3/builtins/functions.html#float) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – min_quote_chars: the gates, see [`apply_gates()`](#raglab.apply_gates).
* **Return type:**
  [`Evaluator`](raglab.agent.html.md#raglab.agent.Evaluator)

:param : min_quote_chars: the gates, see [`apply_gates()`](#raglab.apply_gates).
:type relevant_level: [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
:param relevant_level: the lowest level kept in `Judgement.relevant`.
:type refine: [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`SubTask`](raglab.agent.html.md#raglab.agent.SubTask), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`GradedLabel`](raglab.agent.html.md#raglab.agent.GradedLabel)]], [`SubTask`](raglab.agent.html.md#raglab.agent.SubTask) | [`None`](https://docs.python.org/3/builtins/constants.html#None)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)
:param refine: the back-edge policy `(task, graded) -> SubTask | None`; by

> default none, so the loop stops after one round.
* **Parameters:**
  **rng** ([`Random`](https://docs.python.org/3/library/random.html#random.Random) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the source of the per-call level order (seed it for reproducible
  prompts).
* **Return type:**
  [*Evaluator*](raglab.agent.html.md#raglab.agent.Evaluator)

Every candidate is judged once per goal: one judge call per (source,
artifact) the first time it appears in a pool for that goal, its label
remembered for later rounds, so a kept hit is not re-judged (nor lost to a
later failing call). `Judgement.graded` is keyed by `artifact_id`; an id
found in several sources carries its best label. A reply that does not parse
leaves the candidate ungraded (logged); when every judge call of a round
raises, the first error is re-raised rather than returning an empty,
“sufficient” round. This evaluator is for labelling a pool, not for cheap
filtering.

### raglab.make_llm_evaluator(, judge=None, select_strategy='conservative', select_kwargs=None, prerank=None, prompt='A search agent is pursuing this goal:\\n\\n{goal}\\n\\nA calibrated selector reviewed the retrieved candidates and committed to the\\nresults below (it abstained if this list is empty):\\n\\n{results}\\n\\nDecide whether these results are SUFFICIENT to satisfy the goal.\\n- If they are, reply with exactly: SUFFICIENT\\n- If they are not, reply with: INSUFFICIENT\\n  then, on the next line, a single improved search query that would retrieve what\\n  is still missing. Keep it a terse search phrase — no prose, no numbering.\\n', max_result_chars=500, \*\*prompt_function_kwargs)

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
  * **identity** ([`Union`](https://docs.python.org/3/library/typing.html#typing.Union)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`SearchHit`], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – opt-in cross-source duplicate detection (e.g. `"pointer"`)
    — see `ir.retrieve.Identity`. Default: never merge across
    sources.
* **Return type:**
  [`Reranker`](raglab.agent.html.md#raglab.agent.Reranker)

### raglab.make_search_agent(sources, , planner=None, formulator=None, evaluator=None, reranker=None, citer=None, budget=None, pool_cap=None, run_log=None, memory=None)

Build a [`SingleContextAgent`](#raglab.SingleContextAgent) over *sources* with smart defaults.

`sources` is a `Mapping[name, Retriever]` — e.g.
`{"skills": ir.as_retriever("skills")}`, [`ir_sources()`](#raglab.ir_sources), or the lazy
`ir.retrievers()` view. Every role defaults to its no-LLM thin-slice
implementation, so `make_search_agent(sources)("query")` just works;
inject an LLM `formulator` / `evaluator` to turn on rewriting and the
back-edge. Pass a `run_log` (any callable taking a [`RoundRecord`](#raglab.RoundRecord), e.g.
`list.append`) to record what each round decided, and a `memory` (e.g. a
[`raglab.memory.PurposeStore`](raglab.memory.html.md#raglab.memory.PurposeStore)) to journal what runs keep and reuse it.

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

| [`agent`](raglab.agent.html.md#module-raglab.agent)     | The Composable Search Agent — roles wired by a control loop (ir_09).                      |
|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [`bench`](raglab.bench.html.md#module-raglab.bench)     | Cross-run benchmark: the instrument for purpose-indexed memory (raglab#15, item 4).       |
| [`judging`](raglab.judging.html.md#module-raglab.judging) | Graded rubric judging — every candidate gets a level, three axes and evidence (raglab#8). |
| [`llm`](raglab.llm.html.md#module-raglab.llm)         | LLM-backed roles — the Formulator and Evaluator (ir_09 §8 steps 1 & 3).                   |
| [`memory`](raglab.memory.html.md#module-raglab.memory)   | Purpose-indexed memory: what past runs found, kept per purpose (raglab#6, stage 0).       |
| [`recipes`](raglab.recipes.html.md#module-raglab.recipes) | Recipes: whole workflows composed from raglab's roles, ready to call.                     |
