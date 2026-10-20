# raglab.bench

Cross-run benchmark: the instrument for purpose-indexed memory (raglab#15, item 4).

The follow-ups of the memory design (CONSOLIDATE, SUPERSEDE, purpose transfer, DECAY)
are each gated on an *observed* trigger. This module is what observes: it replays a
workload of purposes through an agent twice, once cold (no memory) and once warm (with
a [`PurposeStore`](raglab.memory.html.md#raglab.memory.PurposeStore)), counting the source calls of each, and reports

* `sweeps_cold` / `sweeps_warm`: retriever calls without and with memory (what
  memory saves),
* `recall`: the share of the cold run’s hits the warm run also returned (what memory
  costs in answer quality: `1.0` means nothing was lost),
* `extractions`: units kept per purpose after the warm run (the CONSOLIDATE
  trigger is “this exceeds a threshold”).

Cold and warm calls interleave on the same sources, purpose by purpose, so a source
that changes over time shows as lost recall: that is the SUPERSEDE trigger, but so is
plain non-determinism (an approximate index, a live API). Run the benchmark on a
deterministic snapshot to read `recall` as what memory alone lost.

Hermetic and offline: pass any sources; nothing here calls an LLM:

```default
report = cross_run_benchmark(sources, ["invoices 2023", "2023 invoices"])
report.sweeps_saved, report.recall
```

### Functions

| [`cross_run_benchmark`](#raglab.bench.cross_run_benchmark)(sources, workload, \*[, ...])   | Replay `workload` (purposes, in order, repeats and paraphrases welcome) cold and warm, and report what memory saved and what it lost.   |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| [`extraction_counts`](#raglab.bench.extraction_counts)(store)                            | `{purpose: number of units kept}` (each purpose under its own statement).                                                               |

### Classes

| [`CrossRunReport`](#raglab.bench.CrossRunReport)(rows[, extractions])   | Totals over a workload, with the per-purpose `rows` and `extractions`.       |
|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`PurposeRow`](#raglab.bench.PurposeRow)(purpose, sweeps_cold, ...) | One purpose of the workload: its source calls cold and warm, and its recall. |

### *class* raglab.bench.CrossRunReport(rows, extractions=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Totals over a workload, with the per-purpose `rows` and `extractions`.

#### *property* recall *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Mean, over purposes, of the share of cold hits the warm run also returned.

#### *property* sweeps_saved *: [float](https://docs.python.org/3/builtins/functions.html#float)*

Share of the cold source calls the warm run did not make.

### *class* raglab.bench.PurposeRow(purpose, sweeps_cold, sweeps_warm, recall)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One purpose of the workload: its source calls cold and warm, and its recall.

### raglab.bench.cross_run_benchmark(sources, workload, \*, agent_factory=<function make_search_agent>, memory_factory=<class 'raglab.memory.PurposeStore'>, \*\*agent_kwargs)

Replay `workload` (purposes, in order, repeats and paraphrases welcome) cold
and warm, and report what memory saved and what it lost.

`agent_factory(sources, memory=..., **agent_kwargs)` builds the agent (default
[`make_search_agent()`](raglab.agent.html.md#raglab.agent.make_search_agent)); the cold agent gets `memory=None`.
The warm agent shares one `memory_factory()` store across the whole workload.

* **Return type:**
  [`CrossRunReport`](#raglab.bench.CrossRunReport)

### raglab.bench.extraction_counts(store)

`{purpose: number of units kept}` (each purpose under its own statement).

```pycon
>>> from raglab.memory import ExtractionUnit
>>> s = PurposeStore()
>>> s["q"] = [ExtractionUnit("q", {"artifact_id": "a"}, 0.0)]
>>> extraction_counts(s)
{'q': 1}
```

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`int`](https://docs.python.org/3/builtins/functions.html#int)]
