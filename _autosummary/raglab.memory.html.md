# raglab.memory

Purpose-indexed memory: what past runs found, kept per purpose (raglab#6, stage 0).

The retrieval index is purpose-free by design; the search agent is the one layer that
sees *purpose* (the sub-task goal, the refinement chain, the evaluator’s verdicts).
Without memory, everything a run learned dies when it returns. A [`PurposeStore`](#raglab.memory.PurposeStore)
keeps the precipitate: the hits each run committed to, as [`ExtractionUnit`](#raglab.memory.ExtractionUnit) s
keyed by purpose, carrying their provenance (`(source, artifact_id)` plus the whole
`SearchHit.to_dict()`, raw magnitudes only).

It is a [`RunLog`](raglab.html.md#raglab.RunLog) sink (writes come from the run-log’s
[`RoundRecord`](raglab.html.md#raglab.RoundRecord) s, the one shared record shape) and a
[`Memory`](raglab.html.md#raglab.Memory) the agent reads from before sweeping:

```default
store = PurposeStore()
agent = make_search_agent(sources, memory=store)
agent("invoices from 2023")  # sweeps, journals what it kept
agent("invoices from 2023")  # recalls those hits; sweeps only if not sufficient
```

Progressive disclosure: `store[purpose]` is the related-purpose read. With no
`embedder` only the same purpose (normalized text) is related; inject one
(`str -> vector`, e.g. a vd/ir embedder) and similar purposes share their findings.

The four verbs of the design (report 14) are injected strategies; stage 0 ships two
(`write_strategy`: ADD, or UPDATE an artifact already kept; `read_strategy`:
relevance x importance x recency). CONSOLIDATE and DECAY are follow-ups, gated on
observed need.

### Functions

| [`json_clean`](#raglab.memory.json_clean)(obj)                          | `obj` as plain JSON types (numpy values via `tolist`, the rest via `str`).                                         |
|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [`make_relevance_recency_reader`](#raglab.memory.make_relevance_recency_reader)(\*[, ...]) | A [`ReadStrategy`](#raglab.memory.ReadStrategy) scoring `relevance**alpha * importance * recency`. |
| [`normalize_purpose`](#raglab.memory.normalize_purpose)(statement)             | The canonical text of a purpose: case-folded, whitespace collapsed.                                                |
| [`purpose_id`](#raglab.memory.purpose_id)(statement)                    | Content-hash id of a purpose (of its [`normalize_purpose()`](#raglab.memory.normalize_purpose) text).   |
| `relevance_recency_reader`(candidates, \*, now)                                           |                                                                                                                    |
| [`upsert_write`](#raglab.memory.upsert_write)(existing, new)              | ADD new artifacts; UPDATE (replace with the newer unit) those already kept.                                        |

### Classes

| [`ExtractionUnit`](#raglab.memory.ExtractionUnit)(purpose, hit, recorded_at[, ...])   | One piece of evidence a run committed to, for one purpose.                       |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| [`PurposeStore`](#raglab.memory.PurposeStore)([backend, embedder, ...])             | `MutableMapping[purpose, list[ExtractionUnit]]` over an injected `backend`.      |
| [`ReadStrategy`](#raglab.memory.ReadStrategy)(\*args, \*\*kwargs)                   | Rank `(relevance, unit)` candidates (from related purposes) into what to recall. |
| [`WriteStrategy`](#raglab.memory.WriteStrategy)(\*args, \*\*kwargs)                  | Merge `new` units into a purpose's `existing` ones (ADD/UPDATE/SUPERSEDE/NOOP).  |

### *class* raglab.memory.ExtractionUnit(purpose, hit, recorded_at, importance=1.0, confidence=None, level='raw', supersedes=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

One piece of evidence a run committed to, for one purpose.

`hit` is the `SearchHit.to_dict()` of the kept hit: provenance
(`source`, `artifact_id`) and its raw magnitude, never a fused score.
`level` is the consolidation level (`"raw"` until CONSOLIDATE exists);
`supersedes` names the artifact id a later unit replaces (for SUPERSEDE).

#### *classmethod* from_dict(d)

Inverse of [`to_dict()`](#raglab.memory.ExtractionUnit.to_dict).

* **Return type:**
  [`ExtractionUnit`](#raglab.memory.ExtractionUnit)

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

### *class* raglab.memory.PurposeStore(backend=dict(), embedder=None, write_strategy=<function upsert_write>, read_strategy=<function make_relevance_recency_reader.<locals>.read>, min_relevance=0.8, clock=<built-in function time>)

Bases: [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)

`MutableMapping[purpose, list[ExtractionUnit]]` over an injected `backend`.

`backend` maps [`purpose_id()`](#raglab.memory.purpose_id) to a JSON-clean record
`{"purpose", "embedding", "units"}`: any `MutableMapping` (a dict, a dol
store of JSON files, …). `store[purpose]` recalls from every *related*
purpose (the same one, or one whose embedding is at least `min_relevance`
cosine-similar); `store[purpose] = units` replaces that purpose’s units.
Calling the store with a `RoundRecord` journals what the round kept.

#### clock()

time() -> floating-point number

Return the current time in seconds since the Epoch.
Fractions of a second may be present if the system clock provides them.

#### recall(purpose)

The units of every purpose related to `purpose`, ranked by `read_strategy`.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`ExtractionUnit`](#raglab.memory.ExtractionUnit)]

#### recall_hits(purpose)

[`recall()`](#raglab.memory.PurposeStore.recall) as `SearchHit` s, ready to seed the agent’s pool.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[`SearchHit`]

#### write(purpose, units)

Merge `units` into `purpose`’s, with the `write_strategy`.

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

#### write_strategy(new)

ADD new artifacts; UPDATE (replace with the newer unit) those already kept.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`ExtractionUnit`](#raglab.memory.ExtractionUnit)]

### *class* raglab.memory.ReadStrategy(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Rank `(relevance, unit)` candidates (from related purposes) into what to recall.

### *class* raglab.memory.WriteStrategy(\*args, \*\*kwargs)

Bases: [`Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)

Merge `new` units into a purpose’s `existing` ones (ADD/UPDATE/SUPERSEDE/NOOP).

### raglab.memory.json_clean(obj)

`obj` as plain JSON types (numpy values via `tolist`, the rest via `str`).

```pycon
>>> json_clean({"p": (1, 2), "x": {1}})
{'p': [1, 2], 'x': '{1}'}
```

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### raglab.memory.make_relevance_recency_reader(, half_life=2592000.0, alpha=1.0)

A [`ReadStrategy`](#raglab.memory.ReadStrategy) scoring `relevance**alpha * importance * recency`.

`recency` halves every `half_life` seconds. Each artifact is recalled once,
at its best score, best first.

* **Return type:**
  [`ReadStrategy`](#raglab.memory.ReadStrategy)

### raglab.memory.normalize_purpose(statement)

The canonical text of a purpose: case-folded, whitespace collapsed.

```pycon
>>> normalize_purpose("  Invoices   from 2023 ")
'invoices from 2023'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### raglab.memory.purpose_id(statement)

Content-hash id of a purpose (of its [`normalize_purpose()`](#raglab.memory.normalize_purpose) text).

```pycon
>>> purpose_id("Invoices from 2023") == purpose_id("invoices  from 2023")
True
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### raglab.memory.upsert_write(existing, new)

ADD new artifacts; UPDATE (replace with the newer unit) those already kept.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`ExtractionUnit`](#raglab.memory.ExtractionUnit)]
