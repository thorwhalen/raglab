# raglab — agent instructions

`raglab` is the **agentic-search / RAG orchestration layer**: it turns a
retrieval substrate into a *Search Agent* (plan → formulate → retrieve →
evaluate → re-query → rerank → cite) and, later, RAG pipelines on top.

> Fresh start (v0.2.0+). This repo took over the `raglab` PyPI name; the old
> backend lives at `raglab_bak`. Development is just beginning — greenfield.

## The architecture we're building toward

The reference design is **`ir_09 — A Composable Search Agent`**, in the **ir
repo** at `$PP/i/ir/misc/docs/ir_09 -- A Composable Search Agent ...md`, and the
cross-repo plan is **i2mint/ir epic #38**. Read both before non-trivial work.

raglab is the **orchestration layer on top of `ir`** (the retrieval substrate).
"Structure over concretion": a small set of **roles** (Protocols) wired by a
control loop, with concrete tools injected at the leaves.

### raglab owns (the *agent*)

- **Value types** (frozen, plain data, ir_09 §3): `Query`, `SubTask(goal, sources)`,
  `LowLevelQuery(source, spec)`, `Judgement(relevant, sufficient, refinement)`.
  (`Result` = ir's `SearchHit`/`Disclosure`, reused as-is — do not redefine it.)
- **Role Protocols** (open-closed strategy seams, injected callables):
  `Planner`, `Formulator`, `Retriever`, `Evaluator`, `Reranker`, `Citer`.
- **The control loop with the back-edge** (evaluator → reformulate). v1 = an
  imperative `while` loop over a mutable `AgentState` (ir_09 §9), not a
  cyclic-graph runner. This back-edge is what makes it an agent vs a DAG.
- **Budget governor** (`max_rounds` / `max_sources_per_task` / `max_results_per_task`);
  termination as a **separately injected policy** (ir_09 §9), not folded into the
  evaluator.
- **Source registry** = a live `Mapping[name, Retriever]` across *heterogeneous*
  backends (ir corpora + web/SQL/graph); cross-source merge + global rerank at
  the fan-in point.
- **Two orchestrators behind one interface** (ir_09 §7): `SingleContextAgent`
  (default, cheap, one ReAct loop) and `MultiAgentAgent` (one subagent per
  sub-task/source, ~15× cost, breadth-first only). Promotion swaps only the
  orchestrator, keeping role contracts identical.

### What raglab CONSUMES from ir (do not reimplement these)

- `ir.as_retriever(corpus)` → register an ir corpus as one `Retriever` key (#33).
- `ir.registry.retrievers()` → the ir-corpus slice of the source registry (#34).
- `ir.make_llm_formulator` / the `formulate=` seam → the Formulator role (#32).
- `ir.Selection` + its derived `sufficient` signal → Evaluator input (#35).
- `ir.disclose(..., store=...)` + `SearchHit.to_dict()` → pointer-passing / lazy
  deref across the subagent boundary (#36).

### What belongs ELSEWHERE

- **Generation / answer synthesis and the Citer/Verifier** (which needs a
  generated claim) sit with the RAG/generation layer (`srag`), not the search
  agent. The agent's deliverable is **pointers + extractions**, not an essay.

## Dependency direction (load-bearing)

**`raglab` imports `ir` (and `oa` for LLM strategies); `ir` NEVER imports
`raglab`.** Keep LLM ops (`oa`) lazy/opt-in so `import raglab` stays offline.

## Build order (ir_09 §8 / epic #38)

1. `Retriever` Protocol + source registry with 2–3 real backends (ir corpora via
   `ir.as_retriever`; one web/SQL).
2. `SingleContextAgent` with a trivial planner + pass-through evaluator wrapping
   ir's search/select/disclose — the thin slice, **no loop yet**.
3. LLM `Formulator` + `Evaluator` and **turn on the back-edge**.
4. Reranker at fan-in; Citer (in `srag`).
5. Budget governor + run-log / observability.
6. `MultiAgentAgent` — only if breadth justifies the cost.

## House style (i2mint ecosystem)

Functional > OOP; SOLID when OOP; facades, SSOT, dependency injection;
progressive disclosure; keyword-only beyond the 3rd positional; `collections.abc`
+ frozen `dataclass`es; `Protocol`s for the role seams; every module has a
top-level docstring. Never `pip install` local ecosystem packages (`ir`, `ef`,
`vd`, `dol`, `oa`, …) — they're local via `.pth`. wads CI auto-publishes on merge.
