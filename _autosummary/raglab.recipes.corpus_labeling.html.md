# raglab.recipes.corpus_labeling

Label a corpus against a topic, facet by facet, with a graded judge (raglab#8, part B).

[`label_corpus()`](#raglab.recipes.corpus_labeling.label_corpus) composes existing pieces and adds no orchestrator:

- a planner that emits one [`SubTask`](raglab.agent.html.md#raglab.agent.SubTask) per **facet** of the
  topic, for broad recall;
- a formulator (identity by default; [`raglab.make_llm_formulator()`](raglab.html.md#raglab.make_llm_formulator) for
  query expansion);
- the graded rubric evaluator ([`raglab.make_graded_rubric_evaluator()`](raglab.html.md#raglab.make_graded_rubric_evaluator)),
  which judges **every** retrieved candidate on its disclosed content;
- a **coverage** check per facet (enough candidates at `coverage_level`),
  which, when a facet looks thin and a `reformulate` policy is given, emits a
  refinement sub-task: the back-edge of [`SingleContextAgent`](raglab.agent.html.md#raglab.agent.SingleContextAgent).

The deliverable is a frozen [`CorpusLabeling`](#raglab.recipes.corpus_labeling.CorpusLabeling): each judged artifact’s best
[`GradedLabel`](raglab.agent.html.md#raglab.agent.GradedLabel) (its `facet` set) and each facet’s coverage.
Nothing here touches `ir`; the corpus is read through a retriever
(`ir.as_retriever()` by default).

### Functions

| [`label_corpus`](#raglab.recipes.corpus_labeling.label_corpus)(corpus, \*, facets[, retriever, ...])   | Label `corpus` for each of `facets` with a graded judge; see the module.   |
|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|

### Classes

| [`CorpusLabeling`](#raglab.recipes.corpus_labeling.CorpusLabeling)([labels, coverage, levels])   | The graded labels of a corpus for one topic, and each facet's coverage.                     |
|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| [`FacetCoverage`](#raglab.recipes.corpus_labeling.FacetCoverage)(facet, judged, covered, ...)   | How well one facet was covered: candidates judged, those at the coverage level, rounds run. |

### *class* raglab.recipes.corpus_labeling.CorpusLabeling(labels=dict(), coverage=dict(), levels=('core', 'strong', 'uses-tools', 'tangential', 'none'))

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The graded labels of a corpus for one topic, and each facet’s coverage.

#### at_least(level)

The artifact ids labelled `level` or better, on the levels they were labelled on.

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### *class* raglab.recipes.corpus_labeling.FacetCoverage(facet, judged, covered, rounds, sufficient)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

How well one facet was covered: candidates judged, those at the coverage level, rounds run.

### raglab.recipes.corpus_labeling.label_corpus(corpus, \*, facets, retriever=None, formulator=<function identity_formulator>, judge=None, reformulate=None, levels=('core', 'strong', 'uses-tools', 'tangential', 'none'), coverage_level='strong', min_covered=1, budget=None, k=50, \*\*evaluator_kwargs)

Label `corpus` for each of `facets` with a graded judge; see the module.

* **Parameters:**
  * **corpus** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – an `ir` corpus or corpus name, read through
    `ir.as_retriever(corpus, k=k)` unless `retriever` is given.
  * **facets** ([`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – the topic’s facets, as `{name: goal}` or names (a name is its
    own goal). One sub-task each.
  * **retriever** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), list[SearchHit]]]) – inject a `Retriever` instead (a test
    double, a federated source); `corpus` is then not read.
  * **formulator** ([`Formulator`](raglab.agent.html.md#raglab.agent.Formulator)) – the Formulator role (identity, or
    [`raglab.make_llm_formulator()`](raglab.html.md#raglab.make_llm_formulator)).
  * **judge** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[`...`](https://docs.python.org/3/builtins/constants.html#Ellipsis), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the graded judge (built lazily on `oa` when omitted).
  * **reformulate** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`GradedLabel`](raglab.agent.html.md#raglab.agent.GradedLabel)]], str | None] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – gap-driven reformulation; when a facet has fewer than
    `min_covered` labels at `coverage_level` it gets a new query, which
    becomes the facet’s next sub-task. `None` (default): one round each.
  * **budget** ([`Budget`](raglab.agent.html.md#raglab.agent.Budget) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – the loop bounds (`max_rounds` caps the reformulations).
  * **\*\*evaluator_kwargs** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – forwarded to
    [`make_graded_rubric_evaluator()`](raglab.judging.html.md#raglab.judging.make_graded_rubric_evaluator) (`rubric`,
    `calibration_offset`, `disclose_kwargs`, `rng`, …).
* **Return type:**
  [`CorpusLabeling`](#raglab.recipes.corpus_labeling.CorpusLabeling)
