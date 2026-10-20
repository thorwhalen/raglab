"""Cross-run benchmark: the instrument for purpose-indexed memory (raglab#15, item 4).

The follow-ups of the memory design (CONSOLIDATE, SUPERSEDE, purpose transfer, DECAY)
are each gated on an *observed* trigger. This module is what observes: it replays a
workload of purposes through an agent twice, once cold (no memory) and once warm (with
a :class:`~raglab.memory.PurposeStore`), counting the source calls of each, and reports

* ``sweeps_cold`` / ``sweeps_warm``: retriever calls without and with memory (what
  memory saves),
* ``recall``: the share of the cold run's hits the warm run also returned (what memory
  costs in answer quality: ``1.0`` means nothing was lost),
* ``extractions``: units kept per purpose after the warm run (the CONSOLIDATE
  trigger is "this exceeds a threshold").

Cold and warm calls interleave on the same sources, purpose by purpose, so a source
that changes over time shows as lost recall: that is the SUPERSEDE trigger, but so is
plain non-determinism (an approximate index, a live API). Run the benchmark on a
deterministic snapshot to read ``recall`` as what memory alone lost.

Hermetic and offline: pass any sources; nothing here calls an LLM::

    report = cross_run_benchmark(sources, ["invoices 2023", "2023 invoices"])
    report.sweeps_saved, report.recall
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .agent import Retriever, make_search_agent
from .memory import PurposeStore, purpose_id

AgentFactory = Callable[..., Callable[[str], list]]


@dataclass(frozen=True)
class PurposeRow:
    """One purpose of the workload: its source calls cold and warm, and its recall."""

    purpose: str
    sweeps_cold: int
    sweeps_warm: int
    recall: float


@dataclass(frozen=True)
class CrossRunReport:
    """Totals over a workload, with the per-purpose ``rows`` and ``extractions``."""

    rows: tuple[PurposeRow, ...]
    extractions: Mapping[str, int] = field(default_factory=dict)

    @property
    def sweeps_cold(self) -> int:
        return sum(r.sweeps_cold for r in self.rows)

    @property
    def sweeps_warm(self) -> int:
        return sum(r.sweeps_warm for r in self.rows)

    @property
    def sweeps_saved(self) -> float:
        """Share of the cold source calls the warm run did not make."""
        return 1 - self.sweeps_warm / self.sweeps_cold if self.sweeps_cold else 0.0

    @property
    def recall(self) -> float:
        """Mean, over purposes, of the share of cold hits the warm run also returned."""
        return sum(r.recall for r in self.rows) / len(self.rows) if self.rows else 1.0


def extraction_counts(store: PurposeStore) -> dict[str, int]:
    """``{purpose: number of units kept}`` (each purpose under its own statement).

    >>> from raglab.memory import ExtractionUnit
    >>> s = PurposeStore()
    >>> s["q"] = [ExtractionUnit("q", {"artifact_id": "a"}, 0.0)]
    >>> extraction_counts(s)
    {'q': 1}
    """
    return {
        purpose: len(store._units(purpose_id(purpose)))  # its own units, not related ones
        for purpose in store
    }


class _Counting(Mapping):
    """``sources``, lazily, with each retriever call appended to ``calls``."""

    def __init__(self, sources: Mapping[str, Retriever], calls: list):
        self._sources, self._calls = sources, calls

    def __getitem__(self, name: str) -> Retriever:
        retrieve = self._sources[name]

        def counted(*args, **kwargs):
            self._calls.append(name)
            return retrieve(*args, **kwargs)

        return counted

    def __iter__(self):
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)


def _hit_key(hit: Any) -> tuple:
    return (getattr(hit, "source", None), hit.artifact_id)


def cross_run_benchmark(
    sources: Mapping[str, Retriever],
    workload: Iterable[str],
    *,
    agent_factory: AgentFactory = make_search_agent,
    memory_factory: Callable[[], PurposeStore] = PurposeStore,
    **agent_kwargs: Any,
) -> CrossRunReport:
    """Replay ``workload`` (purposes, in order, repeats and paraphrases welcome) cold
    and warm, and report what memory saved and what it lost.

    ``agent_factory(sources, memory=..., **agent_kwargs)`` builds the agent (default
    :func:`~raglab.agent.make_search_agent`); the cold agent gets ``memory=None``.
    The warm agent shares one ``memory_factory()`` store across the whole workload.
    """
    if "memory" in agent_kwargs:
        raise TypeError(
            "cross_run_benchmark sets memory itself (none cold, memory_factory() warm)"
        )
    workload = list(workload)
    cold_calls: list = []
    cold = agent_factory(_Counting(sources, cold_calls), memory=None, **agent_kwargs)
    warm_calls: list = []
    store = memory_factory()
    warm = agent_factory(_Counting(sources, warm_calls), memory=store, **agent_kwargs)

    rows = []
    for purpose in workload:
        n_cold, n_warm = len(cold_calls), len(warm_calls)
        cold_hits = {_hit_key(h) for h in cold(purpose)}
        warm_hits = {_hit_key(h) for h in warm(purpose)}
        recall = len(cold_hits & warm_hits) / len(cold_hits) if cold_hits else 1.0
        rows.append(
            PurposeRow(
                purpose,
                sweeps_cold=len(cold_calls) - n_cold,
                sweeps_warm=len(warm_calls) - n_warm,
                recall=recall,
            )
        )
    return CrossRunReport(tuple(rows), extraction_counts(store))
