"""Purpose-indexed memory: what past runs found, kept per purpose (raglab#6, stage 0).

The retrieval index is purpose-free by design; the search agent is the one layer that
sees *purpose* (the sub-task goal, the refinement chain, the evaluator's verdicts).
Without memory, everything a run learned dies when it returns. A :class:`PurposeStore`
keeps the precipitate: the hits each run committed to, as :class:`ExtractionUnit` s
keyed by purpose, carrying their provenance (``(source, artifact_id)`` plus the whole
``SearchHit.to_dict()``, raw magnitudes only).

It is a :class:`~raglab.agent.RunLog` sink (writes come from the run-log's
:class:`~raglab.agent.RoundRecord` s, the one shared record shape) and a
:class:`~raglab.agent.Memory` the agent reads from before sweeping::

    store = PurposeStore()
    agent = make_search_agent(sources, memory=store)
    agent("invoices from 2023")  # sweeps, journals what it kept
    agent("invoices from 2023")  # recalls those hits; sweeps only if not sufficient

Progressive disclosure: ``store[purpose]`` is the related-purpose read. With no
``embedder`` only the same purpose (normalized text) is related; inject one
(``str -> vector``, e.g. a vd/ir embedder) and similar purposes share their findings.

The four verbs of the design (report 14) are injected strategies; stage 0 ships two
(``write_strategy``: ADD, or UPDATE an artifact already kept; ``read_strategy``:
relevance x importance x recency). CONSOLIDATE and DECAY are follow-ups, gated on
observed need.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from .agent import Result, RoundRecord

Embedder = Callable[[str], Sequence[float]]

DFLT_HALF_LIFE_SECONDS = 30 * 24 * 3600.0
DFLT_MIN_RELEVANCE = 0.8


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):  # numpy scalars and arrays
        return obj.tolist()
    return str(obj)


def json_clean(obj: Any) -> Any:
    """``obj`` as plain JSON types (numpy values via ``tolist``, the rest via ``str``).

    >>> json_clean({"p": (1, 2), "x": {1}})
    {'p': [1, 2], 'x': '{1}'}
    """
    return json.loads(json.dumps(obj, default=_json_default))


def normalize_purpose(statement: str) -> str:
    """The canonical text of a purpose: case-folded, whitespace collapsed.

    >>> normalize_purpose("  Invoices   from 2023 ")
    'invoices from 2023'
    """
    return " ".join(statement.casefold().split())


def purpose_id(statement: str) -> str:
    """Content-hash id of a purpose (of its :func:`normalize_purpose` text).

    >>> purpose_id("Invoices from 2023") == purpose_id("invoices  from 2023")
    True
    """
    return hashlib.sha256(normalize_purpose(statement).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ExtractionUnit:
    """One piece of evidence a run committed to, for one purpose.

    ``hit`` is the ``SearchHit.to_dict()`` of the kept hit: provenance
    (``source``, ``artifact_id``) and its raw magnitude, never a fused score.
    ``level`` is the consolidation level (``"raw"`` until CONSOLIDATE exists);
    ``supersedes`` names the artifact id a later unit replaces (for SUPERSEDE).
    """

    purpose: str
    hit: Mapping[str, Any]
    recorded_at: float
    importance: float = 1.0
    confidence: float | None = None
    level: str = "raw"
    supersedes: str | None = None

    @property
    def identity(self) -> tuple[str | None, str]:
        """``(source, artifact_id)``: ir's identity of the artifact this points at."""
        return self.hit.get("source"), self.hit["artifact_id"]

    def to_result(self) -> Result:
        """The ``SearchHit`` this unit was made from."""
        return Result(**self.hit)

    def to_dict(self) -> dict:
        """JSON-clean form."""
        return {**asdict(self), "hit": dict(self.hit)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ExtractionUnit:
        """Inverse of :meth:`to_dict`."""
        return cls(**d)


@runtime_checkable
class WriteStrategy(Protocol):
    """Merge ``new`` units into a purpose's ``existing`` ones (ADD/UPDATE/SUPERSEDE/NOOP)."""

    def __call__(
        self, existing: Sequence[ExtractionUnit], new: Sequence[ExtractionUnit]
    ) -> list[ExtractionUnit]: ...


@runtime_checkable
class ReadStrategy(Protocol):
    """Rank ``(relevance, unit)`` candidates (from related purposes) into what to recall."""

    def __call__(
        self, candidates: Sequence[tuple[float, ExtractionUnit]], *, now: float
    ) -> list[ExtractionUnit]: ...


def upsert_write(
    existing: Sequence[ExtractionUnit], new: Sequence[ExtractionUnit]
) -> list[ExtractionUnit]:
    """ADD new artifacts; UPDATE (replace with the newer unit) those already kept."""
    by_identity = {u.identity: u for u in existing}
    for unit in new:
        by_identity[unit.identity] = unit
    return list(by_identity.values())


def make_relevance_recency_reader(
    *, half_life: float = DFLT_HALF_LIFE_SECONDS, alpha: float = 1.0
) -> ReadStrategy:
    """A :class:`ReadStrategy` scoring ``relevance**alpha * importance * recency``.

    ``recency`` halves every ``half_life`` seconds. Each artifact is recalled once,
    at its best score, best first.
    """

    def read(
        candidates: Sequence[tuple[float, ExtractionUnit]], *, now: float
    ) -> list[ExtractionUnit]:
        best: dict[tuple, tuple[float, ExtractionUnit]] = {}
        for relevance, unit in candidates:
            age = max(0.0, now - unit.recorded_at)
            score = relevance**alpha * unit.importance * 0.5 ** (age / half_life)
            if unit.identity not in best or score > best[unit.identity][0]:
                best[unit.identity] = (score, unit)
        ranked = sorted(best.values(), key=lambda su: su[0], reverse=True)
        return [unit for _, unit in ranked]

    return read


relevance_recency_reader = make_relevance_recency_reader()


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(x * x for x in v))
    return sum(x * y for x, y in zip(u, v, strict=True)) / norm if norm else 0.0


@dataclass
class PurposeStore(MutableMapping):
    """``MutableMapping[purpose, list[ExtractionUnit]]`` over an injected ``backend``.

    ``backend`` maps :func:`purpose_id` to a JSON-clean record
    ``{"purpose", "embedding", "units"}``: any ``MutableMapping`` (a dict, a dol
    store of JSON files, ...). ``store[purpose]`` recalls from every *related*
    purpose (the same one, or one whose embedding is at least ``min_relevance``
    cosine-similar); ``store[purpose] = units`` replaces that purpose's units.
    Calling the store with a :class:`RoundRecord` journals what the round kept.
    """

    backend: MutableMapping[str, dict] = field(default_factory=dict)
    embedder: Embedder | None = None
    write_strategy: WriteStrategy = upsert_write
    read_strategy: ReadStrategy = relevance_recency_reader
    min_relevance: float = DFLT_MIN_RELEVANCE
    clock: Callable[[], float] = time.time

    # --- the verbs --------------------------------------------------------- #

    def write(self, purpose: str, units: Sequence[ExtractionUnit]) -> None:
        """Merge ``units`` into ``purpose``'s, with the ``write_strategy``."""
        existing = self._units(purpose_id(purpose))
        self._put(purpose, self.write_strategy(existing, list(units)))

    def recall(self, purpose: str) -> list[ExtractionUnit]:
        """The units of every purpose related to ``purpose``, ranked by ``read_strategy``."""
        candidates = [
            (relevance, unit)
            for relevance, pid in self._related(purpose)
            for unit in self._units(pid)
        ]
        return self.read_strategy(candidates, now=self.clock())

    def recall_hits(self, purpose: str) -> list[Result]:
        """:meth:`recall` as ``SearchHit`` s, ready to seed the agent's pool."""
        return [unit.to_result() for unit in self.recall(purpose)]

    def __call__(self, record: RoundRecord) -> None:
        """Journal a round's committed hits under its purpose (the RunLog sink).

        The recall round (``round < 0``) is not journaled: its hits are already
        here, and re-writing them would reset their recency and copy a related
        purpose's units into this one.
        """
        if record.round >= 0 and record.committed:
            now = self.clock()
            units = [
                ExtractionUnit(
                    purpose=record.purpose, hit=json_clean(h.to_dict()), recorded_at=now
                )
                for h in record.committed
            ]
            self.write(record.purpose, units)

    # --- MutableMapping ---------------------------------------------------- #

    def __getitem__(self, purpose: str) -> list[ExtractionUnit]:
        """The related-purpose read; raises ``KeyError`` only if it finds nothing and
        ``purpose`` itself is not stored (so ``store[p] = []; store[p] == []``)."""
        units = self.recall(purpose)
        if units or purpose in self:
            return units
        raise KeyError(purpose)

    def __contains__(self, purpose: object) -> bool:
        """Exact: is this purpose (normalized) stored? Unlike ``store[purpose]``,
        related purposes do not count, so ``in``, ``iter`` and ``len`` agree."""
        return isinstance(purpose, str) and purpose_id(purpose) in self.backend

    def __setitem__(self, purpose: str, units: Sequence[ExtractionUnit]) -> None:
        self._put(purpose, list(units))

    def __delitem__(self, purpose: str) -> None:
        if purpose not in self:
            raise KeyError(purpose)
        del self.backend[purpose_id(purpose)]

    def __iter__(self) -> Iterator[str]:
        return (rec["purpose"] for rec in self.backend.values())

    def __len__(self) -> int:
        return len(self.backend)

    # --- internals --------------------------------------------------------- #

    def _units(self, pid: str) -> list[ExtractionUnit]:
        rec = self.backend.get(pid)
        return [ExtractionUnit.from_dict(u) for u in rec["units"]] if rec else []

    def _put(self, purpose: str, units: list[ExtractionUnit]) -> None:
        pid = purpose_id(purpose)
        rec = self.backend.get(pid)
        if rec is not None and rec.get("embedding") is not None:
            embedding = rec["embedding"]
        else:
            embedding = self._embed(purpose)
        self.backend[pid] = {
            "purpose": normalize_purpose(purpose),
            "embedding": embedding,
            "units": [u.to_dict() for u in units],
        }

    def _embed(self, purpose: str) -> list[float] | None:
        if self.embedder is None:
            return None
        return [float(x) for x in self.embedder(normalize_purpose(purpose))]

    def _related(self, purpose: str) -> list[tuple[float, str]]:
        pid = purpose_id(purpose)
        related = [(1.0, pid)] if pid in self.backend else []
        query = self._embed(purpose)
        if query is None:
            return related
        for other, rec in list(self.backend.items()):
            if other == pid:
                continue
            if rec.get("embedding") is None:  # stored before there was an embedder
                rec = {**rec, "embedding": self._embed(rec["purpose"])}
                self.backend[other] = rec
            relevance = _cosine(query, rec["embedding"])
            if relevance >= self.min_relevance:
                related.append((relevance, other))
        return related

