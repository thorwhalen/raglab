"""Tests for the cross-run benchmark (raglab#15, item 4). Hermetic: fakes only."""

from ir import SearchHit

from raglab import PurposeStore
from raglab.bench import cross_run_benchmark


def _hits(*ids, source="s"):
    return [SearchHit(aid, "k", 1.0 - i / 10, f"text {aid}", {}, source) for i, aid in enumerate(ids)]


def _static(*ids):
    return lambda query, **kw: _hits(*ids)


VOCAB = ("invoice", "invoices", "2023", "cats")


def _bag_embedder(text):
    words = text.split()
    return [float(words.count(w)) + (w == "invoice") * words.count("invoices") for w in VOCAB]


def test_repeats_are_served_from_memory_with_nothing_lost():
    report = cross_run_benchmark({"s": _static("a", "b")}, ["q", "q", "q", "cats"])
    assert [r.sweeps_cold for r in report.rows] == [1, 1, 1, 1]
    assert [r.sweeps_warm for r in report.rows] == [1, 0, 0, 1]
    assert report.sweeps_saved == 0.5
    assert report.recall == 1.0
    assert report.extractions == {"q": 2, "cats": 2}


def test_paraphrases_share_findings_only_with_an_embedder():
    workload = ["invoices 2023", "2023 invoices"]
    plain = cross_run_benchmark({"s": _static("a")}, workload)
    embedded = cross_run_benchmark(
        {"s": _static("a")}, workload, memory_factory=lambda: PurposeStore(embedder=_bag_embedder)
    )
    assert plain.sweeps_warm == 2
    assert embedded.sweeps_warm == 1


def test_source_mutation_shows_as_lost_recall():
    # the SUPERSEDE trigger: the source changed, memory still serves the old hit
    calls = []

    def mutating(query, **kw):
        calls.append(query)
        # the first purpose is swept by both runs, then the source moves on
        return _hits("a" if len(calls) <= 2 else "z")

    report = cross_run_benchmark({"s": mutating}, ["q", "q"])
    assert [r.recall for r in report.rows] == [1.0, 0.0]
    assert report.recall == 0.5
