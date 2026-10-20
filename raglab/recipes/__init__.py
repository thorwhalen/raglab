"""Recipes: whole workflows composed from raglab's roles, ready to call."""

from .corpus_labeling import CorpusLabeling, FacetCoverage, label_corpus

__all__ = ["CorpusLabeling", "FacetCoverage", "label_corpus"]
