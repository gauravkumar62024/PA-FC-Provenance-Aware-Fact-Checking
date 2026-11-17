"""PA-FC — Provenance-Aware Fact Checking.

Metrics live here; the F1-F4 feature implementations are vendored under
`pa_fc/` (the package name the stage scripts import).
"""
from .evaluation import macro_f1, weighted_f1, accuracy, per_class  # noqa: F401

__all__ = ["macro_f1", "weighted_f1", "accuracy", "per_class"]
