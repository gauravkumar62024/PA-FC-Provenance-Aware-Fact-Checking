"""Regression: the metrics module must not shadow the stdlib `statistics`.

An earlier layout placed these functions in `src/pafc/statistics.py`, which
shadowed the standard library module for anything that added `src/pafc` to
sys.path and broke `from statistics import mean` downstream.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_module_named_statistics_in_package():
    assert not os.path.exists(os.path.join(ROOT, "src", "pafc", "statistics.py"))


def test_stdlib_statistics_still_importable_with_src_on_path():
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import pafc  # noqa: F401
    from statistics import mean
    assert mean([1, 2, 3]) == 2
