"""Metrics used by the primary results table."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from pafc import macro_f1, weighted_f1, accuracy


def test_perfect_predictions():
    pairs = [("A", "A")] * 5 + [("B", "B")] * 5
    assert macro_f1(pairs) == 1.0 and accuracy(pairs) == 1.0


def test_accuracy_matches_hand_count():
    assert accuracy([("A", "A"), ("A", "B"), ("B", "B"), ("B", "B")]) == 0.75


def test_macro_f1_ignores_labels_absent_from_gold():
    """A label the model invents must not be counted as a class and deflate
    the macro average; it should only cost recall on the true class."""
    pairs = [("A", "A")] * 10 + [("B", "B")] * 10
    assert macro_f1(pairs) == 1.0
    assert macro_f1(pairs + [("A", "ZZZ")]) < 1.0


def test_macro_vs_weighted_differ_on_imbalance():
    pairs = [("A", "A")] * 90 + [("B", "A")] * 10      # minority class never predicted
    assert weighted_f1(pairs) > macro_f1(pairs)


def test_empty_input_is_zero_not_crash():
    assert accuracy([]) == 0.0
