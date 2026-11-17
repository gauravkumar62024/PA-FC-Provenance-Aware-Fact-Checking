"""Metrics for the primary results table.

Named `evaluation` rather than `statistics` on purpose: a module called
`statistics.py` on sys.path shadows the Python standard library module of the
same name and breaks `from statistics import mean` anywhere downstream.
"""
from __future__ import annotations


def macro_f1(pairs):
    """Macro-F1 over (gold, pred) pairs.

    Labels that never appear in gold are skipped: a label the model emits but
    which is not a real class must not deflate the macro average.
    """
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    total, n = 0.0, 0
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        total += 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        n += 1
    return total / n if n else 0.0


def weighted_f1(pairs):
    labels = sorted({g for g, _ in pairs})
    tot, acc = 0, 0.0
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        sup = tp + fn
        if sup == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / sup
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc += f1 * sup
        tot += sup
    return acc / tot if tot else 0.0


def accuracy(pairs):
    return sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0


def per_class(pairs):
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    out = {}
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn)
        out[lab] = dict(precision=prec, recall=rec,
                        f1=2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
                        support=tp + fn)
    return out
