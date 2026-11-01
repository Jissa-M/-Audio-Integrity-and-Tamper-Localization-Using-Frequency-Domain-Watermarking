"""Validation-tuned local-splice detector; keeps the original window grid."""
import numpy as np
from tamper_detector import detect_intervals


def detect_adaptive(windows, absolute_cap, minimum_drop):
    """Return (intervals, diagnostics), adding suspicious to input windows.

    A window must fall below both absolute_cap and the file median minus
    minimum_drop. Only valid windows define the median. Parameters must be
    selected on validation data. Confidence remains a margin, not probability.
    Assumes most of the file is intact; not suitable for whole-file replacement.
    """
    if not np.isfinite(absolute_cap) or not -1 <= absolute_cap <= 1:
        raise ValueError("absolute_cap must be finite and in [-1, 1]")
    if not np.isfinite(minimum_drop) or not 0 < minimum_drop <= 2:
        raise ValueError("minimum_drop must be finite and in (0, 2]")
    scores = [w['score'] for w in windows if w['valid']]
    if not scores:
        for w in windows: w['suspicious'] = False
        return [], {'status': 'undetermined', 'median_score': None, 'threshold': None}
    median = float(np.median(scores))
    threshold = max(-1.0, min(float(absolute_cap), median - float(minimum_drop)))
    intervals = detect_intervals(windows, threshold)
    return intervals, {'status': 'assessed', 'median_score': median, 'threshold': threshold}
