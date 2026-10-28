"""Window, file and one-to-one interval metrics; no hidden exclusion of misses."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def union_intervals(intervals):
    result = []
    for s, e in sorted(intervals):
        if not np.isfinite(s + e) or s < 0 or e <= s:
            raise ValueError('Invalid ground-truth interval')
        if result and s <= result[-1][1]:
            result[-1][1] = max(e, result[-1][1])
        else:
            result.append([s, e])
    return result


def overlap(a, b):
    return max(0., min(a[1], b[1]) - max(a[0], b[0]))


def iou(a, b):
    inter = overlap(a, b)
    return inter / (a[1] - a[0] + b[1] - b[0] - inter)


def evaluate_file(windows, predictions, truth, fraction=0.5):
    truth = union_intervals(truth)
    result = dict(tp=0, fp=0, fn=0, tn=0, undetermined_windows=0,
                  undetermined_tampered_windows=0, total_windows=len(windows))
    for w in windows:
        target = sum(overlap((w['start'], w['end']), t) for t in truth) / (w['end'] - w['start']) >= fraction
        if not w['valid']:
            result['undetermined_windows'] += 1
            result['undetermined_tampered_windows'] += int(target)
        # Abstention is never a positive prediction. Tampered abstentions count
        # as false negatives in the headline recall, so silence cannot inflate it.
        predicted = w.get('suspicious', False) and w['valid']
        result['tp' if target and predicted else 'fn' if target else 'fp' if predicted else 'tn'] += 1
    pairs = [(p['start'], p['end']) for p in predictions]
    matrix = np.array([[iou(p, t) for t in truth] for p in pairs])
    matched = []
    if pairs and truth:
        ii, jj = linear_sum_assignment(-matrix)
        for i, j in zip(ii, jj):
            if matrix[i, j] > 0:
                matched.append({'prediction_index': int(i), 'truth_index': int(j),
                                'iou': float(matrix[i, j]),
                                'start_error': abs(pairs[i][0] - truth[j][0]),
                                'end_error': abs(pairs[i][1] - truth[j][1])})
    result.update(truth_intervals=len(truth), predicted_intervals=len(pairs),
                  missed_splices=len(truth) - len(matched),
                  extra_intervals=len(pairs) - len(matched),
                  spliced=bool(truth), flagged=bool(pairs), detected=bool(matched),
                  matches=matched)
    return result


def summarize(results):
    totals = {k: sum(r[k] for r in results) for k in
              ('tp', 'fp', 'fn', 'tn', 'undetermined_windows', 'undetermined_tampered_windows',
               'total_windows', 'truth_intervals', 'predicted_intervals', 'missed_splices', 'extra_intervals')}
    def ratio(a, b):
        return a / b if b else None
    tp, fp, fn = (totals[k] for k in ('tp', 'fp', 'fn'))
    benign = [r for r in results if not r['spliced']]
    spliced = [r for r in results if r['spliced']]
    matches = [m for r in results for m in r['matches']]
    totals.update(files=len(results), benign_files=len(benign), spliced_files=len(spliced),
                  precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                  f1=ratio(2 * tp, 2 * tp + fp + fn),
                  file_false_positive_rate=ratio(sum(r['flagged'] for r in benign), len(benign)),
                  file_detection_rate=ratio(sum(r['detected'] for r in spliced), len(spliced)),
                  window_false_positive_rate=ratio(fp, fp + totals['tn']),
                  evidence_coverage=ratio(totals['total_windows'] - totals['undetermined_windows'], totals['total_windows']),
                  mean_iou_including_misses=ratio(sum(m['iou'] for m in matches), totals['truth_intervals']))
    for key in ('iou', 'start_error', 'end_error'):
        vals = [m[key] for m in matches]
        totals[key + '_matched'] = {'mean': float(np.mean(vals)), 'median': float(np.median(vals)),
                                    'worst': float(min(vals) if key == 'iou' else max(vals))} if vals else None
    return totals
