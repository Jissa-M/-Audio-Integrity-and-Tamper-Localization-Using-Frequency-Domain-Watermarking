import csv
from pathlib import Path
import numpy as np
from member4_config import DetectorConfig


def validate_evidence(evidence, config=DetectorConfig()):
    required = ('frame_index', 'time_seconds', 'watermark_score', 'valid')
    if any(k not in evidence for k in required):
        raise ValueError('Evidence requires ' + ', '.join(required))
    arrays = {k: np.asarray(evidence[k]) for k in required}
    n = len(arrays['frame_index'])
    if n < 2 or any(a.ndim != 1 or len(a) != n for a in arrays.values()):
        raise ValueError('Evidence must contain equal-length 1-D arrays and >=2 frames')
    if not np.array_equal(arrays['frame_index'], np.arange(n)):
        raise ValueError('Keep every frame, starting at zero; do not drop invalid rows')
    times = arrays['time_seconds'].astype(float)
    if not np.allclose(times, np.arange(n) * config.hop / config.sample_rate,
                       atol=1e-6, rtol=0):
        raise ValueError('Frame times disagree with shared hop/sample-rate convention')
    flags = arrays['valid']
    if flags.dtype.kind != 'b':
        values = [str(v).strip().lower() for v in flags]
        if any(v not in ('true', 'false', 'yes', 'no', '1', '0') for v in values):
            raise ValueError('Invalid validity flag')
        flags = np.array([v in ('true', 'yes', '1') for v in values])
    scores = arrays['watermark_score'].astype(float)
    if np.any(~np.isfinite(scores[flags])) or np.any(np.abs(scores[flags]) > 1):
        raise ValueError('Valid scores must be finite and in [-1, 1]')
    return {'frame_index': np.arange(n), 'time_seconds': times,
            'watermark_score': scores, 'valid': flags}


def load_evidence_csv(path, config=DetectorConfig()):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Evidence file is empty')
    ev = {k: [r[k] for r in rows] for k in
          ('frame_index', 'time_seconds', 'watermark_score', 'valid')}
    ev['frame_index'] = np.asarray(ev['frame_index'], dtype=int)
    return validate_evidence(ev, config)


def score_windows(evidence, duration, config=DetectorConfig()):
    ev = validate_evidence(evidence, config)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('A positive actual audio duration is required')
    if abs(ev['time_seconds'][-1] - duration) > config.hop / config.sample_rate + 1e-6:
        raise ValueError('Evidence timeline does not match audio duration')
    step = config.window_seconds * (1 - config.overlap)
    windows = []
    # Half-open time windows, anchored to sample zero; include a truncated tail.
    for start in np.arange(0, duration, step):
        end = min(float(start + config.window_seconds), duration)
        mask = (ev['time_seconds'] >= start) & (ev['time_seconds'] < end)
        valid = mask & ev['valid']
        count, nvalid = int(mask.sum()), int(valid.sum())
        determined = count > 0 and nvalid >= config.min_valid_fraction * count
        windows.append({'start': float(start), 'end': end,
                        'score': float(ev['watermark_score'][valid].mean()) if determined else None,
                        'valid': bool(determined), 'valid_frames': nvalid,
                        'total_frames': count})
        if end >= duration:
            break
    return windows


def calibrate_threshold(benign_windows, config=DetectorConfig()):
    scores = [w['score'] for w in benign_windows if w['valid']]
    if not scores:
        raise ValueError('No valid untouched/benign validation windows for calibration')
    return float(np.quantile(scores, config.benign_quantile))


def detect_intervals(windows, threshold):
    if not np.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('threshold must be finite and in [-1, 1]')
    intervals, current, scores = [], None, []
    def finish():
        if current is not None:
            mean = float(np.mean(scores))
            intervals.append({**current, 'mean_score': mean,
                              'confidence': float(np.clip((threshold - mean) / 2, 0, 1))})
    for w in windows:
        w['suspicious'] = bool(w['valid'] and w['score'] < threshold)
        if w['suspicious']:
            if current is None:
                current = {'start': w['start'], 'end': w['end']}
                scores = []
            else:
                current['end'] = w['end']
            scores.append(w['score'])
        else:
            finish()
            current = None
            scores = []
    finish()
    return intervals


def save_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

