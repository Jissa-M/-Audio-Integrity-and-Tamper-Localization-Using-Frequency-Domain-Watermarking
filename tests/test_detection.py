import numpy as np
import pytest
from member4_config import DetectorConfig
from tamper_detector import score_windows, detect_intervals, calibrate_threshold, validate_evidence, load_evidence_csv
from member4_evaluation import evaluate_file, summarize, union_intervals


def evidence(duration=15, dip=None):
    t = np.arange(0, duration, .016)
    s = np.full(len(t), .8)
    if dip: s[(t >= dip[0]) & (t < dip[1])] = .05
    return dict(frame_index=np.arange(len(t)), time_seconds=t, watermark_score=s, valid=np.ones(len(t), bool))


def test_known_dip():
    w = score_windows(evidence(dip=(5, 8)), 15)
    p = detect_intervals(w, .4)
    assert len(p) == 1
    assert abs(p[0]['start'] - 5) <= .5
    assert abs(p[0]['end'] - 8) <= .5


def test_clean_has_no_detections():
    assert detect_intervals(score_windows(evidence(), 15), .4) == []


def test_invalid_is_not_tampered():
    ev = evidence(); ev['valid'][:] = False; ev['watermark_score'][:] = np.nan
    windows = score_windows(ev, 15)
    assert not any(w['valid'] for w in windows)
    assert detect_intervals(windows, .4) == []
    with pytest.raises(ValueError): calibrate_threshold(windows)


def test_partial_invalid_mean():
    ev = evidence(1)
    ev['valid'][::3] = False; ev['watermark_score'][::3] = -1
    w = score_windows(ev, 1)
    assert all(x['score'] == pytest.approx(.8) for x in w)


def test_time_and_tail():
    ev = evidence(1.1)
    w = score_windows(ev, 1.1)
    assert w[0]['start'] == 0
    assert w[-1]['end'] == 1.1
    assert all(x['end'] > x['start'] for x in w)
    ev['time_seconds'] += .1
    with pytest.raises(ValueError): score_windows(ev, 1.1)


def test_dropped_frame_rejected():
    ev = evidence()
    ev = {k: np.delete(v, 4) for k, v in ev.items()}
    with pytest.raises(ValueError): validate_evidence(ev)


def test_csv_false_string(tmp_path):
    p = tmp_path / 'e.csv'
    p.write_text('frame_index,time_seconds,watermark_score,valid\n0,0,0,False\n1,0.016,0.8,True\n')
    assert load_evidence_csv(p)['valid'].tolist() == [False, True]


def test_nonfinite_valid_score_rejected():
    ev = evidence(); ev['watermark_score'][2] = np.nan
    with pytest.raises(ValueError): validate_evidence(ev)


def test_calibration_quantile():
    assert calibrate_threshold([{'valid': True, 'score': x} for x in [.1, .2, .3]],
                               DetectorConfig(benign_quantile=.5)) == .2


def test_hand_computed_metrics():
    windows = [dict(start=0, end=1, valid=True, suspicious=True),
               dict(start=1, end=2, valid=True, suspicious=False),
               dict(start=2, end=3, valid=True, suspicious=True),
               dict(start=3, end=4, valid=True, suspicious=False)]
    r = evaluate_file(windows, [dict(start=0, end=1), dict(start=2, end=3)], [(0, 2)])
    m = summarize([r])
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (1, 1, 1, 1)
    assert m['precision'] == .5 and m['recall'] == .5 and m['f1'] == .5
    assert m['mean_iou_including_misses'] == .5
    assert m['extra_intervals'] == 1


def test_one_prediction_cannot_match_two_splices():
    r = evaluate_file([], [dict(start=1, end=7)], [(1, 2), (6, 7)])
    assert len(r['matches']) == 1 and r['missed_splices'] == 1


def test_miss_and_abstention_not_hidden():
    r = evaluate_file([dict(start=0, end=1, valid=False)], [], [(0, 1)])
    m = summarize([r])
    assert m['recall'] == 0 and m['evidence_coverage'] == 0
    assert m['missed_splices'] == 1 and m['mean_iou_including_misses'] == 0
    assert m['start_error_matched'] is None


def test_no_truth_metrics_are_explicit():
    r = evaluate_file([], [], [])
    m = summarize([r])
    assert m['recall'] is None and m['precision'] is None
    assert m['file_false_positive_rate'] == 0


def test_ground_truth_union():
    assert union_intervals([(1, 3), (2, 4), (6, 7)]) == [[1, 4], [6, 7]]


def test_short_and_empty_evidence():
    with pytest.raises(ValueError): score_windows(evidence(0), 0)


def test_undetermined_breaks_flagged_run():
    w = [dict(start=0, end=.5, valid=True, score=.1),
         dict(start=.25, end=.75, valid=False, score=None),
         dict(start=.5, end=1, valid=True, score=.1)]
    assert len(detect_intervals(w, .4)) == 2


def test_real_watermark_splice_integration():
    import watermark as wm
    from tamper_simulation import splice, verify_splice
    rng = np.random.default_rng(42)
    original = rng.normal(0, .1, 16000 * 10)
    host = wm.embed(original, 'integration-test')
    donor = rng.normal(0, .1, len(host))
    tampered, info = splice(host, donor, 4, 7)
    assert verify_splice(host, tampered, donor, info)
    clean_w = score_windows(wm.extract(host, 'integration-test'), 10)
    threshold = calibrate_threshold(clean_w, DetectorConfig(benign_quantile=.05))
    windows = score_windows(wm.extract(tampered, 'integration-test'), 10)
    predictions = detect_intervals(windows, threshold)
    assert any(min(p['end'], 7) > max(p['start'], 4) for p in predictions)
