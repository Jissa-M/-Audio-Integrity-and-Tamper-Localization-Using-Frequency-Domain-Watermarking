"""Member 4 entry point. Run from repository root; see MEMBER4_README.txt."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from dataclasses import asdict
import numpy as np
from member4_config import DetectorConfig
from tamper_detector import score_windows, calibrate_threshold, detect_intervals, save_csv
from member4_evaluation import evaluate_file, summarize, union_intervals, overlap
from member4_plots import comparison, calibration_plot, summary_plot


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def watermark_api():
    import watermark
    for name in ('embed', 'extract', 'get_settings', 'save_evidence_csv'):
        if not callable(getattr(watermark, name, None)):
            raise RuntimeError('Missing watermark.' + name + ': merge current main before running real audio.')
    settings = watermark.get_settings()
    if settings['sr'] != 16000 or settings['hop'] != 256:
        raise RuntimeError('Watermark settings differ from the agreed 16 kHz / 256 hop')
    return watermark


def build(root, seed):
    import soundfile as sf
    import tamper_simulation as sim
    wm = watermark_api()
    if sim.embed is not wm.embed:
        raise RuntimeError('Simulation is using a placeholder embed function')
    manifest = []
    seen = {}
    for folder in ('sources', 'donors'):
        paths = sorted(Path('data', folder).glob('*.wav'))
        if not paths:
            raise RuntimeError(f'Missing data/{folder} WAV files: use Member 3 branch')
        for p in paths:
            info = sf.info(p)
            if info.samplerate != 16000 or info.channels != 1 or info.duration < 8:
                raise ValueError(f'{p}: need mono, 16 kHz, at least 8 seconds before simulation')
            x, _ = sf.read(p)
            if not np.isfinite(x).all() or np.max(np.abs(x)) > 1 or np.max(np.abs(x)) == 0:
                raise ValueError(f'Invalid audio samples in {p}')
            digest = sha(p)
            split = sim.SPLITS.get(p.stem, 'donor') if folder == 'sources' else 'donor'
            if digest in seen and seen[digest] != split:
                raise ValueError('Identical recordings occur in different splits/donor pool')
            seen[digest] = split
            manifest.append({'file': str(p), 'sha256': digest, 'split': split,
                             'duration': info.duration, 'sample_rate': info.samplerate})
    # Verify embedding is not a silent identity fallback, on a controlled signal.
    probe = np.random.default_rng(0).normal(0, .1, 16000)
    if np.array_equal(wm.embed(probe, sim.KEY), probe):
        raise RuntimeError('Embedding returned unchanged audio')
    with tempfile.TemporaryDirectory() as d:
        previous = os.environ.get('PATH', '')
        if not shutil.which('ffmpeg'):
            import imageio_ffmpeg
            executable = Path(d) / ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
            shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), executable)
            os.environ['PATH'] = d + os.pathsep + previous
        try:
            rows = sim.build_dataset(out_dir=str(root / 'audio'),
                                     gt_path=str(root / 'ground_truth.csv'), seed=seed)
        finally:
            os.environ['PATH'] = previous
    # Record artifacts after build; calibration refuses stale/unproven datasets.
    files = sorted({r['file_name'] for r in rows})
    dump(root / 'dataset_manifest.json', {'seed': seed, 'inputs': manifest,
         'watermark_settings': wm.get_settings(), 'watermark_sha256': sha('watermark.py'),
         'simulation_sha256': sha('tamper_simulation.py'), 'key_sha256': hashlib.sha256(sim.KEY.encode()).hexdigest(),
         'ground_truth_sha256': sha(root / 'ground_truth.csv'),
         'audio_sha256': {f: sha(root / 'audio' / f) for f in files}})
    print(f'Built {len(files)} real audio cases, {len(rows)} ground-truth rows', flush=True)


def cases(root):
    import tamper_simulation as sim
    wm = watermark_api()
    manifest = json.loads((root / 'dataset_manifest.json').read_text())
    if manifest['watermark_sha256'] != sha('watermark.py') or manifest['simulation_sha256'] != sha('tamper_simulation.py'):
        raise RuntimeError('Modules changed since build: use a new output directory and rebuild')
    if manifest['key_sha256'] != hashlib.sha256(sim.KEY.encode()).hexdigest():
        raise RuntimeError('Watermark key changed since build')
    if manifest['ground_truth_sha256'] != sha(root / 'ground_truth.csv'):
        raise RuntimeError('Ground truth changed since dataset build')
    grouped, splits = {}, {}
    with open(root / 'ground_truth.csv', newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            name = r['file_name']
            if Path(name).name != name:
                raise ValueError('Ground-truth file names must be plain basenames')
            if r['split'] not in ('training', 'validation', 'test'):
                raise ValueError('Unknown split')
            old = splits.setdefault(r['source_file'], r['split'])
            if old != r['split']: raise ValueError('Source recording appears in multiple splits')
            c = grouped.setdefault(name, {**r, 'truth': [], 'parameters': json.loads(r['parameters'] or '{}')})
            if c['split'] != r['split'] or c['source_file'] != r['source_file']:
                raise ValueError('Inconsistent duplicate ground-truth rows')
            if bool(r['tamper_start']) != bool(r['tamper_end']): raise ValueError('Incomplete interval')
            if r['tamper_start']: c['truth'].append((float(r['tamper_start']), float(r['tamper_end'])))
    for name, c in grouped.items():
        c['truth'] = union_intervals(c['truth'])
        if manifest['audio_sha256'].get(name) != sha(root / 'audio' / name):
            raise RuntimeError(f'Audio changed since build: {name}')
    return grouped, wm


def analyze(root, name, wm, cfg):
    import soundfile as sf
    import tamper_simulation as sim
    audio, sr = sf.read(root / 'audio' / name)
    if sr != cfg.sample_rate or audio.ndim != 1: raise ValueError('Audio format mismatch')
    ev = wm.extract(audio, sim.KEY)
    duration = len(audio) / sr
    windows = score_windows(ev, duration, cfg)
    folder = root / 'evidence'; folder.mkdir(exist_ok=True)
    wm.save_evidence_csv(ev, folder / (Path(name).stem + '.csv'))
    return audio, ev, windows


def calibrate(root, cfg):
    target = root / 'calibration.json'
    if target.exists(): raise RuntimeError('Calibration is already frozen. Use a new --out directory for a new experiment.')
    data, wm = cases(root)
    benign, spliced, used = [], [], []
    for name, c in data.items():
        if c['split'] != 'validation': continue
        print('Validation:', name, flush=True)
        _, _, windows = analyze(root, name, wm, cfg)
        used.append(name)
        if not c['truth']: benign.extend(windows)
        else:
            spliced.extend(w['score'] for w in windows if w['valid'] and
                           sum(overlap((w['start'], w['end']), t) for t in c['truth']) /
                           (w['end'] - w['start']) >= cfg.truth_overlap_fraction)
    threshold = calibrate_threshold(benign, cfg)
    values = [w['score'] for w in benign if w['valid']]
    record = {'threshold': threshold, 'config': asdict(cfg), 'validation_files': used,
              'method': 'Low quantile of all valid untouched/benign validation window scores',
              'valid_benign_windows': len(values), 'benign_window_alarm_rate': float(np.mean(np.array(values) < threshold)),
              'dataset_manifest_sha256': sha(root / 'dataset_manifest.json'),
              'detector_sha256': sha('tamper_detector.py'), 'watermark_settings': wm.get_settings()}
    dump(target, record)
    calibration_plot(root / 'calibration.png', values, spliced, threshold)
    print(f'Frozen threshold: {threshold:.6f}', flush=True)
    return record


def evaluate(root, plot_limit=6):
    import soundfile as sf
    frozen = json.loads((root / 'calibration.json').read_text())
    if frozen['dataset_manifest_sha256'] != sha(root / 'dataset_manifest.json') or frozen['detector_sha256'] != sha('tamper_detector.py'):
        raise RuntimeError('Dataset/detector changed after calibration; start a new experiment')
    cfg, threshold = DetectorConfig(**frozen['config']), frozen['threshold']
    data, wm = cases(root)
    results, detections, window_rows, failures = [], [], [], []
    groups, plotted = {}, set()
    for name, c in data.items():
        if c['split'] != 'test': continue
        audio, ev, windows = analyze(root, name, wm, cfg)
        if any(t[1] > len(audio) / cfg.sample_rate for t in c['truth']): raise ValueError('Truth beyond file duration')
        predictions = detect_intervals(windows, threshold)
        r = evaluate_file(windows, predictions, c['truth'], cfg.truth_overlap_fraction)
        r.update(file_name=name, condition=c['tamper_type'], source_file=c['source_file'])
        results.append(r)
        for key, val in [('condition', c['tamper_type'])] + list(c['parameters'].items()):
            if key not in ('condition', 'length_s', 'position', 'donor_type', 'target_snr_db', 'bitrate_kbps', 'gain', 'via_hz'): continue
            groups.setdefault(f'{key}={val}', []).append(r)
        for p in predictions:
            detections.append({'file_name': name, 'interval_start': p['start'], 'interval_end': p['end'],
                               'confidence': p['confidence'], 'mean_score': p['mean_score']})
        window_rows.extend({'file_name': name, **w} for w in windows)
        print(name + ': ' + (', '.join(f"[{p['start']:.3f}, {p['end']:.3f}) s" for p in predictions) or 'no suspicious intervals'), flush=True)
        if r['missed_splices'] or r['extra_intervals'] or r['undetermined_windows']:
            failures.append({'file_name': name, 'missed_splices': r['missed_splices'],
                             'extra_intervals': r['extra_intervals'], 'undetermined_windows': r['undetermined_windows'],
                             'review_note': 'Observed failure/abstention; inspect score and audio plots before assigning a cause.'})
        if len(plotted) < plot_limit and c['tamper_type'] not in plotted:
            ref, _ = sf.read(root / 'audio' / (c['source_file'] + '_untouched.wav'))
            comparison(root / 'plots' / (Path(name).stem + '.png'), name, ev, windows, predictions,
                       c['truth'], threshold, ref, audio)
            plotted.add(c['tamper_type'])
    if not results: raise RuntimeError('No test files')
    summary = summarize(results)
    breakdown = {k: summarize(v) for k, v in groups.items()}
    dump(root / 'metrics.json', {'summary': summary, 'by_condition': breakdown,
         'notes': ['Boundary errors are absolute seconds for matched intervals only.',
                   'Missed intervals contribute zero to mean_iou_including_misses.',
                   'Undetermined tampered windows count as false negatives; coverage is reported.',
                   'File detection means at least one positive-overlap one-to-one match.',
                   'Confidence is normalized threshold margin, not a calibrated probability.',
                   'Speech/music labels and recording provenance are not supplied; no invented breakdown.'],
         'per_file': results})
    save_csv(root / 'detected_intervals.csv', detections,
             ['file_name', 'interval_start', 'interval_end', 'confidence', 'mean_score'])
    save_csv(root / 'window_scores.csv', window_rows,
             ['file_name', 'start', 'end', 'score', 'valid', 'valid_frames', 'total_frames', 'suspicious'])
    save_csv(root / 'failure_review.csv', failures,
             ['file_name', 'missed_splices', 'extra_intervals', 'undetermined_windows', 'review_note'])
    save_csv(root / 'metrics_by_condition.csv', [{'group': k, **v} for k, v in breakdown.items()],
             ['group', 'files', 'precision', 'recall', 'f1', 'file_false_positive_rate', 'file_detection_rate',
              'missed_splices', 'extra_intervals', 'evidence_coverage', 'mean_iou_including_misses'])
    summary_plot(root / 'summary.png', {k.removeprefix('condition='): v for k, v in breakdown.items() if k.startswith('condition=')})
    dump(root / 'run_summary.json', {'status': 'completed', 'calibration_sha256': sha(root / 'calibration.json'),
                                   'summary': summary})
    print(json.dumps(summary, indent=2))


def synthetic(root):
    cfg = DetectorConfig()
    rng = np.random.default_rng(2026)
    t = np.arange(0, 15, cfg.hop / cfg.sample_rate)

    def make_evidence(scores):
        valid = np.ones(len(t), dtype=bool)
        valid[0] = valid[-1] = False
        scores = scores.copy()
        scores[~valid] = 0.0
        return {
            'frame_index': np.arange(len(t)),
            'time_seconds': t,
            'watermark_score': scores,
            'valid': valid,
        }

    # Independent synthetic validation evidence: intact scores.
    validation = make_evidence(
        np.clip(rng.normal(0.07, 0.08, len(t)), -1, 1)
    )
    threshold = calibrate_threshold(
        score_windows(validation, 15.0, cfg), cfg
    )

    # Separate synthetic test evidence with a known splice.
    scores = rng.normal(0.07, 0.08, len(t))
    dip = (t >= 5.0) & (t < 8.0)
    scores[dip] = rng.normal(0.0, 0.08, dip.sum())
    ev = make_evidence(np.clip(scores, -1, 1))

    windows = score_windows(ev, 15.0, cfg)
    predictions = detect_intervals(windows, threshold)

    dump(root / 'synthetic_only.json', {
        'label': 'SYNTHETIC SCORES ONLY - not audio performance',
        'threshold': threshold,
        'intervals': predictions,
        'metrics': summarize([
            evaluate_file(windows, predictions, [(5.0, 8.0)])
        ]),
    })
    comparison(
        root / 'synthetic_only.png',
        'Synthetic score test (not real audio results)',
        ev, windows, predictions, [(5.0, 8.0)], threshold,
    )
    print('Synthetic threshold:', threshold)
    print('Synthetic detected intervals:', predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['synthetic', 'build', 'calibrate', 'evaluate', 'all'])
    parser.add_argument('--out', type=Path, default=Path('results/member4'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--window', type=float, default=.5)
    parser.add_argument('--quantile', type=float, default=.01)
    parser.add_argument('--plots', type=int, default=6)
    args = parser.parse_args(); root = args.out
    root.mkdir(parents=True, exist_ok=True)
    cfg = DetectorConfig(window_seconds=args.window, benign_quantile=args.quantile)
    if args.command in ('all', 'build'):
        if (root / 'dataset_manifest.json').exists() or (root / 'calibration.json').exists():
            raise RuntimeError('Output already contains an experiment: choose a fresh --out directory')
        build(root, args.seed)
    if args.command in ('all', 'calibrate'): calibrate(root, cfg)
    if args.command in ('all', 'evaluate'): evaluate(root, args.plots)
    if args.command == 'synthetic': synthetic(root)


if __name__ == '__main__':
    main()
