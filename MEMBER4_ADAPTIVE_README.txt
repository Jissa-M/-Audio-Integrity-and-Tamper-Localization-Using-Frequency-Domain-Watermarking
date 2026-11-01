MEMBER 4 - EXPERIMENTAL HIGH-RECALL DETECTOR

Result: recall improved, but false alarms increased. This is an optional mode, not a replacement for the conservative default.

Metric | final baseline | adaptive candidate
precision: 0.898734 -> 0.766949
recall: 0.306034 -> 0.780172
f1: 0.456592 -> 0.773504
file_detection_rate: 0.750000 -> 0.958333
file_false_positive_rate: 0.025000 -> 0.175000
mean_iou_including_misses: 0.326262 -> 0.778391

Frozen rule: suspicious iff score < min(0.09, median(valid file window scores) - 0.20).
Original 0.5-second windows and 0.25-second step unchanged. Invalid windows remain undetermined.
Validation selection: maximum window F1 subject to <=5% benign-file false alarms, grid of 170 pairs.
Validation: recall 89.66%, precision 96.30%, F1 92.86%, false alarms 1/40.
Test: recall 78.02%, precision 76.69%, F1 77.35%, false alarms 7/40; 23/24 spliced files detected.
7 new rule tests passed. Existing 17 detector tests passed before these additive changes.
Parameters were frozen before candidate test evaluation. The same test set had been inspected historically.
This is exploratory improvement on existing recordings, not independent confirmation. Only four original test recordings.
No test-based parameter adjustments were made after seeing candidate results.
Assumes a mostly intact file. Whole-file replacement or very long edits can defeat median-relative detection.
No waveform, watermark, ground truth, validity labels or metric definitions were changed.
No branch was pushed and no teammate files were overwritten.

INSTALL
Use the matching final watermark branch (commit 5d8d531 or later with unchanged watermark).
Copy the CONTENTS of repo_files into the repo root, preserving the results folders.
This updates demo_app.py and adds adaptive_detector.py and test_adaptive_detector.py.
Install existing requirements.txt; regenerate audio if needed with python tamper_simulation.py.
Run: python -m pytest test_adaptive_detector.py tests/test_detection.py -q
Run: python -m streamlit run demo_app.py
The default remains Conservative. Choose High recall (experimental) to show the new rule.
Do not feed adaptive_calibration.json to the original main.py; its schema is intentionally different.