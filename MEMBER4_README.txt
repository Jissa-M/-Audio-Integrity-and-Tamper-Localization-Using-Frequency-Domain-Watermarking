MEMBER 4 - ADD THESE FILES TO THE REPOSITORY

This package implements your detector, evaluation, plots and main.py.
Nothing has been pushed to GitHub. Work on your own branch.

1. COMBINE THE TEAM'S EXISTING WORK FIRST

The tested revisions were:
  main: 47d9edfda526dab1ba9d7226d1b8c32df08342a2
  member3-tamper-simulation: 1331248ea07eece942f579da4175ab8a88608509

Member 3's branch alone has an empty watermark.py and an old 22050 Hz loader.
The newer main has the actual watermark embed/extract implementation and a
16000 Hz loader. Both are needed. They merged cleanly in the local test copy.
Do not replace main's watermark.py with the empty file from the older branch.

In a terminal, from a clean working tree of your existing clone:

  git fetch origin
  git switch -c codex/member4-detection origin/main
  git merge origin/member3-tamper-simulation

If you already have uncommitted work, save it before switching branches.
If newer upstream commits cause a conflict, resolve it before copying files.
Do not run a reset or discard your own changes.

2. COPY THE FILES

Copy the CONTENTS of repo_files/ from this ZIP into the repository root,
preserving the tests/ directory. Replace the existing empty main.py,
tamper_detector.py, and tests/test_detection.py. Add the other files.
No changes to other members' Python modules or shared config.py are required.
The separate member4_config.py contains detector defaults.

3. INSTALL (tested on Windows with Python 3.12)

  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements-member4.txt

On macOS/Linux use .venv/bin/python instead. An internet connection is needed
for package installation. requirements-member4.txt pins the versions used
for this run; a different Python version may require compatible versions.
imageio-ffmpeg supplies a real FFmpeg executable automatically when ffmpeg
is not on PATH. The runner temporarily adds it only for the dataset build.

4. RUN YOUR TESTS

  .venv\Scripts\python.exe -m pytest tests/test_detection.py -q

Expected: 17 passed. The integration test uses actual embedding/extraction
on generated audio; the other tests exercise detector and metric behavior.
The combined team's available tests also passed: 31 tests, with FFmpeg on PATH.
Member 3's standalone compression tests require ffmpeg on PATH; main.py
handles that dependency automatically during dataset generation.

5. OPTIONAL QUICK DETECTOR DEMO

  .venv\Scripts\python.exe main.py synthetic --out results/member4_synthetic

Expected: an interval from 5.0 to 8.0 seconds. This uses SYNTHETIC SCORES.
It is a software test, not measured audio-tampering performance.

6. RUN THE REAL PIPELINE

  .venv\Scripts\python.exe main.py all --out results/member4

Run from the repository root. This uses Member 3's data/sources and data/donors,
Member 2's watermark.py and Member 3's tamper_simulation.py. It rebuilds the
watermarked test set, extracts real scores, calibrates on validation ONLY,
freezes the threshold, evaluates the test split, and generates plots.
It takes several minutes depending on the machine and writes hundreds of MB
of audio. Keep the console open until run_summary.json says completed.

For separate stages:
  .venv\Scripts\python.exe main.py build --out results/member4
  .venv\Scripts\python.exe main.py calibrate --out results/member4
  .venv\Scripts\python.exe main.py evaluate --out results/member4

Do not run 'build' after 'all' for the same output. Calibration cannot be
silently overwritten. For another experiment choose a fresh folder, e.g.
--out results/member4_run2. Evaluation reads frozen settings, not new CLI
window/quantile arguments. Choose those on validation BEFORE the test run.

7. OUTPUTS

  ground_truth.csv           Member 3's exact generated ground truth
  detected_intervals.csv     Your estimated intervals and confidence margins
  window_scores.csv         All test window scores and validity/predictions
  metrics.json              Overall, condition-level and per-file metrics
  metrics_by_condition.csv  Flat breakdown table for your report
  failure_review.csv        Misses, extra intervals and abstentions to review
  calibration.json          Frozen threshold, config and validation provenance
  calibration.png           Validation benign/spliced score distributions
  summary.png               File detection and false-alarm rates by condition
  plots/                    Six comparison figures on a common time axis
  evidence/                 Real extraction CSVs
  dataset_manifest.json     Seeds, source metadata, settings and SHA256 hashes
  run_summary.json          Completion record and overall metrics
  audio/                    Rebuilt WAVs (do not commit this large folder)

The ZIP's verified_results/ folder holds results from the actual test run.
These are a baseline, not a promise of high accuracy. Generated WAVs and
frame-evidence CSVs are omitted to keep the ZIP small; the runner rebuilds them.
Do not copy verified_results/ over a fresh experiment folder before running.

8. METHOD AND CONVENTIONS

- Frames are centered at frame_index * 256 / 16000 seconds. Keep invalid rows.
- Time windows start at zero, are 0.5 s long with a 0.25 s stride, and include
  a truncated last window. Boundaries are start-inclusive/end-exclusive.
- Average VALID frame scores only. Fewer than half valid => undetermined.
- Threshold = 1st percentile of all valid untouched/benign validation windows.
  This targets approximately 1% validation WINDOW alarms; it does NOT promise
  a 1% file false-alarm rate. Every input file uses the same frozen threshold.
- Flag valid scores strictly below threshold. Merge consecutive flagged windows.
  Undetermined or unflagged windows break runs. No tuned cleanup/gap filling.
- Report first flagged window start through last flagged window end. These
  are estimates, limited by window size and stride; no exact-boundary claim.
- Confidence = clip((threshold - mean window score)/2, 0, 1). This is a
  monotonic margin, NOT a calibrated probability or percentage certainty.
- A ground-truth window has at least 50% temporal overlap with the UNION of
  true intervals. Multiple CSV rows for one file are grouped before evaluation.
- Invalid/undetermined windows never become positive predictions. A tampered
  abstention counts as a false negative in headline recall; evidence coverage
  and tampered abstention counts are also reported. Clean abstentions count
  as negatives, so inspect coverage alongside false-positive rates.
- Interval matching maximizes total IoU one-to-one, keeping positive-overlap
  pairs only. Extra detections and missed splices are counted separately.
  Mean IoU including misses gives each missed truth interval zero.
- Boundary errors are ABSOLUTE seconds for MATCHED intervals only. Report
  missed counts beside these errors. Undefined metrics are JSON null.
- Condition breakdowns include available length, position, donor, noise,
  bitrate, gain and resampling parameters. Speech/music and original recording
  provenance are unavailable in the supplied metadata: ask Member 3 to add
  them rather than invent labels or claim recording-independent validation.
- Original/reference plot panels use the untouched WATERMARKED reference,
  explicitly labelled; detector input is only the analyzed audio, never the
  original. Ground truth is used for evaluation, not to make predictions.

9. WHAT TO COMMIT

Commit the new/replaced code, tests, dependency list, instructions, and small
result CSV/JSON/PNG artifacts that you have reviewed. Add these ignore patterns
to your existing .gitignore (append; do not replace Member 3's entries):

  .venv/
  .pytest_cache/
  results/member4*/audio/
  results/member4*/evidence/

Example staging list (then inspect git diff --cached before committing):
  git add main.py tamper_detector.py member4_config.py member4_evaluation.py member4_plots.py requirements-member4.txt tests/test_detection.py MEMBER4_README.txt

Use your team's normal PR process. Nothing in this package automatically
commits, merges remotely, or pushes anything.

10. REPORT / DEMO

Show one untouched, one spliced, and one benign example. Explain the real
metrics and failure cases in VERIFIED_RUN.txt. Do not claim general deepfake
detection or compression robustness beyond tested cases. If you change the
watermark after seeing test results, those results become development feedback;
use genuinely new held-out recordings for any fresh final performance claim.
