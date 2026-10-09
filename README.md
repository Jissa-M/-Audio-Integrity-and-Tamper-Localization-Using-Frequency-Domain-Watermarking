# Audio Integrity & Tamper Localization Using Frequency-Domain Watermarking

## Overview

This project explores **audio integrity verification and temporal localization of audio splicing** using a frequency-domain watermark and signal-processing techniques.

A watermark is embedded into an audio recording. Later, the system checks whether the expected watermark can still be recovered throughout the recording. If a section is replaced with audio from another source, the watermark evidence in that section is expected to weaken or become inconsistent. A sliding-window detector uses this evidence to identify suspicious regions and estimate their start and end times.

The project focuses on **controlled audio-splicing detection**, rather than claiming to detect deepfakes or every possible form of audio manipulation.

> **Implementation scope:** The planned implementation uses classical digital signal processing (DSP), including the Short-Time Fourier Transform (STFT), frequency-domain watermark embedding and extraction, normalized correlation, and sliding-window analysis. It is not a deep-learning implementation unless a learned component is added and evaluated separately.

## Objectives

- Standardize input recordings so that all modules use consistent audio settings.
- Represent audio in the time-frequency domain using the STFT.
- Embed a reproducible, key-dependent watermark in selected frequency bins.
- Extract a time-indexed watermark-matching score without requiring the original audio during detection.
- Create controlled test recordings, including untouched, benignly processed, and spliced audio.
- Detect suspicious regions and estimate their temporal boundaries.
- Compare predicted intervals against known ground-truth timestamps.
- Measure detection performance, localization error, and false alarms.
- Visualize waveforms, spectrograms, watermark scores, thresholds, and detected intervals.
- Keep experiments reproducible through recorded parameters and random seeds.

## How It Works

The pipeline has eight main stages:

1. **Audio preprocessing:** Load WAV files, convert stereo audio to mono, resample to a common sample rate, and check duration and amplitude.
2. **STFT analysis:** Convert the waveform into a time-frequency representation.
3. **Watermark embedding:** Modify selected STFT magnitudes according to a reproducible pseudo-random pattern while preserving phase.
4. **Test-data generation:** Create untouched recordings, benignly processed versions, and recordings with known audio-splicing intervals.
5. **Watermark extraction:** Re-analyze each recording and calculate how closely its spectral evidence matches the expected watermark.
6. **Sliding-window detection:** Average valid frame scores in overlapping windows and flag windows with scores below a calibrated threshold.
7. **Temporal localization:** Merge consecutive suspicious windows into intervals and convert their indices into timestamps.
8. **Evaluation and visualization:** Compare detected intervals with ground truth, calculate metrics, and generate plots.

## Technical Approach

### 1. Audio preprocessing

All recordings are standardized before further processing.

- **Format:** WAV
- **Channels:** Mono; stereo channels are averaged
- **Sample rate:** 16,000 Hz
- **Data type:** Floating point, with amplitude in the range `[-1, 1]`
- **Normalization:** Do not normalize by default. Scale down only when the peak amplitude exceeds 1, to prevent clipping.
- Empty, silent, or too-short recordings should be rejected or handled explicitly. The project plan proposes rejecting recordings shorter than 5 seconds.
- Record the original sample rate, processed sample rate, duration, sample count, and peak amplitude.

### 2. Short-Time Fourier Transform (STFT)

The STFT represents how frequency content changes over time. Instead of analyzing the entire recording as one block, it divides the signal into overlapping frames and computes a frequency representation for each frame.

The agreed starting parameters are:

| Parameter | Starting value |
|---|---:|
| Sample rate | 16,000 Hz |
| Window | Hann |
| Window length | 1,024 samples (64 ms) |
| Hop length | 256 samples (16 ms) |
| Overlap | 75% |
| Watermark band | Approximately 1.5–6 kHz, configurable |
| Detection window | Approximately 0.5 seconds, configurable |
| Detection-window overlap | 50% |

At 16 kHz, a 1,024-sample window gives a nominal frequency-bin spacing of approximately 15.625 Hz. A hop of 256 samples corresponds to 16 ms between adjacent frame positions.

The Hann window and overlapping frames support signal reconstruction. The implementation should verify the inverse-STFT round trip and preserve the input signal's length.

**Timing convention:** Times are measured in seconds from sample 0 of the processed file. Report suspicious intervals as estimates, not exact boundaries: STFT framing and overlapping detection windows limit temporal precision.

### 3. Watermark generation and embedding

The watermark is a reproducible pseudo-random pattern containing positive and negative values. It is generated using a secret key and the frame index, so each frame has its own pattern.

The design follows these principles:

- Select pseudo-random frequency bins in a configurable mid-frequency band, initially around 1.5–6 kHz.
- Start with approximately 100–200 selected bins per frame, then tune experimentally.
- Slightly increase or decrease the magnitude of selected bins according to the pattern.
- Keep the phase unchanged during magnitude modification.
- Tune the embedding strength to balance recoverability and audio quality.
- Skip frames below a documented energy threshold because very quiet frames cannot reliably carry the watermark.
- Reconstruct the watermarked waveform using the shared inverse-STFT implementation.
- Store the settings needed to reproduce extraction, including the key identifier, frequency band, number of bins, embedding strength, and energy floor.

The full embedding–reconstruction–reanalysis–extraction path must be tested. The reconstructed waveform's STFT will not necessarily match the initially modified STFT matrix exactly because overlapping frames interact during reconstruction.

### 4. Watermark extraction

Extraction is designed to be **blind**: it does not require the original, unwatermarked recording.

For each frame, the system:

1. Computes the STFT using the shared configuration.
2. Selects the bins used by the watermark.
3. Calculates the log-magnitude spectrum.
4. Removes a smoothed spectral envelope to reduce the influence of the host audio's broad frequency shape.
5. Compares the residual spectral pattern with the expected key- and frame-dependent watermark pattern.
6. Produces a normalized correlation score in the range `[-1, 1]`.
7. Marks frames as valid or invalid.

A higher positive score indicates stronger agreement with the expected watermark. Unwatermarked or wrong-key audio is expected to score near zero, while correctly watermarked audio should score higher.

Quiet frames, edge frames, and frames with insufficient residual energy must be marked **invalid**, rather than automatically classified as tampered.

The extraction output should preserve one row per STFT frame, including invalid frames, so that the timeline remains continuous.

Suggested evidence columns:

| Column | Meaning |
|---|---|
| `frame_index` | STFT frame index |
| `time_seconds` | Frame time in seconds |
| `watermark_score` | Normalized watermark-matching score |
| `valid` | Whether the frame provides usable evidence |

### 5. Tampering simulation and dataset preparation

The main simulated attack is **audio splicing**: replacing a known interval of a watermarked recording with a segment from another recording.

The test set should contain:

- **Untouched audio:** Watermarked recordings with no subsequent modification.
- **Benignly processed audio:** Examples with noise, compression, mild volume changes, or resampling.
- **Spliced audio:** Watermarked recordings with a known interval replaced by another segment.
- **Wrong-key replacements:** Where available, replacement segments watermarked using a different key.

The plan proposes collecting 10–20 openly licensed recordings, each approximately 10–30 seconds long, covering speech and some music. Record each source and its licence in `data/README.md`. Avoid committing large audio files or private recordings directly to the repository.

Separate recordings into training, validation, and test groups. A suggested split is approximately 30% / 30% / 40%, respectively. **Do not place clips from the same source recording in more than one split.**

For reproducible splicing:

- Choose splice intervals in samples and use a consistent rounding rule.
- Replace the interval with a donor segment of the same length to preserve duration and frame alignment.
- Use a short crossfade at splice boundaries to reduce audible clicks.
- Keep the splice away from the first and last second of the recording.
- Record the true start and end times, donor type, source, random seed, and transformation parameters.
- Verify that the output length is unchanged and that the replacement is at the intended location.

The planned splice lengths include approximately 0.5, 1, 2, 3, and 5 seconds, with different positions and donor types. These are experimental targets, not a claim that every case has already been generated.

Benign-processing tests may include additive noise at selected SNR levels, MP3/AAC compression at selected bitrates, mild volume changes, and resampling to another rate and back. Record the actual settings used. If a real encoder is unavailable, the plan allows low-pass filtering plus coarse quantization as a simulated compression alternative; label it clearly as simulated rather than actual MP3/AAC compression.

### 6. Sliding-window detection

The detector receives the time-indexed watermark evidence from the extraction module.

- Slide a window of approximately 0.5 seconds over the frame scores, with about 50% overlap.
- Calculate the mean of the valid frame scores in each window.
- Mark windows with too few valid frames as **undetermined**, not suspicious.
- Compare valid window scores against a threshold calibrated using validation recordings.
- Flag a window as suspicious when its score is below the threshold.
- If validation experiments support it, remove isolated one-window detections and fill very short gaps.
- Merge consecutive suspicious windows into intervals.
- Convert interval boundaries to seconds using the shared time-mapping functions.

The general decision rule is:

`Flag window i if s_i < τ`

where `s_i` is the watermark-matching score for window `i`, and `τ` is the calibrated threshold.

A low score indicates a mismatch with the expected watermark. On its own, it does **not** prove that the audio was maliciously altered.

The threshold must be selected using validation data and frozen before final test-set evaluation. Invalid or undetermined evidence must not be silently treated as low scores.

### 7. Evaluation

Evaluation should measure both whether tampering is detected and whether its location is estimated accurately.

**Window-level metrics**
- Precision
- Recall
- F1-score

**File-level metrics**
- False-positive rate on untouched and benignly processed files
- Fraction of spliced files in which at least one overlapping interval is detected

**Interval-level metrics**
- Intersection over Union (IoU) between predicted and ground-truth intervals
- Start-boundary error in seconds
- End-boundary error in seconds
- Mean, median, and worst-case localization errors
- Counts of missed splices and spurious extra intervals

Where the dataset permits, break down results by splice length, splice position, donor type, benign-processing type and level, and speech versus music.

Do not tune the final threshold on the test set. Keep validation recordings separate from final test recordings.

### 8. Visualization

The planned figures include:

- Original and spliced waveforms on a shared time axis
- Original and tampered spectrograms
- Frame-level and window-level watermark scores
- The detection threshold plotted alongside the scores
- Ground-truth and detected intervals shown as shaded regions
- Summary plots comparing metrics across test conditions

These visualizations help reveal whether score drops align with actual splice intervals and whether benign processing creates false alarms.

## Repository Structure

The planned repository structure is:

```text
audio-tamper-localization/
├── audio_processing.py
├── watermark.py
├── tamper_simulation.py
├── tamper_detector.py
├── config.py
├── main.py
├── tests/
│   ├── test_audio_processing.py
│   ├── test_watermark.py
│   └── test_detection.py
├── data/
│   └── README.md
├── results/
│   ├── ground_truth.csv
│   ├── detected_intervals.csv
│   └── plots/
├── requirements.txt
└── README.md
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `audio_processing.py` | Audio loading, preprocessing, STFT/inverse STFT, time mapping, and plotting helpers |
| `watermark.py` | Signature generation, watermark embedding and extraction, quality metrics |
| `tamper_simulation.py` | Splicing, benign transformations, ground-truth generation, and verification |
| `tamper_detector.py` | Sliding-window scoring, thresholding, interval merging, and localization |
| `config.py` | Shared audio parameters and agreed configuration values |
| `main.py` | Runs the end-to-end pipeline |
| `tests/` | Unit and integration checks for the modules |
| `data/README.md` | Dataset sources, licences, and instructions for preparing or obtaining audio |
| `results/ground_truth.csv` | Ground-truth tampering intervals and test parameters |
| `results/detected_intervals.csv` | Predicted suspicious intervals and confidence values |
| `results/plots/` | Generated figures |

The project plan assigns the initial `config.py` and README/repository hygiene to Member 1, and `main.py` to Member 4. Shared configuration changes should be agreed upon by the team.

## Ground-Truth and Detection Outputs

### Ground-truth file

`results/ground_truth.csv` should contain one row per test file, with fields such as:

- `file_name`
- `split` (`validation` or `test`)
- `source_file`
- `tamper_start`
- `tamper_end`
- `tamper_type`
- `replacement_source`
- `random_seed`
- Parameters used, such as noise level or bitrate

Untouched and benignly processed files should have empty tamper start and end fields.

### Detected-interval file

`results/detected_intervals.csv` should contain fields such as:

- `file_name`
- `interval_start`
- `interval_end`
- `confidence`

The confidence value may, for example, be based on the mean watermark score over the detected interval. The final choice should be documented.

## Installation and Usage

The project plan specifies Python. The exact dependency versions and final command-line interface should be recorded in `requirements.txt` and updated to match the implementation.

### 1. Clone the repository

```bash
git clone <repository-url>
cd audio-tamper-localization
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The project plan identifies Python audio-processing and scientific-computing tools such as NumPy, SciPy, librosa, and Matplotlib as potential dependencies. The actual requirements file should be treated as authoritative.

### 4. Prepare the audio data

Follow the instructions in `data/README.md`. Use audio that is openly licensed or otherwise permitted for the project, and preserve source and licence information. Do not assume the dataset is included in the repository.

### 5. Run the pipeline

The intended design is to run the complete workflow from `main.py`, including preprocessing, watermark embedding, test-case generation, extraction, detection, evaluation, and plotting.

```bash
python main.py
```

**Note:** This command is the planned entry point. It will work only once `main.py` and the required modules have been implemented and configured. If the current implementation uses different commands or arguments, replace this section with the verified instructions.

## Testing and Reproducibility

Each module should include a small runnable example and tests. Random choices must use stored seeds, and the configuration used for each experiment must be recorded.

Recommended checks include:

### Audio processing
- STFT followed by inverse STFT reconstructs the signal with very small error.
- Output length matches input length.
- A known sine tone peaks near its expected frequency bin.
- A click at a known time maps to the expected frame.
- Stereo and non-16-kHz inputs are handled consistently.

### Watermarking
- The same key and frame index reproduce the same pattern.
- Different keys produce different, nearly uncorrelated patterns.
- Correctly watermarked audio produces stronger scores than unwatermarked or wrong-key controls.
- Quiet frames are marked invalid.
- The score timeline and audio length remain aligned.
- SNR and log-spectral distance are measured between original and watermarked audio.

### Tampering simulation
- Spliced audio has the same duration as its host.
- Samples outside the splice remain unchanged, subject to documented boundary handling.
- The inserted segment matches the intended donor segment, allowing for the crossfade.
- Noise transformations achieve their requested SNR within tolerance.
- Compression delay is checked and compensated where necessary.
- Ground-truth rows match the saved transformation parameters.

### Detection
- Synthetic evidence with a known score dip produces an interval near the expected region.
- A flat high-score timeline produces no suspicious interval.
- Undetermined windows are not counted as detections.
- Frame/window-to-time conversion is correct.
- Metrics are checked against small hand-calculated examples.

For reliable evaluation, calibrate the threshold on validation recordings, freeze the pipeline, and evaluate the test set only after the design is fixed.

## Division of Work

The project is organized into four cooperating modules:

| Member | Main responsibility | Primary deliverables |
|---|---|---|
| Member 1 | Audio preprocessing and STFT | `audio_processing.py`, shared configuration, preprocessing tests, plots, README/repository hygiene |
| Member 2 | Watermark generation, embedding, and extraction | `watermark.py`, extraction evidence, watermark validation experiments |
| Member 3 | Tampering simulation and dataset preparation | `tamper_simulation.py`, prepared test cases, `ground_truth.csv`, `data/README.md` |
| Member 4 | Detection, localization, and evaluation | `tamper_detector.py`, `main.py`, `detected_intervals.csv`, metrics, final plots |

The most important interface agreement is between the watermark extraction module and the detector. They must agree on score meaning, time alignment, invalid-frame handling, and the evidence-table format. The simulation module must also share sample-rate and timestamp conventions with the preprocessing module.

Team practices:
- Use one shared Git repository and make small, focused commits.
- Test each module independently before integration.
- Do not change another member's files without notifying them.
- Announce interface changes to the team.
- Use a short daily check-in to surface integration problems early.

## Limitations and Responsible Claims

The project evaluates a specific watermark-based approach to detecting audio splicing under controlled conditions.

- It does not establish general deepfake detection capability.
- It does not establish resistance to deliberate, informed attacks. An attacker may attempt to remove or re-embed a watermark.
- It does not guarantee detection of splices from audio carrying the same key at the same time position.
- Noise and compression robustness must be demonstrated at the exact levels and settings tested; it should not be assumed.
- Short splices, low-energy regions, donor audio similar to the host, and heavy compression may make detection more difficult.
- The estimated boundaries are limited by STFT framing and detection-window size. Report measured boundary errors rather than claiming exact timestamps.
- A low watermark score is evidence of a mismatch, not proof of malicious intent.
- The project is inspired by the general idea of watermark-based integrity checking, while the proposed implementation uses transparent DSP instead of learned components.

## Expected Deliverables

The completed project is intended to include:

1. An audio preprocessing and STFT module.
2. A watermark generation, embedding, and extraction module.
3. A reproducible tampering simulation module.
4. A sliding-window detector and timestamp-localization module.
5. Ground-truth and detected-interval CSV files.
6. Detection and localization metrics.
7. Waveform, spectrogram, and detection-score visualizations.
8. A documented setup and execution workflow.
9. A report describing methodology, experimental results, and limitations.
10. A demonstration comparing the original recording, spliced recording, detected suspicious interval, and ground-truth interval.

## Project Status

This README describes the agreed design and planned workflow. **Final performance numbers should be added only after experiments have been run and verified.** Document the actual dataset, configuration, threshold, test conditions, measured metrics, and known failure cases in the final report.

## Acknowledgements

The project follows a team-defined workflow for frequency-domain watermarking, controlled audio-splicing experiments, and sliding-window localization. The implementation and final claims should be based on the code and experiments actually completed by the team.
