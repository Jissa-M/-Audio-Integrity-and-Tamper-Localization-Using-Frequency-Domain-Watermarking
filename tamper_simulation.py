"""Tamper simulation: splicing and verification (Member 3)."""
import numpy as np

SR = 16000   # frozen project sample rate
HOP = 256    # frozen STFT hop (16 ms)


def sec_to_samples(t, sr=SR):
    """The ONE rounding rule used everywhere: round to nearest sample."""
    return int(round(t * sr))


def samples_to_sec(n, sr=SR):
    return n / sr


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def splice(host, donor, start_s, end_s, donor_start_s=0.0,
           sr=SR, crossfade_ms=5.0, match_gain=True):
    """Replace host[start:end] with a same-length donor segment.

    The crossfade sits INSIDE the interval, so samples outside
    [start, end) are untouched and the nominal boundaries stay exact.
    Returns (spliced_audio, info_dict).
    """
    start = sec_to_samples(start_s, sr)
    end = sec_to_samples(end_s, sr)
    n = end - start
    d0 = sec_to_samples(donor_start_s, sr)
    xf = int(round(crossfade_ms * 1e-3 * sr))

    if n <= 0:
        raise ValueError("end must be after start")
    if start < sr or end > len(host) - sr:
        raise ValueError("splice must stay out of the first and last second")
    if d0 + n > len(donor):
        raise ValueError("donor is too short for this splice")
    if xf >= HOP:
        raise ValueError("crossfade must be shorter than one STFT hop")
    if 2 * xf >= n:
        raise ValueError("splice too short for this crossfade")

    seg = donor[d0:d0 + n].astype(np.float64)

    gain = 1.0
    if match_gain:
        rms_d = _rms(donor)
        if rms_d > 1e-8:
            gain = _rms(host) / rms_d
    seg = seg * gain
    peak = float(np.max(np.abs(seg)))
    if peak > 0.99:                      # prevent clipping
        scale = 0.99 / peak
        seg *= scale
        gain *= scale

    host_seg = host[start:end].astype(np.float64)
    w = np.linspace(0.0, 1.0, xf)        # host weight at the edges
    region = seg.copy()
    region[:xf] = host_seg[:xf] * (1 - w) + seg[:xf] * w
    region[-xf:] = host_seg[-xf:] * w + seg[-xf:] * (1 - w)
    # note: at each edge the first/last sample is pure host, so no click

    out = host.copy()
    out[start:end] = region.astype(host.dtype)

    info = {
        "start_sample": start, "end_sample": end,
        "start_s": samples_to_sec(start, sr), "end_s": samples_to_sec(end, sr),
        "donor_start_sample": d0, "gain": gain, "crossfade_samples": xf,
    }
    return out, info


def verify_splice(host, spliced, donor, info, atol=1e-5):
    """Mandatory checks. Raises AssertionError if anything is wrong."""
    s, e, d0 = info["start_sample"], info["end_sample"], info["donor_start_sample"]
    xf = info["crossfade_samples"]

    assert len(spliced) == len(host), "length changed"
    assert np.array_equal(spliced[:s], host[:s]), "samples before start changed"
    assert np.array_equal(spliced[e:], host[e:]), "samples after end changed"

    expected = donor[d0:d0 + (e - s)].astype(np.float64) * info["gain"]
    core = slice(xf, (e - s) - xf)       # skip the crossfade zones
    inside = spliced[s:e].astype(np.float64)
    assert np.allclose(inside[core], expected[core], atol=atol), \
        "interval does not match donor"
    return True


import subprocess
import tempfile
import os
import soundfile as sf
from scipy.signal import correlate


def add_noise(x, snr_db, seed):
    """Add white Gaussian noise at a target SNR. Returns (noisy, info)."""
    rng = np.random.default_rng(seed)
    x64 = x.astype(np.float64)
    p_sig = np.mean(x64 ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(x64)) * np.sqrt(p_noise)
    noisy = x64 + noise
    achieved = 10 * np.log10(p_sig / np.mean(noise ** 2))
    peak = np.max(np.abs(noisy))
    if peak > 0.99:
        noisy *= 0.99 / peak
    return noisy.astype(x.dtype), {
        "target_snr_db": snr_db, "achieved_snr_db": float(achieved), "seed": seed}


def estimate_delay(ref, test, max_lag=4000):
    """Lag (samples) of `test` relative to `ref`; positive = test is late."""
    n = min(len(ref), len(test))
    a = ref[:n].astype(np.float64)
    b = test[:n].astype(np.float64)
    c = correlate(b, a, mode="full", method="fft")
    lags = np.arange(-(n - 1), n)
    keep = np.abs(lags) <= max_lag
    return int(lags[keep][np.argmax(c[keep])])


def compress_roundtrip(x, codec="mp3", bitrate_kbps=128, sr=SR):
    """Encode with ffmpeg, decode back, remove delay, restore exact length.
    Returns (decoded, info)."""
    ext = {"mp3": "mp3", "aac": "m4a"}[codec]
    lib = {"mp3": "libmp3lame", "aac": "aac"}[codec]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.wav")
        enc = os.path.join(d, f"enc.{ext}")
        dec = os.path.join(d, "out.wav")
        sf.write(src, x, sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", src, "-c:a", lib, "-b:a", f"{bitrate_kbps}k", enc],
                       check=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", enc, "-ac", "1", "-ar", str(sr), dec], check=True)
        y, _ = sf.read(dec, dtype="float64")

    lag = estimate_delay(x, y)
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        y = np.concatenate([np.zeros(-lag), y])
    if len(y) >= len(x):
        y = y[:len(x)]
    else:
        y = np.concatenate([y, np.zeros(len(x) - len(y))])

    residual = estimate_delay(x, y)
    return y.astype(x.dtype), {
        "codec": codec, "bitrate_kbps": bitrate_kbps,
        "delay_removed_samples": lag, "residual_delay_samples": residual,
        "tool": "ffmpeg " + lib}

import csv
import json
import glob
from scipy.signal import resample_poly

try:                                   # Member 2's function, when it exists
    from watermark import embed        # expected: embed(audio, key) -> audio
except ImportError:
    def embed(audio, key):             # PLACEHOLDER: returns audio unchanged
        return audio.copy()

KEY = "project_secret_key"
WRONG_KEY = "some_other_key"
SPLICE_LENGTHS = [0.5, 1.0, 2.0, 3.0, 5.0]
POSITIONS = ["early", "mid", "late"]
# Provisional split for the few clips we have. Revise when all 12 exist.
SPLITS = {"src01": "validation", "src02": "validation",
          "src03": "test", "src04": "test"}


def _fit(y, n):
    return y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y))])


def _pick_start(rng, dur, length, zone):
    lo, hi = 1.2, dur - 1.2 - length
    third = (hi - lo) / 3
    return float(rng.uniform(lo + zone * third, lo + (zone + 1) * third))


def _pick_donor(rng, donors, length):
    name = sorted(donors)[int(rng.integers(len(donors)))]
    seg_max = len(donors[name]) / SR - length
    return name, float(rng.uniform(0, seg_max))


def build_dataset(src_dir="data/sources", donor_dir="data/donors",
                  out_dir="data/generated", gt_path="results/ground_truth.csv",
                  seed=2026):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(gt_path), exist_ok=True)
    rng = np.random.default_rng(seed)
    donors = {os.path.basename(p)[:-4]: sf.read(p, dtype="float64")[0]
              for p in glob.glob(os.path.join(donor_dir, "*.wav"))}
    donors_wm = {k: embed(v, WRONG_KEY) for k, v in donors.items()}
    rows = []

    def save(name, audio, split, src, ttype, start, end, repl, fseed, params):
        sf.write(os.path.join(out_dir, name), audio, SR, subtype="FLOAT")
        rows.append({"file_name": name, "split": split, "source_file": src,
                     "tamper_start": start, "tamper_end": end,
                     "tamper_type": ttype, "replacement_source": repl,
                     "random_seed": fseed, "parameters": json.dumps(params)})

    for sid, split in sorted(SPLITS.items()):
        x = sf.read(os.path.join(src_dir, sid + ".wav"), dtype="float64")[0]
        host = embed(x, KEY)
        dur = len(host) / SR
        save(f"{sid}_untouched.wav", host, split, sid, "untouched",
             "", "", "", seed, {})

        # --- single splices ---
        for i, L in enumerate(SPLICE_LENGTHS):
            pos = POSITIONS[i % 3]
            dtype = "unwatermarked" if i % 2 == 0 else "wrongkey"
            fseed = int(rng.integers(1_000_000))
            r = np.random.default_rng(fseed)
            start = _pick_start(r, dur, L, i % 3)
            dname, dstart = _pick_donor(r, donors, L)
            pool = donors if dtype == "unwatermarked" else donors_wm
            out, info = splice(host, pool[dname], start, start + L, dstart)
            verify_splice(host, out, pool[dname], info)
            save(f"{sid}_splice_{dtype}_{L}s_{pos}.wav", out, split, sid,
                 "splicing", info["start_s"], info["end_s"], dname, fseed,
                 {"length_s": L, "position": pos, "donor_type": dtype,
                  "gain": info["gain"], "start_sample": info["start_sample"],
                  "end_sample": info["end_sample"]})

        # --- one file with two splices (one ground-truth row per splice) ---
        fseed = int(rng.integers(1_000_000))
        r = np.random.default_rng(fseed)
        cur, splices = host, []
        for L, zone in [(1.0, 0), (2.0, 2)]:
            start = _pick_start(r, dur, L, zone)
            dname, dstart = _pick_donor(r, donors, L)
            out, info = splice(cur, donors[dname], start, start + L, dstart)
            verify_splice(cur, out, donors[dname], info)
            splices.append((info, dname))
            cur = out
        name = f"{sid}_splice_double.wav"
        sf.write(os.path.join(out_dir, name), cur, SR, subtype="FLOAT")
        for info, dname in splices:
            rows.append({"file_name": name, "split": split, "source_file": sid,
                         "tamper_start": info["start_s"],
                         "tamper_end": info["end_s"],
                         "tamper_type": "splicing_double",
                         "replacement_source": dname, "random_seed": fseed,
                         "parameters": json.dumps(
                             {"start_sample": info["start_sample"],
                              "end_sample": info["end_sample"]})})

        # --- benign processing ---
        for snr in (40, 30, 20):
            y, info = add_noise(host, snr, seed=seed + snr)
            save(f"{sid}_noise_{snr}dB.wav", y, split, sid, "noise",
                 "", "", "", seed + snr, info)
        for kbps in (192, 128, 64):
            y, info = compress_roundtrip(host, "mp3", kbps)
            save(f"{sid}_mp3_{kbps}k.wav", y, split, sid, "mp3",
                 "", "", "", seed, info)
        for g in (0.7, 1.3):
            y = np.clip(host * g, -0.99, 0.99)
            save(f"{sid}_volume_{g}.wav", y, split, sid, "volume",
                 "", "", "", seed, {"gain": g})
        y = _fit(resample_poly(resample_poly(host, 3, 4), 4, 3), len(host))
        save(f"{sid}_resample_12k.wav", y, split, sid, "resample",
             "", "", "", seed, {"via_hz": 12000})

    with open(gt_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    r = build_dataset()
    print(f"wrote {len(r)} ground-truth rows")