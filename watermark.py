"""Frequency-domain watermark: embedding and blind extraction (Member 2).

Public API
    embed(audio, key)            -> watermarked audio, same length (float, 16 kHz)
    extract(audio, key)          -> per-frame evidence (one row per STFT frame)
    save_evidence_csv(ev, path)  -> writes frame_index,time_seconds,watermark_score,valid
    snr_db(a, b), lsd_db(a, b)   -> quality metrics
    get_settings()               -> parameters needed to reproduce extraction

Design (matches the project README)
    * STFT: Hann, 1024 window, 256 hop, 16 kHz.
    * Per frame, a key- and frame-index-dependent set of N_BINS bins is chosen
      pseudo-randomly inside the 1.5-6 kHz band, with a +/-1 pattern.
    * Embedding multiplies those magnitudes by (1 + ALPHA * pattern); phase is kept.
    * Frames whose band level is below ENERGY_FLOOR are skipped (not embedded).
    * Extraction is blind: log-magnitude, remove smoothed spectral envelope,
      normalized correlation with the expected pattern -> score in [-1, 1].
    * Quiet / edge / low-residual frames are marked valid = False (not "tampered").

NOTE: constants below mirror config.py. Once Member 1's config.py and the shared
STFT/ISTFT in audio_processing.py exist, import from them instead (see _stft/_istft).
"""
import csv
import hashlib

import numpy as np

# ---- parameters (keep in sync with config.py) -------------------------------
SR = 16000
N_FFT = 1024            # 64 ms
HOP = 256               # 16 ms, 75% overlap
BAND_HZ = (1500.0, 6000.0)
N_BINS = 150            # selected bins per frame (100-200 suggested)
ALPHA = 0.20            # embedding strength (relative magnitude change); tune
ENERGY_FLOOR = 1e-3     # min band level (per-bin amplitude units) to embed / be valid
RESID_FLOOR = 1e-6      # min residual norm for a valid score
ENV_SMOOTH_BINS = 21    # spectral-envelope smoothing width (bins)
KEY_ID = "default"      # recorded in settings for reproducibility


# ---- helpers ----------------------------------------------------------------
def _key_to_int(key):
    if isinstance(key, (int, np.integer)):
        return int(key)
    return int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "big")


def _window():
    return np.hanning(N_FFT + 1)[:-1]  # periodic Hann


def _band_bins():
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    return np.where((freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1]))[0]


def _frame_pattern(key_int, t, band):
    """Selected bins (sorted) and +/-1 pattern for frame t. Reproducible."""
    rng = np.random.default_rng([key_int, int(t)])
    sel = np.sort(rng.choice(band, size=min(N_BINS, len(band)), replace=False))
    pn = rng.choice([-1.0, 1.0], size=len(sel))
    return sel, pn


def _stft(x):
    """Centered STFT; frame t is centred on sample t*HOP. Returns (spec, padded_len)."""
    win = _window()
    pad = N_FFT // 2
    n_frames = int(np.ceil(len(x) / HOP)) + 1
    total = (n_frames - 1) * HOP + N_FFT
    xp = np.zeros(total)
    xp[pad:pad + len(x)] = x
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * win, axis=1), total


def _istft(spec, total, length):
    win = _window()
    frames = np.fft.irfft(spec, n=N_FFT, axis=1) * win
    out = np.zeros(total)
    norm = np.zeros(total)
    for t, fr in enumerate(frames):
        s = t * HOP
        out[s:s + N_FFT] += fr
        norm[s:s + N_FFT] += win ** 2
    out /= np.maximum(norm, 1e-8)
    pad = N_FFT // 2
    return out[pad:pad + length]


def _band_level(mag_band):
    """Per-frame band level, roughly per-bin amplitude (window-gain normalised)."""
    return np.sqrt(np.mean(mag_band ** 2, axis=-1)) / np.sqrt(np.sum(_window() ** 2))


def _smooth_envelope(logm):
    """Moving average across frequency with edge normalisation."""
    k = np.ones(ENV_SMOOTH_BINS)
    num = np.convolve(logm, k, mode="same")
    den = np.convolve(np.ones_like(logm), k, mode="same")
    return num / den


# ---- embedding --------------------------------------------------------------
def embed(audio, key, alpha=None):
    """Embed the watermark. audio: 1-D float array at 16 kHz. Same-length output."""
    a = np.asarray(audio)
    if a.ndim != 1:
        raise ValueError("audio must be 1-D mono")
    if len(a) == 0:
        return a.copy()
    alpha = ALPHA if alpha is None else alpha
    x = a.astype(np.float64)
    k = _key_to_int(key)
    band = _band_bins()

    spec, total = _stft(x)
    mag, phase = np.abs(spec), np.angle(spec)
    level = _band_level(mag[:, band])
    for t in range(spec.shape[0]):
        if level[t] < ENERGY_FLOOR:
            continue  # too quiet to carry the watermark
        sel, pn = _frame_pattern(k, t, band)
        mag[t, sel] *= 1.0 + alpha * pn
    y = _istft(mag * np.exp(1j * phase), total, len(x))
    out_dtype = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float32
    return y.astype(out_dtype)


# ---- extraction (blind) -----------------------------------------------------
def extract(audio, key):
    """Return per-frame evidence: dict of arrays, one entry per STFT frame.

    keys: frame_index, time_seconds, watermark_score (in [-1,1]), valid (bool)
    Invalid frames keep a row (score = 0.0) so the timeline stays continuous.
    """
    x = np.asarray(audio, dtype=np.float64)
    k = _key_to_int(key)
    band = _band_bins()
    spec, _ = _stft(x)
    n = spec.shape[0]
    mag = np.abs(spec)
    level = _band_level(mag[:, band])

    scores = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    for t in range(n):
        if t == 0 or t == n - 1 or level[t] < ENERGY_FLOOR:
            continue  # edge / quiet frame -> invalid
        logm = np.log(mag[t, band] + 1e-9)
        resid = logm - _smooth_envelope(logm)
        sel, pn = _frame_pattern(k, t, band)
        r = resid[np.searchsorted(band, sel)]
        rn = np.linalg.norm(r)
        if rn < RESID_FLOOR:
            continue
        scores[t] = float(np.dot(r, pn) / (rn * np.linalg.norm(pn)))
        valid[t] = True
    idx = np.arange(n)
    return {
        "frame_index": idx,
        "time_seconds": idx * HOP / SR,  # frame centre time
        "watermark_score": scores,
        "valid": valid,
    }


def save_evidence_csv(ev, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "time_seconds", "watermark_score", "valid"])
        for i, t, s, v in zip(ev["frame_index"], ev["time_seconds"],
                              ev["watermark_score"], ev["valid"]):
            w.writerow([int(i), f"{t:.6f}", f"{s:.6f}", bool(v)])


# ---- quality metrics --------------------------------------------------------
def snr_db(original, processed):
    o = np.asarray(original, dtype=np.float64)
    p = np.asarray(processed, dtype=np.float64)
    noise = np.sum((o - p) ** 2)
    return float("inf") if noise == 0 else float(10 * np.log10(np.sum(o ** 2) / noise))


def lsd_db(original, processed):
    """Log-spectral distance (dB), mean over frames."""
    so, _ = _stft(np.asarray(original, dtype=np.float64))
    sp, _ = _stft(np.asarray(processed, dtype=np.float64))
    d = 20 * np.log10((np.abs(so) + 1e-9) / (np.abs(sp) + 1e-9))
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=1))))


def get_settings():
    return {"key_id": KEY_ID, "sr": SR, "n_fft": N_FFT, "hop": HOP,
            "band_hz": BAND_HZ, "n_bins": N_BINS, "alpha": ALPHA,
            "energy_floor": ENERGY_FLOOR, "env_smooth_bins": ENV_SMOOTH_BINS}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal(SR * 8)).astype(np.float32)
    y = embed(x, "demo")
    ev = extract(y, "demo")
    print("length ok:", len(y) == len(x), "| SNR dB:", round(snr_db(x, y), 1))
    print("mean valid score:", ev["watermark_score"][ev["valid"]].mean())
