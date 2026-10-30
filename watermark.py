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

# Use Member 1's shared STFT / inverse STFT when available so frame counts and
# time mapping match the rest of the pipeline. Falls back to a local numpy
# implementation (same Hann/1024/256 settings) for standalone testing.
try:
    from audio_processing import compute_stft, reconstruct_audio
    from audio_processing import N_FFT as _SH_NFFT, HOP_LENGTH as _SH_HOP
    _SHARED = True
except ImportError:
    _SHARED = False

# ---- parameters (keep in sync with config.py) -------------------------------
SR = 16000
N_FFT = _SH_NFFT if _SHARED else 1024   # 64 ms
HOP = _SH_HOP if _SHARED else 256        # 16 ms, 75% overlap
BAND_HZ = (1500.0, 6000.0)
N_BINS = 150            # selected bins per frame (100-200 suggested)
ALPHA = 0.35            # fixed-mode strength (relative magnitude change); tune
ENERGY_FLOOR = 1e-3     # min band level (per-bin amplitude units) to embed / be valid
RESID_FLOOR = 1e-6      # min residual norm for a valid score
ENV_SMOOTH_BINS = 21    # spectral-envelope smoothing width (bins)
PATTERN_BLOCK = 4       # frames sharing one pattern (= N_FFT // HOP) so the change survives overlap-add
ITERS = 3               # consistency iterations (STFT -> re-impose target magnitudes -> ISTFT)
ALPHA_MIN, ALPHA_MAX = 0.10, 0.60   # bounds for adaptive strength
TARGET_SNR_DB = 30.0    # adaptive mode: strongest alpha that still gives SNR >= this (None = fixed ALPHA)
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
    rng = np.random.default_rng([key_int, int(t) // PATTERN_BLOCK])
    sel = np.sort(rng.choice(band, size=min(N_BINS, len(band)), replace=False))
    pn = rng.choice([-1.0, 1.0], size=len(sel))
    return sel, pn


def _stft(x):
    """Centered STFT; frame t is centred on sample t*HOP. Returns (spec, padded_len)."""
    if _SHARED:
        return compute_stft(x).T, len(x)  # (bins, frames) -> (frames, bins)
    win = _window()
    pad = N_FFT // 2
    n_frames = int(np.ceil(len(x) / HOP)) + 1
    total = (n_frames - 1) * HOP + N_FFT
    xp = np.zeros(total)
    xp[pad:pad + len(x)] = x
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * win, axis=1), total


def _istft(spec, total, length):
    if _SHARED:
        return reconstruct_audio(spec.T, length)
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
def _embed_once(x, key_int, alpha):
    band = _band_bins()
    spec, total = _stft(x)
    mag, phase = np.abs(spec), np.angle(spec)
    level = _band_level(mag[:, band])
    for t in range(spec.shape[0]):
        if level[t] < ENERGY_FLOOR:
            continue  # too quiet to carry the watermark
        sel, pn = _frame_pattern(key_int, t, band)
        mag[t, sel] *= 1.0 + alpha * pn
    target = mag
    y = _istft(target * np.exp(1j * phase), total, len(x))
    for _ in range(ITERS):  # make the STFT of the output match the target magnitudes
        S, _t = _stft(y)
        n = min(S.shape[0], target.shape[0])
        y = _istft(target[:n] * np.exp(1j * np.angle(S[:n])), total, len(x))
    return y


def embed(audio, key, alpha=None):
    """Embed the watermark. audio: 1-D float array at 16 kHz. Same-length output.

    alpha=None and TARGET_SNR_DB set -> adaptive: strongest alpha in
    [ALPHA_MIN, ALPHA_MAX] whose SNR is still >= TARGET_SNR_DB (extraction does not
    depend on alpha, so strength may differ per file).
    alpha given (or TARGET_SNR_DB=None) -> fixed strength.
    """
    a = np.asarray(audio)
    if a.ndim != 1:
        raise ValueError("audio must be 1-D mono")
    if len(a) == 0:
        return a.copy()
    x = a.astype(np.float64)
    k = _key_to_int(key)
    if alpha is not None or TARGET_SNR_DB is None:
        y = _embed_once(x, k, ALPHA if alpha is None else alpha)
    else:
        y = _embed_once(x, k, ALPHA_MAX)
        if snr_db(x, y) < TARGET_SNR_DB:
            lo, hi = ALPHA_MIN, ALPHA_MAX
            y = _embed_once(x, k, lo)  # fallback if even the weakest misses the target
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                ym = _embed_once(x, k, mid)
                if snr_db(x, ym) >= TARGET_SNR_DB:
                    lo, y = mid, ym
                else:
                    hi = mid
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
