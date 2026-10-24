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
        rms_d = _rms(seg)
        if rms_d > 1e-8:
            gain = _rms(host[start:end]) / rms_d
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