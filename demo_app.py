import io
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st
import watermark as w
import tamper_detector as d

KEY = "project_secret_key"
FROZEN = -0.00441344   # frozen threshold reported by Member 4 (commit 5d8d531)

thr = FROZEN
src = "value reported by Member 4 (commit 5d8d531)"

st.title("Audio Tamper Localization Demo")
st.caption(f"Frozen threshold: {thr:.5f} (source: {src}). "
           "A window is flagged when its watermark score is below this.")

gt = pd.read_csv("results/ground_truth.csv")
names = sorted(gt.file_name.unique())
name = st.selectbox("Choose a test file", names)
rows = gt[gt.file_name == name]
split = rows.split.iloc[0]
st.write(f"Type: **{rows.tamper_type.iloc[0]}** | split: **{split}**")
if split != "test":
    st.info("This file is in the training/validation split. "
            "Prefer test files when presenting results.")

x, sr = sf.read(f"data/generated/{name}")

# 16-bit copy so every browser can play it
buf = io.BytesIO()
sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
st.audio(buf.getvalue(), format="audio/wav")

true = [(r.tamper_start, r.tamper_end) for r in rows.itertuples()
        if pd.notna(r.tamper_start)]
st.write("True tampered interval(s):",
         [(round(a, 2), round(b, 2)) for a, b in true] or "none")

if st.button("Analyse"):
    ev = w.extract(x, KEY)
    windows = d.score_windows(ev, len(x) / sr)
    found = d.detect_intervals(windows, thr)
    valid = [v for v in windows if v["valid"]]
    tm = [(v["start"] + v["end"]) / 2 for v in valid]
    sc = [v["score"] for v in valid]

    if found:
        txt = "; ".join(f"{f['start']:.2f} to {f['end']:.2f} s" for f in found)
        st.error(f"Suspicious interval(s): {txt}")
    else:
        st.success("No tampering found at the frozen threshold.")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(tm, sc, marker="o", ms=3, label="window score")
    ax.axhline(thr, color="black", ls="--", label="frozen threshold")
    for i, (a, b) in enumerate(true):
        ax.axvspan(a, b, color="red", alpha=0.2,
                   label="true" if i == 0 else None)
    for i, f in enumerate(found):
        ax.axvspan(f["start"], f["end"], color="blue", alpha=0.2,
                   label="detected" if i == 0 else None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Watermark score")
    ax.legend(loc="lower left")
    st.pyplot(fig)
    st.caption("Detected intervals can cover only part of a splice because the "
               "threshold is conservative (window recall about 31%).")