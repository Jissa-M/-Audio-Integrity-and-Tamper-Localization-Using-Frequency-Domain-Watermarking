# save as check_demo.py
import numpy as np, pandas as pd, soundfile as sf
import watermark as w, tamper_detector as d
thr = -0.00441344
g = pd.read_csv("results/ground_truth.csv")
g = g[g.split == "test"]
for n, rows in g.groupby("file_name"):
    x, sr = sf.read("data/generated/" + n)
    f = d.detect_intervals(d.score_windows(w.extract(x, "project_secret_key"), len(x)/sr), thr)
    true = [(r.tamper_start, r.tamper_end) for r in rows.itertuples() if pd.notna(r.tamper_start)]
    hit = any(a < t1 and b > t0 for a, b in [(q["start"], q["end"]) for q in f] for t0, t1 in true)
    print(f"{n:55s} true={'Y' if true else 'N'} flagged={len(f)} overlap={'Y' if hit else '-'}")