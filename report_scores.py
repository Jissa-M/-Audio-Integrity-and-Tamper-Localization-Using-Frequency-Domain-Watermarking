import soundfile as sf, numpy as np, watermark as w

print("file    mean_score  SNR_dB  valid_frac  LSD_dB")
for i in range(1, 13):
    sid = f"src{i:02d}"
    a = sf.read(f"data/sources/{sid}.wav")[0]
    b = sf.read(f"data/generated/{sid}_untouched.wav")[0]
    ev = w.extract(b, "project_secret_key")
    ok = ev["valid"].astype(bool)
    snr = 10 * np.log10(np.sum(a**2) / np.sum((a - b)**2))
    lsd = w.lsd_db(a, b)
    print(f"{sid}   {np.mean(ev['watermark_score'][ok]):9.3f}  {snr:6.1f}  {ok.mean():9.2f}  {lsd:6.2f}")