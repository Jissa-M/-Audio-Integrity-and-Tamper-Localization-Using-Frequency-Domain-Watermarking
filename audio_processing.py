import numpy as np
import librosa

N_FFT = 1024
HOP_LENGTH = 256

def load_audio(path, sr=16000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def compute_stft(y):
    return librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann"
    )

def reconstruct_audio(S, length):
    return librosa.istft(
        S,
        hop_length=HOP_LENGTH,
        window="hann",
        length=length
    )
def reconstruction_metrics(y, y_reconstructed):
    error = y - y_reconstructed

    rmse = np.sqrt(np.mean(error ** 2))
    max_error = np.max(np.abs(error))

    signal_power = np.mean(y ** 2)
    noise_power = np.mean(error ** 2)

    snr_db = (
        10 * np.log10(signal_power / noise_power)
        if noise_power > 0
        else float("inf")
    )

    return rmse, max_error, snr_db
