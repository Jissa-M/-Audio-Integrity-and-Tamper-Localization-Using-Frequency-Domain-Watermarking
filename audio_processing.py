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
