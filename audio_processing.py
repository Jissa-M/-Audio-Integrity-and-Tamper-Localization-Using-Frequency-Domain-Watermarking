

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display 

def load_audio(path, sr=16000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

y, sr = load_audio("audio.wav")

print("Sample rate:", sr)
print("Duration:", len(y) / sr, "seconds")
print("Samples:", len(y))
print("Peak amplitude:", np.max(np.abs(y)))


# In[3]:


N_FFT = 1024
HOP_LENGTH = 256

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

S = compute_stft(y)
y_reconstructed = reconstruct_audio(S, len(y))

print("Original shape:", y.shape)
print("STFT shape:", S.shape)
print("Reconstructed shape:", y_reconstructed.shape)


# In[4]:


time = np.arange(len(y)) / sr

plt.figure(figsize=(12, 3))
plt.plot(time, y)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Audio Waveform")
plt.tight_layout()
plt.show()


# In[5]:


error = y - y_reconstructed

rmse = np.sqrt(np.mean(error**2))
max_error = np.max(np.abs(error))

signal_power = np.mean(y**2)
noise_power = np.mean(error**2)

snr_db = (
    10 * np.log10(signal_power / noise_power)
    if noise_power > 0 else float("inf")
)

print("RMSE:", rmse)
print("Maximum absolute error:", max_error)
print("Reconstruction SNR:", snr_db, "dB")


# In[6]:


plt.figure(figsize=(12, 3))
plt.plot(time, error)
plt.xlabel("Time (seconds)")
plt.ylabel("Error")
plt.title("Reconstruction Error")
plt.tight_layout()
plt.show()


# In[8]:


import numpy as np
import librosa

N_FFT = 1024
HOP_LENGTH = 256

def load_audio(path, sr=22050):
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

