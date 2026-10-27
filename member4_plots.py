"""Headless comparison plots for Member 4."""
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'audio-member4-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def comparison(path, name, evidence, windows, predictions, truth, threshold,
               original=None, audio=None, sr=16000):
    have_audio = original is not None and audio is not None
    fig, axes = plt.subplots(5 if have_audio else 1, 1, figsize=(12, 10 if have_audio else 4),
                             sharex=True, squeeze=False, constrained_layout=True)
    axes = axes[:, 0]
    if have_audio:
        for ax, signal, label in [(axes[0], original, 'Untouched watermarked reference'),
                                   (axes[1], audio, 'Analyzed recording')]:
            stride = max(1, len(signal) // 12000)
            ax.plot(np.arange(len(signal))[::stride] / sr, signal[::stride], lw=.5)
            ax.set_ylabel('Amplitude'); ax.set_title(label, fontsize=10)
        for ax, signal, label in [(axes[2], original, 'Reference spectrogram'),
                                   (axes[3], audio, 'Analyzed spectrogram')]:
            ax.specgram(signal, NFFT=1024, Fs=sr, noverlap=768, cmap='magma')
            ax.set_ylabel('Hz'); ax.set_title(label, fontsize=10)
    ax = axes[-1]
    valid = evidence['valid']
    ax.plot(evidence['time_seconds'], np.where(valid, evidence['watermark_score'], np.nan),
            color='0.65', lw=.5, label='Frame score')
    ax.plot([(w['start'] + w['end']) / 2 for w in windows],
            [w['score'] if w['valid'] else np.nan for w in windows], label='Window mean', lw=1.2)
    ax.axhline(threshold, color='black', ls='--', label=f'Threshold {threshold:.4f}')
    for i, (s, e) in enumerate(truth):
        for a in axes: a.axvspan(s, e, color='green', alpha=.15)
        if i == 0: ax.plot([], [], color='green', lw=6, alpha=.4, label='Ground truth')
    for i, p in enumerate(predictions):
        ax.axvspan(p['start'], p['end'], color='red', alpha=.17, label='Detected' if i == 0 else None)
    ax.set_ylabel('Watermark score'); ax.set_xlabel('Time (seconds)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    fig.suptitle(name, fontsize=12)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140); plt.close(fig)


def calibration_plot(path, benign, spliced, threshold):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.hist(benign, bins=40, density=True, alpha=.6, label='Validation untouched/benign')
    if spliced: ax.hist(spliced, bins=40, density=True, alpha=.5, label='Validation spliced regions')
    ax.axvline(threshold, color='black', ls='--', label='Frozen threshold')
    ax.set(xlabel='Window score', ylabel='Density', title='Validation only: score separation')
    ax.legend(fontsize=8); fig.savefig(path, dpi=140); plt.close(fig)


def summary_plot(path, groups):
    names = list(groups)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, metric, label in zip(axes, ['file_detection_rate', 'file_false_positive_rate'],
                               ['Spliced-file detection rate', 'Benign-file false-positive rate']):
        pairs = [(name, groups[name][metric]) for name in names if groups[name][metric] is not None]
        if pairs:
            ax.bar(range(len(pairs)), [p[1] for p in pairs])
            ax.set_xticks(range(len(pairs)), [p[0] for p in pairs], rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_title(label); ax.set_ylabel('Fraction')
    fig.savefig(path, dpi=140); plt.close(fig)
