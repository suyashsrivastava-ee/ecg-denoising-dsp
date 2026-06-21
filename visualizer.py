"""
visualizer.py
-------------
Generates two publication-style plots:
    1. Time-domain pipeline  — shows each denoising stage in the time domain
    2. Frequency spectrum    — FFT comparison (clean vs noisy vs denoised)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless rendering (comment out if running locally)
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- Color palette ---
C = {
    'clean':   '#2ECC71',   # green
    'noisy':   '#E74C3C',   # red
    'hp':      '#F39C12',   # orange
    'notch':   '#3498DB',   # blue
    'final':   '#8E44AD',   # purple
    'bg':      '#F7F9FC',
    'grid':    '#E0E6EE',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': C['grid'],
    'grid.linewidth': 0.6,
    'figure.facecolor': C['bg'],
    'axes.facecolor': C['bg'],
})


# ── Plot 1: Time-domain pipeline ────────────────────────────────────────────

def plot_pipeline(t, clean, noisy, after_hp, after_notch, denoised,
                  fs, output_dir='output'):
    """5-panel plot showing every stage of the denoising pipeline."""

    os.makedirs(output_dir, exist_ok=True)

    # Show only first 4 seconds — easier to read PQRST details
    n_show = min(int(4 * fs), len(t))
    ts = t[:n_show]

    stages = [
        (clean[:n_show],      'Clean ECG  (Ground Truth)',                      C['clean']),
        (noisy[:n_show],      'Noisy ECG  (Baseline Wander + Powerline + EMG)', C['noisy']),
        (after_hp[:n_show],   'Stage 1 → After High-Pass Filter  (≥ 0.5 Hz)',   C['hp']),
        (after_notch[:n_show],'Stage 2 → After Notch Filter  (50 Hz removed)',  C['notch']),
        (denoised[:n_show],   'Stage 3 → Final Denoised ECG',                   C['final']),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(15, 13), sharex=True)
    fig.suptitle('ECG Signal Denoising — Multi-Stage DSP Pipeline',
                 fontsize=16, fontweight='bold', y=1.005)

    for ax, (sig, label, color) in zip(axes, stages):
        ax.plot(ts, sig, color=color, linewidth=1.1, alpha=0.92)
        ax.set_title(label, fontsize=11, fontweight='semibold', loc='left', pad=5)
        ax.set_ylabel('Amplitude\n(mV)', fontsize=9)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel('Time (seconds)', fontsize=11)
    plt.tight_layout(pad=1.5, h_pad=0.8)

    out_path = os.path.join(output_dir, 'pipeline.png')
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ── Plot 2: Frequency spectrum (FFT) ────────────────────────────────────────

def plot_spectrum(clean, noisy, denoised, fs, output_dir='output'):
    """
    Side-by-side FFT magnitude plots.
    The 50 Hz spike (powerline) is clearly visible in the noisy spectrum
    and gone in the denoised one — this is the money shot for LinkedIn.
    """
    os.makedirs(output_dir, exist_ok=True)

    N = len(clean)
    freqs = fftfreq(N, d=1.0 / fs)
    pos   = freqs[:N // 2]           # positive frequencies only

    # Limit x-axis to 0–80 Hz where all the interesting stuff lives
    max_idx = int(80.0 * N / fs)

    def magnitude(sig):
        return (2.0 / N) * np.abs(fft(sig)[:N // 2])

    specs = [
        (magnitude(clean),    'Clean ECG Spectrum',    C['clean']),
        (magnitude(noisy),    'Noisy ECG Spectrum',    C['noisy']),
        (magnitude(denoised), 'Denoised ECG Spectrum', C['final']),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Frequency Domain Analysis  (FFT Magnitude Spectra)',
                 fontsize=14, fontweight='bold')

    for ax, (spec, label, color) in zip(axes, specs):
        ax.plot(pos[:max_idx], spec[:max_idx], color=color, linewidth=1.1)
        ax.set_title(label, fontsize=11, fontweight='semibold')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Magnitude (mV)', fontsize=10)
        ax.tick_params(labelsize=8)

        # Annotate key frequencies
        ax.axvline(x=50.0, color='#E74C3C', linestyle='--',
                   linewidth=0.9, alpha=0.55, label='50 Hz (powerline)')
        ax.axvline(x=0.5,  color='#F39C12', linestyle='--',
                   linewidth=0.9, alpha=0.55, label='0.5 Hz (HP cutoff)')
        ax.axvline(x=40.0, color='#3498DB', linestyle='--',
                   linewidth=0.9, alpha=0.55, label='40 Hz (LP cutoff)')
        ax.legend(fontsize=7.5, framealpha=0.6)

    plt.tight_layout(pad=1.5)

    out_path = os.path.join(output_dir, 'spectrum.png')
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ── SNR report ───────────────────────────────────────────────────────────────

def print_report(clean, noisy, denoised, snr_fn):
    snr_before = snr_fn(clean, noisy)
    snr_after  = snr_fn(clean, denoised)
    improvement = snr_after - snr_before

    print()
    print("╔══════════════════════════════════════════╗")
    print("║     ECG DENOISING — PERFORMANCE REPORT   ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  SNR  (noisy ECG)     :  {snr_before:>7.2f} dB        ║")
    print(f"║  SNR  (denoised ECG)  :  {snr_after:>7.2f} dB        ║")
    print(f"║  Improvement          :  +{improvement:>6.2f} dB        ║")
    print("╚══════════════════════════════════════════╝")
    print()
