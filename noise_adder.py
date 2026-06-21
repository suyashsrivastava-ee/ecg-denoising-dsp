"""
noise_adder.py
--------------
Adds three clinically realistic noise types to a clean ECG signal.

In real hospital recordings, ECG signals get corrupted by:

1. BASELINE WANDER (~0.1 – 0.5 Hz)
   Caused by patient breathing. The whole signal slowly drifts up and down.
   Frequency: very low (0.1–0.5 Hz), far below heart-rate signal (1–40 Hz).

2. POWERLINE INTERFERENCE (50 Hz in India, 60 Hz in USA)
   Electrical interference from the power supply in the room.
   Shows up as a constant high-amplitude sine wave at exactly 50 Hz.

3. EMG NOISE (Muscle Artifact) — random, wideband
   Caused by patient movement or muscle contractions near electrodes.
   Looks like random Gaussian noise spread across all frequencies.
"""

import numpy as np


def add_baseline_wander(ecg, t, amplitude=0.40, freq=0.25):
    """
    Simulate slow breathing-induced drift.
    We mix two sine waves at slightly different freqs for a more organic feel.
    """
    wander = amplitude * np.sin(2 * np.pi * freq * t)
    wander += 0.12 * amplitude * np.sin(2 * np.pi * 0.1 * t + np.pi / 3)
    return ecg + wander


def add_powerline_noise(ecg, t, freq=50.0, amplitude=0.18):
    """
    Add 50 Hz powerline interference (India standard).
    Pure sine wave at exactly 50 Hz — easy to identify in FFT.
    """
    noise = amplitude * np.sin(2 * np.pi * freq * t)
    return ecg + noise


def add_emg_noise(ecg, amplitude=0.07, seed=42):
    """
    Add random Gaussian noise simulating muscle artifact.
    Using a fixed seed so results are reproducible every run.
    """
    rng = np.random.default_rng(seed)
    noise = amplitude * rng.standard_normal(len(ecg))
    return ecg + noise


def corrupt_ecg(clean_ecg, t, fs):
    """
    Apply all three noise types in sequence.
    This is what the denoiser will try to reverse.
    """
    noisy = add_baseline_wander(clean_ecg, t)
    noisy = add_powerline_noise(noisy, t, freq=50.0)
    noisy = add_emg_noise(noisy)
    return noisy
