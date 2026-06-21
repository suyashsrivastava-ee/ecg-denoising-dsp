"""
ecg_generator.py
----------------
Generates a synthetic ECG signal using Gaussian functions
to approximate real PQRST waveform morphology.

The ECG (Electrocardiogram) records the electrical activity of the heart.
Each heartbeat produces a characteristic wave pattern called PQRST:
    - P wave  : Atrial depolarization (small bump before the spike)
    - QRS complex : Ventricular depolarization (the big spike)
    - T wave  : Ventricular repolarization (recovery hump after spike)
"""

import numpy as np


def _single_beat(t_beat):
    """
    Build one PQRST cycle using Gaussian functions.
    Time is normalized [0, 1] within the beat, so the shapes
    stay consistent regardless of heart rate.
    """
    beat = np.zeros_like(t_beat)

    # Normalize time within this beat to [0, 1]
    if t_beat[-1] == t_beat[0]:
        return beat
    t = (t_beat - t_beat[0]) / (t_beat[-1] - t_beat[0])

    def gauss(center, width, amplitude):
        return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    # P wave — small upward bump (atrial depolarization)
    beat += gauss(center=0.18, width=0.025, amplitude=0.15)

    # Q wave — tiny dip just before the R spike
    beat += gauss(center=0.36, width=0.012, amplitude=-0.09)

    # R wave — the tall spike (ventricular depolarization)
    beat += gauss(center=0.41, width=0.010, amplitude=1.50)

    # S wave — small dip right after R
    beat += gauss(center=0.46, width=0.012, amplitude=-0.22)

    # T wave — recovery hump
    beat += gauss(center=0.62, width=0.045, amplitude=0.33)

    return beat


def generate_ecg(duration=10, fs=500, heart_rate=72):
    """
    Generate a clean synthetic ECG signal.

    Parameters
    ----------
    duration   : signal length in seconds
    fs         : sampling frequency in Hz (samples per second)
    heart_rate : beats per minute (BPM)

    Returns
    -------
    t   : time array  (shape: N,)
    ecg : ECG signal  (shape: N,)  — amplitude in mV
    """
    t = np.arange(0, duration, 1.0 / fs)
    ecg = np.zeros_like(t)

    # How many samples fit in one heartbeat?
    samples_per_beat = int(fs * 60.0 / heart_rate)

    beat_start = 0
    while beat_start + samples_per_beat <= len(t):
        beat_end = beat_start + samples_per_beat
        ecg[beat_start:beat_end] = _single_beat(t[beat_start:beat_end])
        beat_start += samples_per_beat

    return t, ecg
