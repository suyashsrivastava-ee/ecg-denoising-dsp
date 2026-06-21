"""
filters.py
----------
Digital filters for ECG denoising.

All three filters are designed using the Butterworth approximation,
which gives a maximally flat passband — no ripples in the frequencies
we want to keep. We use filtfilt() instead of lfilter() everywhere:
    - lfilter  : one-pass, introduces phase delay (signal shifts in time)
    - filtfilt : two-pass (forward + backward), zero phase delay — critical
                 for ECG where timing of each wave has clinical meaning.

Filter design pipeline:
    analog prototype → digital transform (bilinear) → normalize to [0,1]
    All handled by scipy.signal.butter / iirnotch automatically.
"""

import numpy as np
from scipy import signal


def highpass_filter(data, fs, cutoff=0.5, order=4):
    """
    Remove baseline wander.

    Passes everything ABOVE 0.5 Hz → kills the slow breathing-induced drift.
    Butterworth order 4 = good rolloff without too much distortion.
    """
    nyq = fs / 2.0                   # Nyquist frequency
    norm_cutoff = cutoff / nyq       # Normalize to [0, 1]
    b, a = signal.butter(order, norm_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data)


def notch_filter(data, fs, target_freq=50.0, Q=35.0):
    """
    Remove powerline interference at exactly 50 Hz.

    The notch filter is a band-reject filter with a very narrow stopband.
    Q factor controls how narrow:
        - High Q (e.g. 35) → very narrow notch → kills only 50 Hz ± tiny margin
        - Low Q  (e.g. 5)  → wide notch → kills more surrounding frequencies too
    We want high Q to avoid distorting nearby ECG content.
    """
    b, a = signal.iirnotch(target_freq, Q, fs)
    return signal.filtfilt(b, a, data)


def lowpass_filter(data, fs, cutoff=40.0, order=4):
    """
    Remove high-frequency EMG / muscle artifact noise.

    ECG clinically useful content lives below ~40 Hz.
    Anything above that is mostly noise — cut it.
    """
    nyq = fs / 2.0
    norm_cutoff = cutoff / nyq
    b, a = signal.butter(order, norm_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def run_denoising_pipeline(noisy_ecg, fs):
    """
    Three-stage denoising pipeline. Order matters:
        Stage 1 → High-pass  : removes baseline wander  (< 0.5 Hz)
        Stage 2 → Notch      : removes powerline noise   (50 Hz)
        Stage 3 → Low-pass   : removes EMG artifacts     (> 40 Hz)

    Returns intermediate outputs so we can visualize each stage.
    """
    after_hp    = highpass_filter(noisy_ecg, fs, cutoff=0.5, order=4)
    after_notch = notch_filter(after_hp,    fs, target_freq=50.0, Q=35.0)
    denoised    = lowpass_filter(after_notch, fs, cutoff=40.0, order=4)
    return after_hp, after_notch, denoised


def compute_snr_db(clean, estimate):
    """
    Signal-to-Noise Ratio in decibels.

    SNR (dB) = 10 * log10 ( signal_power / noise_power )

    Higher is better. Positive = signal dominates. Negative = noise dominates.
    We treat (estimate - clean) as the residual noise.
    """
    noise = estimate - clean
    signal_power = np.mean(clean ** 2)
    noise_power  = np.mean(noise ** 2)
    if noise_power < 1e-12:
        return 99.0   # essentially perfect
    return 10.0 * np.log10(signal_power / noise_power)
