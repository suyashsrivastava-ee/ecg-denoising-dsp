"""
main.py
-------
ECG Signal Denoising System
DSP-based multi-stage Butterworth + Notch filter pipeline.

Run:
    python main.py

Outputs (saved to output/ folder):
    pipeline.png  — time-domain view of each denoising stage
    spectrum.png  — FFT comparison (clean vs noisy vs denoised)
"""

import sys
import os

# Make sure imports work regardless of where you run this from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ecg_generator import generate_ecg
from src.noise_adder   import corrupt_ecg
from src.filters       import run_denoising_pipeline, compute_snr_db
from src.visualizer    import plot_pipeline, plot_spectrum, print_report


# ── Parameters ───────────────────────────────────────────────────────────────

FS          = 500    # Sampling frequency (Hz)  — 500 samples every second
DURATION    = 10     # Signal duration (seconds)
HEART_RATE  = 72     # Beats per minute
OUTPUT_DIR  = 'output'


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print()
    print("━" * 50)
    print("   ECG Signal Denoising System")
    print("   DSP-Based Multi-Stage Filter Pipeline")
    print("━" * 50)
    print(f"\n  Fs = {FS} Hz  |  Duration = {DURATION}s  |  HR = {HEART_RATE} BPM")

    # 1. Generate a clean synthetic ECG
    print("\n[1/4]  Generating synthetic ECG signal...")
    t, clean_ecg = generate_ecg(duration=DURATION, fs=FS, heart_rate=HEART_RATE)
    print(f"       {len(t)} samples generated  ({DURATION * HEART_RATE // 60} beats)")

    # 2. Corrupt it with realistic noise
    print("\n[2/4]  Adding noise:")
    print("       • Baseline wander  (0.25 Hz,  ~0.40 mV amplitude)")
    print("       • Powerline noise  (50 Hz,    ~0.18 mV amplitude)")
    print("       • EMG artifact     (Gaussian, ~0.07 mV std dev)")
    noisy_ecg = corrupt_ecg(clean_ecg, t, FS)

    # 3. Run denoising pipeline
    print("\n[3/4]  Running denoising pipeline:")
    print("       Stage 1 → High-pass filter   (cutoff = 0.5 Hz,  order = 4)")
    print("       Stage 2 → Notch filter        (target = 50 Hz,   Q = 35)")
    print("       Stage 3 → Low-pass filter     (cutoff = 40 Hz,  order = 4)")
    after_hp, after_notch, denoised_ecg = run_denoising_pipeline(noisy_ecg, FS)

    # 4. Report SNR improvement
    print_report(clean_ecg, noisy_ecg, denoised_ecg, compute_snr_db)

    # 5. Save plots
    print("[4/4]  Saving plots...")
    p1 = plot_pipeline(t, clean_ecg, noisy_ecg,
                       after_hp, after_notch, denoised_ecg, FS, OUTPUT_DIR)
    p2 = plot_spectrum(clean_ecg, noisy_ecg, denoised_ecg, FS, OUTPUT_DIR)

    print(f"\n  ✓  {p1}")
    print(f"  ✓  {p2}")
    print("\n  Done! Open the output/ folder to see the plots.\n")


if __name__ == '__main__':
    main()
