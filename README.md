# 🫀 ECG Signal Denoising System

> A DSP-based multi-stage digital filter pipeline to remove real-world noise from ECG recordings — built using core concepts from Signals & Systems (Oppenheim & Willsky).

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6?style=flat-square&logo=scipy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What This Does

Real ECG signals recorded in hospitals are always corrupted by three types of interference. This project:

1. **Simulates** a clean synthetic ECG (PQRST waveform model)
2. **Corrupts** it with three clinically realistic noise types
3. **Recovers** the clean signal through a cascaded Butterworth + Notch filter pipeline
4. **Quantifies** improvement using Signal-to-Noise Ratio (SNR in dB)

---

## Results

### Time-Domain — Every Stage of the Pipeline

![ECG Denoising Pipeline](assets/pipeline.png)

### Frequency-Domain — FFT Before and After

![FFT Spectrum Comparison](assets/spectrum.png)

The 50 Hz powerline spike (clearly visible in the middle plot) is completely eliminated in the denoised signal.

### SNR Improvement

| Stage       | SNR      |
|-------------|----------|
| Noisy ECG   | −3.27 dB |
| Denoised ECG| +8.21 dB |
| Improvement | **+11.48 dB** |

---

## Signal Processing Theory

### The ECG Signal
An electrocardiogram measures the heart's electrical activity. Each beat produces a characteristic **PQRST** wave:
- **P wave** — atrial depolarization (small bump, ~0.15 mV)
- **QRS complex** — ventricular depolarization (the tall spike, ~1.5 mV)
- **T wave** — ventricular repolarization (recovery hump, ~0.35 mV)

The heart beats at ~72 BPM, so the fundamental ECG frequency is 72/60 = **1.2 Hz**, with harmonics up to ~40 Hz.

### Why 500 Hz Sampling Rate?
By the **Nyquist-Shannon Sampling Theorem**, to faithfully capture a signal with content up to 40 Hz, we need at least 80 samples/second. We use 500 Hz to give a comfortable margin and capture sharp QRS features.

### The Three Noise Types

| Noise | Frequency | Source |
|-------|-----------|--------|
| Baseline Wander | 0.1 – 0.5 Hz | Patient breathing |
| Powerline Interference | 50 Hz (India) | Electrical supply |
| EMG / Muscle Artifact | Wideband (random) | Patient movement |

### Why Butterworth Filters?
Butterworth filters have a **maximally flat passband** — no ripple in the frequencies we want to keep. The trade-off is a gradual rolloff compared to Chebyshev or Elliptic filters, but for ECG denoising, flatness in the passband is more important than a steep cutoff.

### Why `filtfilt` and Not `lfilter`?
`lfilter` does a single forward pass and introduces **phase delay** — the filtered signal is shifted in time relative to the input. In ECG, the exact timing of each wave (P–R interval, QRS duration) carries clinical meaning. `filtfilt` does a forward + backward pass, yielding **zero phase distortion**.

### Filter Pipeline

```
Noisy ECG
    │
    ▼
[High-Pass Filter]  cutoff = 0.5 Hz, order 4
    │  → Removes slow baseline wander (< 0.5 Hz)
    ▼
[Notch Filter]      target = 50 Hz, Q = 35
    │  → Removes powerline spike at exactly 50 Hz
    ▼
[Low-Pass Filter]   cutoff = 40 Hz, order 4
    │  → Removes high-frequency EMG noise (> 40 Hz)
    ▼
Denoised ECG
```

### What is SNR (Signal-to-Noise Ratio)?

```
SNR (dB) = 10 × log₁₀( signal_power / noise_power )
```
Positive dB = signal dominates. Negative dB = noise dominates. Every +3 dB roughly doubles the signal-to-noise power ratio.

---

## Project Structure

```
ecg-denoising-dsp/
├── main.py                  ← Entry point — run this
├── requirements.txt
├── .gitignore
├── src/
│   ├── ecg_generator.py     ← Synthetic PQRST waveform model
│   ├── noise_adder.py       ← Three noise types
│   ├── filters.py           ← Butterworth + Notch filters, SNR function
│   └── visualizer.py        ← Plotting (pipeline + FFT spectrum)
├── assets/
│   ├── pipeline.png         ← Time-domain comparison
│   └── spectrum.png         ← FFT comparison
└── output/                  ← Generated plots land here
```

---

## Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ecg-denoising-dsp.git
cd ecg-denoising-dsp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python main.py
```

Plots are saved to `output/`. If you are running on a machine with a display and want interactive windows, open `src/visualizer.py` and comment out line 13:
```python
# matplotlib.use('Agg')   ← comment this out for interactive windows
```

---

## Tech Stack

| Library | Used For |
|---------|----------|
| **NumPy** | Signal generation, FFT array operations |
| **SciPy** | Butterworth / Notch filter design (`butter`, `iirnotch`, `filtfilt`) |
| **Matplotlib** | Time-domain and frequency-domain plots |

---

## References

- Oppenheim, A. V., & Willsky, A. S. — *Signals and Systems* (2nd ed.)
- [PhysioNet](https://physionet.org/) — Open ECG database for real clinical signals
- SciPy Signal Processing docs — [scipy.org/doc/scipy/signal](https://docs.scipy.org/doc/scipy/reference/signal.html)

---

## License

MIT — free to use, modify, and distribute.
