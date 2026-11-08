import subprocess
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfreqz, spectrogram
from scipy.io import wavfile

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

FILES = {
    "Original": AUDIO_DIR / "hawayen.mp3",
    "Low-pass 500 Hz": OUTPUT_DIR / "lowpass_500Hz.mp3",
    "Low-pass 1500 Hz": OUTPUT_DIR / "lowpass_1500Hz.mp3",
    "Low-pass 3000 Hz": OUTPUT_DIR / "lowpass_3000Hz.mp3",
    "Vocals emphasized": OUTPUT_DIR / "vocals_emphasized.mp3",
    "Vocals suppressed": OUTPUT_DIR / "vocals_suppressed.mp3",
}

SAMPLE_RATE = 22050


# ==========================================
# LOAD MP3 USING FFMPEG
# ==========================================


def load_mp3(path):

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]

    result = subprocess.run(command, capture_output=True, check=True)

    return np.frombuffer(result.stdout, dtype=np.float32).copy()


# ==========================================
# 1. WAVEFORM COMPARISON
# ==========================================


def plot_waveforms():

    fig, axes = plt.subplots(3, 2, figsize=(15, 10))

    axes = axes.flatten()

    for ax, (name, path) in zip(axes, FILES.items()):

        if not path.exists():
            ax.set_title(f"{name} - Missing")
            continue

        audio = load_mp3(path)

        # Display the first 5 seconds.
        samples = audio[: SAMPLE_RATE * 5]

        time = np.arange(len(samples)) / SAMPLE_RATE

        ax.plot(time, samples, linewidth=0.5)

        ax.set_title(name)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    fig.suptitle("Waveform Comparison of Original and Filtered Audio", fontsize=15)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "waveform_comparison.png", dpi=200)

    plt.close()


# ==========================================
# 2. SPECTROGRAM COMPARISON
# ==========================================


def plot_spectrograms():

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    axes = axes.flatten()

    for ax, (name, path) in zip(axes, FILES.items()):

        if not path.exists():
            ax.set_title(f"{name} - Missing")
            continue

        audio = load_mp3(path)

        frequencies, times, Sxx = spectrogram(
            audio, fs=SAMPLE_RATE, nperseg=2048, noverlap=1536
        )

        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        mesh = ax.pcolormesh(times, frequencies, Sxx_db, shading="auto")

        ax.set_ylim(0, 8000)

        ax.set_title(name)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Frequency (Hz)")

        fig.colorbar(mesh, ax=ax, label="Power (dB)")

    fig.suptitle("Spectrogram Comparison of Filtered Audio", fontsize=15)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "spectrogram_comparison.png", dpi=200)

    plt.close()


# ==========================================
# 3. FILTER FREQUENCY RESPONSE
# ==========================================


def plot_filter_responses():

    fs = 44100
    nyquist = fs / 2

    plt.figure(figsize=(12, 7))

    # Low-pass filters
    for cutoff in [500, 1500, 3000]:

        sos = butter(6, cutoff / nyquist, btype="lowpass", output="sos")

        frequencies, response = sosfreqz(sos, worN=4096, fs=fs)

        magnitude = 20 * np.log10(np.maximum(np.abs(response), 1e-10))

        plt.plot(frequencies, magnitude, label=f"Low-pass {cutoff} Hz")

    # Vocal band-pass
    for filter_type in ["bandpass", "bandstop"]:

        sos = butter(
            6, [300 / nyquist, 3400 / nyquist], btype=filter_type, output="sos"
        )

        frequencies, response = sosfreqz(sos, worN=4096, fs=fs)

        magnitude = 20 * np.log10(np.maximum(np.abs(response), 1e-10))

        plt.plot(frequencies, magnitude, label=f"Vocal range {filter_type}")

    plt.title("Frequency Response of Designed Filters")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")

    plt.xlim(0, 8000)
    plt.ylim(-100, 5)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "filter_frequency_response.png", dpi=200)

    plt.close()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("Generating audio visualizations...")

    plot_waveforms()
    plot_spectrograms()
    plot_filter_responses()

    print("\nAll visualizations generated!")
