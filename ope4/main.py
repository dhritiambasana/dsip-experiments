
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).parent

AUDIO_DIR = BASE_DIR / "audio"
OUTPUT_DIR = BASE_DIR / "output"

VIDEO_FILE = AUDIO_DIR / "hawayein.mp4"
MP3_FILE = AUDIO_DIR / "hawayein.mp3"

OUTPUT_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. CONVERT MP4 VIDEO TO MP3 AUDIO
# ==================================================

def convert_mp4_to_mp3():

    if not VIDEO_FILE.exists():
        raise FileNotFoundError(
            f"Video file not found: {VIDEO_FILE}"
        )

    command = [
        "ffmpeg",
        "-y",
        "-i", str(VIDEO_FILE),
        "-vn",
        "-codec:a", "libmp3lame",
        "-q:a", "2",
        str(MP3_FILE)
    ]

    subprocess.run(command, check=True)

    print("MP4 successfully converted to MP3.")
    print(f"Saved at: {MP3_FILE}")


# ==================================================
# 3. LOAD MP3 AUDIO AS WAV DATA
# ==================================================

def load_audio():

    wav_file = AUDIO_DIR / "temporary.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(MP3_FILE),
        "-vn",
        "-ac", "2",
        "-ar", "44100",
        "-acodec", "pcm_s16le",
        str(wav_file)
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    sample_rate, audio = wavfile.read(wav_file)

    # Convert integer samples to floating point.
    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float64) / max_value
    else:
        audio = audio.astype(np.float64)

    wav_file.unlink(missing_ok=True)

    return sample_rate, audio


# ==================================================
# 4. SAVE FILTERED AUDIO AS MP3
# ==================================================

def save_audio(audio, sample_rate, filename):

    wav_path = OUTPUT_DIR / "temporary.wav"
    mp3_path = OUTPUT_DIR / filename

    # Prevent clipping.
    audio = np.clip(audio, -1.0, 1.0)

    wavfile.write(
        wav_path,
        sample_rate,
        (audio * 32767).astype(np.int16)
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        "-q:a", "2",
        str(mp3_path)
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    wav_path.unlink(missing_ok=True)

    print(f"Saved: {mp3_path}")


# ==================================================
# 5. LOW-PASS BUTTERWORTH FILTER
# ==================================================

def low_pass_filter(audio, sample_rate, cutoff):

    nyquist = sample_rate / 2

    normalized_cutoff = cutoff / nyquist

    sos = butter(
        N=6,
        Wn=normalized_cutoff,
        btype="lowpass",
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        audio,
        axis=0
    )

    return filtered


# ==================================================
# 6. BAND-PASS BUTTERWORTH FILTER
# ==================================================

def band_pass_filter(
    audio,
    sample_rate,
    low_cutoff,
    high_cutoff
):

    nyquist = sample_rate / 2

    sos = butter(
        N=6,
        Wn=[
            low_cutoff / nyquist,
            high_cutoff / nyquist
        ],
        btype="bandpass",
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        audio,
        axis=0
    )

    return filtered


# ==================================================
# 7. BAND-STOP BUTTERWORTH FILTER
# ==================================================

def band_stop_filter(
    audio,
    sample_rate,
    low_cutoff,
    high_cutoff
):

    nyquist = sample_rate / 2

    sos = butter(
        N=6,
        Wn=[
            low_cutoff / nyquist,
            high_cutoff / nyquist
        ],
        btype="bandstop",
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        audio,
        axis=0
    )

    return filtered


# ==================================================
# 8. PART (a): LOW-PASS FILTERING
# ==================================================

def perform_low_pass_experiment(audio, sample_rate):

    print("\nPART (a): LOW-PASS FILTERING")

    cutoff_frequencies = [
        500,
        1500,
        3000
    ]

    for cutoff in cutoff_frequencies:

        filtered = low_pass_filter(
            audio,
            sample_rate,
            cutoff
        )

        filename = f"lowpass_{cutoff}Hz.mp3"

        save_audio(
            filtered,
            sample_rate,
            filename
        )

        print(
            f"Applied low-pass filter: {cutoff} Hz"
        )


# ==================================================
# 9. PART (b): VOCAL AND BACKGROUND MUSIC FILTERING
# ==================================================

def perform_band_filtering(audio, sample_rate):

    print("\nPART (b): BAND FILTERING")

    # Approximate vocal frequency range.
    vocal_low = 300
    vocal_high = 3400

    # 1. Retain the approximate vocal frequency band.
    # Frequencies outside this range are attenuated.

    vocals_emphasized = band_pass_filter(
        audio,
        sample_rate,
        vocal_low,
        vocal_high
    )

    save_audio(
        vocals_emphasized,
        sample_rate,
        "vocals_emphasized.mp3"
    )

    # 2. Suppress the approximate vocal frequency band.
    # This retains lower and higher frequency components.

    vocals_suppressed = band_stop_filter(
        audio,
        sample_rate,
        vocal_low,
        vocal_high
    )

    save_audio(
        vocals_suppressed,
        sample_rate,
        "vocals_suppressed.mp3"
    )


# ==================================================
# 10. MAIN FUNCTION
# ==================================================

def main():

    print("audio processing...")

    # Convert the original video into MP3.
    convert_mp4_to_mp3()

    # Load extracted audio.
    sample_rate, audio = load_audio()

    print(f"\nSample rate: {sample_rate} Hz")
    print(f"Audio shape: {audio.shape}")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

    # Perform both experiments.
    perform_low_pass_experiment(
        audio,
        sample_rate
    )

    perform_band_filtering(
        audio,
        sample_rate
    )

    print("\ncompleted successfully!")


if __name__ == "__main__":
    main()
