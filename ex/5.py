import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1000
fc = 100
num_taps = 51

# FIR filter using the Hamming window
b = signal.firwin(num_taps, cutoff=fc, window="hamming", fs=fs)

# Frequency response
f, H = signal.freqz(b, worN=2048, fs=fs)

print("FIR filter coefficients:\n", b)

plt.figure(figsize=(10, 7))

plt.subplot(2, 1, 1)
plt.stem(np.arange(num_taps), b)
plt.title("FIR Filter Impulse Response")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(f, 20 * np.log10(np.maximum(abs(H), 1e-10)))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("FIR Filter Frequency Response")
plt.grid(True)

plt.tight_layout()
plt.show()
