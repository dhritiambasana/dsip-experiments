import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1000  # Sampling frequency in Hz
fc = 100  # Cutoff frequency in Hz
order = 4
rp = 1  # Chebyshev passband ripple in dB

# Prewarped analog cutoff frequency (rad/s)
wc = 2 * fs * np.tan(np.pi * fc / fs)

# Analog Butterworth filter
b_b, a_b = signal.butter(order, wc, btype="low", analog=True)

# Analog Chebyshev Type-I filter
b_c, a_c = signal.cheby1(order, rp, wc, btype="low", analog=True)

# Bilinear transformation: analog to digital
b_bz, a_bz = signal.bilinear(b_b, a_b, fs=fs)
b_cz, a_cz = signal.bilinear(b_c, a_c, fs=fs)

# Frequency responses
f1, H_b = signal.freqz(b_bz, a_bz, worN=2048, fs=fs)
f2, H_c = signal.freqz(b_cz, a_cz, worN=2048, fs=fs)

print("Butterworth numerator:", b_bz)
print("Butterworth denominator:", a_bz)
print("Chebyshev numerator:", b_cz)
print("Chebyshev denominator:", a_cz)

plt.figure(figsize=(9, 5))
plt.plot(f1, 20 * np.log10(np.maximum(abs(H_b), 1e-10)), label="Butterworth")
plt.plot(f2, 20 * np.log10(np.maximum(abs(H_c), 1e-10)), label="Chebyshev Type I")
plt.axvline(fc, color="gray", linestyle="--", label="Cutoff")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Low-Pass Filter Frequency Responses")
plt.ylim(-80, 5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
