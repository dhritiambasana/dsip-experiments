
import numpy as np
import matplotlib.pyplot as plt

fs = 1000
N = 1000
t = np.arange(N) / fs

# Signal containing 50 Hz and 120 Hz components
x = (np.sin(2*np.pi*50*t)
     + 0.5*np.sin(2*np.pi*120*t))

# FFT and frequency axis
X = np.fft.fft(x)
freq = np.fft.fftfreq(N, d=1/fs)

# IFFT
x_reconstructed = np.fft.ifft(X).real

# Single-sided amplitude spectrum
half = N // 2
amplitude = 2 * np.abs(X[:half]) / N
amplitude[0] /= 2

print("Maximum reconstruction error:",
      np.max(np.abs(x - x_reconstructed)))

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t[:100], x[:100])
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(freq[:half], amplitude)
plt.title("FFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.xlim(0, 200)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t[:100], x_reconstructed[:100])
plt.title("Reconstructed Signal using IFFT")
plt.xlabel("Time (s)")
plt.grid(True)

plt.tight_layout()
plt.show()
