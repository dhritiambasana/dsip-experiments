import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

x = np.array([1, 2, 3, 4])
h = np.array([1, 1, 1])

# Linear convolution
linear = convolve(x, h, mode="full")

# Circular convolution (both sequences padded to length N)
N = max(len(x), len(h))
x_pad = np.pad(x, (0, N - len(x)))
h_pad = np.pad(h, (0, N - len(h)))
circular = np.fft.ifft(np.fft.fft(x_pad) * np.fft.fft(h_pad)).real

print("Linear convolution:", linear)
print("Circular convolution:", np.round(circular, 2))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.stem(np.arange(len(linear)), linear)
plt.title("Linear Convolution")
plt.xlabel("n")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(np.arange(N), circular)
plt.title("Circular Convolution")
plt.xlabel("n")
plt.grid(True)

plt.tight_layout()
plt.show()
