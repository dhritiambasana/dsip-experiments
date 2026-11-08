import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10, 11)

# Discrete-time sequences
impulse = (n == 0).astype(int)
step = (n >= 0).astype(int)
ramp = n * step
exponential = (0.8**n) * step
sinusoidal = np.sin(0.2 * np.pi * n)

sequences = [
    (impulse, "Unit Impulse"),
    (step, "Unit Step"),
    (ramp, "Unit Ramp"),
    (exponential, "Exponential"),
    (sinusoidal, "Sinusoidal"),
]

plt.figure(figsize=(12, 9))

for i, (x, title) in enumerate(sequences, 1):
    plt.subplot(3, 2, i)
    plt.stem(n, x)
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid(True)

plt.tight_layout()
plt.show()
