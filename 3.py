import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4])
y = np.array([2, 1, 0, 1])

# Full autocorrelation and cross-correlation
auto = np.correlate(x, x, mode="full")
cross = np.correlate(x, y, mode="full")

lags_auto = np.arange(-(len(x) - 1), len(x))
lags_cross = np.arange(-(len(y) - 1), len(x))

print("Autocorrelation:", auto)
print("Cross-correlation:", cross)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.stem(lags_auto, auto)
plt.title("Autocorrelation")
plt.xlabel("Lag")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(lags_cross, cross)
plt.title("Cross-correlation")
plt.xlabel("Lag")
plt.grid(True)

plt.tight_layout()
plt.show()
