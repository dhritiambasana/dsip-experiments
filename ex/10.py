import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter, maximum_filter

# Load the real image
img = cv2.imread("images/i.png")

if img is None:
    raise FileNotFoundError("Image not found at images/i.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Add salt-and-pepper noise to investigate noise removal
rng = np.random.default_rng(10)
noisy = gray.copy()
noise = rng.random(gray.shape)

noisy[noise < 0.03] = 0
noisy[noise > 0.97] = 255

# Apply non-linear filters
median = cv2.medianBlur(noisy, 5)
minimum = minimum_filter(noisy, size=2)
maximum = maximum_filter(noisy, size=2)

# Display results
images = [
    (gray, "Original Image"),
    (noisy, "Image with Noise"),
    (median, "Median Filter"),
    (minimum, "Minimum Filter"),
    (maximum, "Maximum Filter"),
]

plt.figure(figsize=(12, 8))

for i, (image, title) in enumerate(images, 1):
    plt.subplot(2, 3, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
