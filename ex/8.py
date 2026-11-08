import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the real image
img = cv2.imread("images/i.png")

if img is None:
    raise FileNotFoundError("Image not found at images/i.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Histogram equalization
equalized = cv2.equalizeHist(gray)

# Create a reference image for histogram matching
rng = np.random.default_rng(10)
target = np.clip(rng.normal(175, 35, gray.shape), 0, 255).astype(np.uint8)

# Calculate histograms and cumulative distributions
src_hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
tgt_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).ravel()

src_cdf = np.cumsum(src_hist) / src_hist.sum()
tgt_cdf = np.cumsum(tgt_hist) / tgt_hist.sum()

# Map source intensities to target intensities
mapping = np.interp(src_cdf, tgt_cdf, np.arange(256))
lookup = np.clip(mapping, 0, 255).astype(np.uint8)
matched = cv2.LUT(gray, lookup)

# Display images and their histograms
images = [
    (gray, "Original Grayscale"),
    (equalized, "Histogram Equalization"),
    (target, "Reference Image"),
    (matched, "Histogram Matching"),
]

plt.figure(figsize=(12, 8))

for i, (im, title) in enumerate(images):
    plt.subplot(2, 4, i + 1)
    plt.imshow(im, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

    plt.subplot(2, 4, i + 5)
    plt.hist(im.ravel(), bins=256, range=(0, 256))
    plt.title(title + " Histogram")
    plt.xlabel("Gray Level")
    plt.ylabel("Frequency")
    plt.xlim(0, 255)

plt.tight_layout()
plt.show()
