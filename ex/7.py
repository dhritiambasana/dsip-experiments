import numpy as np
import matplotlib.pyplot as plt

# Generate a sample grayscale image
y, x = np.indices((256, 256))
img = ((x + y) / 2).astype(np.uint8)

# Gray-level transformations
negative = 255 - img

log_img = (255 * np.log1p(img.astype(float)) / np.log(256)).astype(np.uint8)

gamma = 0.5
gamma_img = (255 * (img / 255.0) ** gamma).astype(np.uint8)

threshold_img = np.where(img >= 128, 255, 0).astype(np.uint8)

images = [
    (img, "Original"),
    (negative, "Image Negative"),
    (log_img, "Log Transformation"),
    (gamma_img, "Gamma Correction"),
    (threshold_img, "Thresholding"),
]

plt.figure(figsize=(10, 7))

for i, (image, title) in enumerate(images, 1):
    plt.subplot(2, 3, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
