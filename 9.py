import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the real image
img = cv2.imread("images/i.png")

if img is None:
    raise FileNotFoundError("Image not found at images/i.png")

# Convert BGR to RGB for displaying correctly
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 1. Average filter (smoothing)
average = cv2.blur(gray, (5, 5))

# 2. Gaussian filter (smoothing)
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Laplacian filter (sharpening)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpened = np.clip(gray.astype(float) - laplacian, 0, 255).astype(np.uint8)

# Display results
images = [
    (img, "Original Image"),
    (gray, "Grayscale Image"),
    (average, "Average Filter"),
    (gaussian, "Gaussian Filter"),
    (sharpened, "Laplacian Sharpening"),
]

plt.figure(figsize=(12, 8))

for i, (image, title) in enumerate(images, 1):
    plt.subplot(2, 3, i)
    if image.ndim == 2:
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(image)

    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
