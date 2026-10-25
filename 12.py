
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the real image
img = cv2.imread("images/ii.png")

if img is None:
    raise FileNotFoundError("Image not found at images/i.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert grayscale image to binary
_, binary = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

# Define structuring element (kernel)
kernel = np.ones((5, 5), dtype=np.uint8)

# Apply morphological operations
dilated = cv2.dilate(binary, kernel, iterations=1)
eroded = cv2.erode(binary, kernel, iterations=1)

opened = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, kernel
)

closed = cv2.morphologyEx(
    binary, cv2.MORPH_CLOSE, kernel
)

# Display results
images = [
    (gray, "Original Grayscale"),
    (binary, "Binary Image"),
    (dilated, "Dilation"),
    (eroded, "Erosion"),
    (opened, "Opening"),
    (closed, "Closing")
]

plt.figure(figsize=(12, 8))

for i, (image, title) in enumerate(images, 1):
    plt.subplot(2, 3, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
