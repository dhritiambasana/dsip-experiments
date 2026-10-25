import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the real image
img = cv2.imread("images/i.png")

if img is None:
    raise FileNotFoundError("Image not found at images/i.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Fourier transform
F = np.fft.fftshift(np.fft.fft2(gray))

rows, cols = gray.shape
cy, cx = rows // 2, cols // 2

Y, X = np.ogrid[:rows, :cols]
D = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

# Design Butterworth filters
D0 = 30
order = 2

low_pass = 1 / (1 + (D / D0) ** (2 * order))
high_pass = 1 - low_pass

# Apply filters
low_filtered = F * low_pass
high_filtered = F * high_pass

# Inverse Fourier transform
smoothed = np.abs(np.fft.ifft2(np.fft.ifftshift(low_filtered)))

high_component = np.fft.ifft2(np.fft.ifftshift(high_filtered)).real

# Sharpen by adding high-frequency details
sharpened = np.clip(gray.astype(float) + high_component, 0, 255).astype(np.uint8)

smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)

# Display results
images = [
    (gray, "Original Grayscale"),
    (low_pass, "Butterworth Low-Pass Mask"),
    (smoothed, "Smoothed Image"),
    (high_pass, "Butterworth High-Pass Mask"),
    (np.abs(high_component), "High-Frequency Components"),
    (sharpened, "Sharpened Image"),
]

plt.figure(figsize=(12, 8))

for i, (image, title) in enumerate(images, 1):
    plt.subplot(2, 3, i)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
