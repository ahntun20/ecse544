import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage import io, util
import matplotlib.pyplot as plt

# Load image
image_path = "blurred_image.jpeg"
image = io.imread(image_path).astype(float) / 255.0  # RGB, [0, 1]

# Simulate noise (skip if your image is noisy)
noisy_gaussian = util.random_noise(image, mode="gaussian", var=0.02)  # Gaussian noise
noisy_sp = util.random_noise(image, mode="s&p", amount=0.05)  # Salt-and-pepper noise

# Gaussian filtering
sigma = 1.5  # Standard deviation of Gaussian kernel (controls smoothing)
denoised_gaussian = np.zeros_like(noisy_gaussian)
for channel in range(image.shape[2]):
    denoised_gaussian[:, :, channel] = gaussian_filter(noisy_gaussian[:, :, channel], sigma=sigma)

# Median filtering
kernel_size = 3  # 3x3 window (adjustable)
denoised_median = np.zeros_like(noisy_sp)
for channel in range(image.shape[2]):
    denoised_median[:, :, channel] = median_filter(noisy_sp[:, :, channel], size=kernel_size)

# Clip to valid range
denoised_gaussian = np.clip(denoised_gaussian, 0, 1)
denoised_median = np.clip(denoised_median, 0, 1)

# Display results
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Original")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(noisy_gaussian)
plt.title("Noisy (Gaussian)")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(denoised_gaussian)
plt.title(f"Denoised (Gaussian, σ={sigma})")
plt.axis("off")
plt.subplot(2, 3, 5)
plt.imshow(noisy_sp)
plt.title("Noisy (Salt & Pepper)")
plt.axis("off")
plt.subplot(2, 3, 6)
plt.imshow(denoised_median)
plt.title(f"Denoised (Median, {kernel_size}x{kernel_size})")
plt.axis("off")
plt.tight_layout()
plt.show()

# Save outputs
io.imsave("denoised_gaussian_output.jpeg", (denoised_gaussian * 255).astype(np.uint8))
io.imsave("denoised_median_output.jpeg", (denoised_median * 255).astype(np.uint8))