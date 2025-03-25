import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import io, restoration

# Load a custom color image (no grayscale conversion)
image_path = "blurred_image.jpeg"  # Change this to your image file
image = io.imread(image_path)  # Load as RGB (assumes 3 channels)

# Ensure image is in float format [0, 1] for processing
image = image.astype(float) / 255.0

# Define a Point Spread Function (PSF) - A simple 5x5 uniform blur
psf = np.ones((5, 5)) / 25

# Apply blur to each RGB channel separately
blurred_image = np.zeros_like(image)
for channel in range(image.shape[2]):  # Loop over R, G, B channels
    blurred_image[:, :, channel] = conv2(image[:, :, channel], psf, 'same')

# Add Poisson noise to each channel
rng = np.random.default_rng()
noisy_image = blurred_image.copy()
for channel in range(image.shape[2]):
    noise = (rng.poisson(lam=25, size=blurred_image[:, :, channel].shape) - 10) / 255.0
    noisy_image[:, :, channel] += noise
noisy_image = np.clip(noisy_image, 0, 1)  # Keep values in valid range

# Deblur each channel using Richardson-Lucy deconvolution
num_iterations = 30  # Adjust for better results
deblurred_image = np.zeros_like(noisy_image)
for channel in range(noisy_image.shape[2]):
    deblurred_image[:, :, channel] = restoration.richardson_lucy(
        noisy_image[:, :, channel], psf, num_iter=num_iterations
    )

# Display the results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

titles = ["Blurred Image (Original)", "Noisy Data", "Deblurred (Richardson-Lucy)"]
images = [blurred_image, noisy_image, deblurred_image]

for i in range(3):
    ax[i].imshow(images[i])  # No vmin/vmax needed for RGB; auto-scales [0, 1]
    ax[i].set_title(titles[i])
    ax[i].axis('off')

plt.tight_layout()
plt.show()

# Optional: Save the color deblurred image
io.imsave("deblurred_color_output.jpeg", (deblurred_image * 255).astype(np.uint8))