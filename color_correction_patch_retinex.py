import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load image
content_path = "color.jpg"
img = Image.open(content_path).convert("RGB")
img_np = np.array(img, dtype="float32") / 255.0  # Normalize to [0, 1]
h, w, _ = img_np.shape

# White Patch Retinex: Find the brightest patch as illuminant
# Option 1: Use the maximum value per channel (simplest)
illuminant = np.max(img_np, axis=(0, 1))  # [R_max, G_max, B_max]

# Option 2: Use a small patch (e.g., top 1% brightest pixels) for robustness
# Flatten image to find brightest pixels
flat_img = img_np.reshape(-1, 3)
bright_idx = np.argsort(np.sum(flat_img, axis=1))[-int(0.01 * h * w):]  # Top 1%
bright_patch = flat_img[bright_idx]
illuminant_patch = np.mean(bright_patch, axis=0)  # Average RGB of bright patch

# Choose illuminant (using patch method for better results)
illuminant = illuminant_patch

# Correct colors
corrected_np = np.zeros_like(img_np)
for c in range(3):  # R, G, B
    corrected_np[:, :, c] = img_np[:, :, c] / illuminant[c]  # Divide by illuminant component
corrected_np = np.clip(corrected_np, 0, 1)  # Ensure [0, 1] range

# Display
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(img_np); plt.title("Original"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(corrected_np); plt.title("Color Corrected (White Patch)"); plt.axis("off")
plt.show()

# Save
Image.fromarray((corrected_np * 255).astype(np.uint8)).save("white_patch_output.jpeg")