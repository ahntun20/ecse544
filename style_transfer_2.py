import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def histogram_match(source, reference):
    """
    Match the histogram of source to reference per channel.
    """
    matched = np.zeros_like(source)
    for c in range(source.shape[2]):  # For each channel (R, G, B)
        # Compute histograms and CDFs
        src_hist, bins = np.histogram(source[:, :, c].flatten(), 256, [0, 1], density=True)
        ref_hist, bins = np.histogram(reference[:, :, c].flatten(), 256, [0, 1], density=True)
        src_cdf = np.cumsum(src_hist) / np.sum(src_hist)
        ref_cdf = np.cumsum(ref_hist) / np.sum(ref_hist)

        # Map source values to reference via CDF
        mapping = np.zeros(256)
        for i in range(256):
            closest = np.argmin(np.abs(src_cdf[i] - ref_cdf))
            mapping[i] = closest / 255.0
        
        # Apply mapping
        matched[:, :, c] = cv2.LUT((source[:, :, c] * 255).astype(np.uint8), (mapping * 255).astype(np.uint8)) / 255.0
    
    return matched

# Load content and style images
content_path = "blurred_image.jpeg"
style_path = "style_image.jpg"
content_img = Image.open(content_path).convert("RGB")
style_img = Image.open(style_path).convert("RGB")

# Resize style image to match content dimensions (optional, for consistency)
content_np = np.array(content_img, dtype="float32") / 255.0
style_np = np.array(style_img.resize(content_img.size, Image.Resampling.LANCZOS), dtype="float32") / 255.0

# Perform histogram matching
styled_np = histogram_match(content_np, style_np)
styled_np = np.clip(styled_np, 0, 1)  # Ensure [0, 1] range

# Display
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(content_np); plt.title("Content"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(style_np); plt.title("Style"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(styled_np); plt.title("Styled (Histogram Matching)"); plt.axis("off")
plt.show()

# Save
Image.fromarray((styled_np * 255).astype(np.uint8)).save("histogram_style_output.jpeg")