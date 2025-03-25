import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def target_and_enhance(image_np, threshold=0.9, enhance_factor=1.5):
    """
    Identify a bright target region and enhance its contrast.
    """
    # Convert to grayscale for targeting
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Create a binary mask for bright regions (target)
    mask = (gray > threshold).astype(np.uint8)  # 1 where bright, 0 elsewhere
    
    # Enhance contrast in the target region
    enhanced = np.copy(image_np)
    for c in range(3):  # R, G, B
        channel = enhanced[:, :, c]
        # Increase contrast: scale values around mean in target region
        target_pixels = channel[mask == 1]
        if len(target_pixels) > 0:
            mean = np.mean(target_pixels)
            channel[mask == 1] = np.clip(mean + enhance_factor * (target_pixels - mean), 0, 1)
    
    # Blend original and enhanced based on mask
    result = image_np * (1 - mask[..., np.newaxis]) + enhanced * mask[..., np.newaxis]
    return result, mask

# Load image
content_path = "blurred_image.jpeg"
img = Image.open(content_path).convert("RGB")
img_np = np.array(img, dtype="float32") / 255.0  # Normalize to [0, 1]

# Apply targeting and enhancement
styled_np, target_mask = target_and_enhance(img_np, threshold=0.9, enhance_factor=1.5)
styled_np = np.clip(styled_np, 0, 1)

# Display
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(img_np); plt.title("Original"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(target_mask, cmap="gray"); plt.title("Target Mask"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(styled_np); plt.title("Targeted Enhancement"); plt.axis("off")
plt.show()

# Save
Image.fromarray((styled_np * 255).astype(np.uint8)).save("targeted_enhancement_output.jpeg")