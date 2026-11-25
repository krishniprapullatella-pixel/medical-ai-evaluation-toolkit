
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

# =========================================
# 1. FIND A REAL CHEST X-RAY
# =========================================

BASE_DIR = Path(__file__).parent / "chest_xray"

normal_folder = BASE_DIR / "train" / "NORMAL"

image_files = list(normal_folder.glob("*.jpeg"))

if not image_files:
    raise FileNotFoundError(
        "No JPEG images found in the NORMAL folder."
    )

image_path = image_files[0]

# =========================================
# 2. LOAD ORIGINAL IMAGE
# =========================================

original_image = Image.open(image_path)

print("Original image:", image_path.name)
print("Original dimensions:", original_image.size)
print("Original mode:", original_image.mode)

# =========================================
# 3. PREPROCESSING PIPELINE
# =========================================

preprocess = transforms.Compose([

    # Convert to grayscale
    transforms.Grayscale(num_output_channels=1),

    # Resize to 224 x 224
    transforms.Resize((224, 224)),

    # Convert pixels to PyTorch tensor
    transforms.ToTensor()

])

# Apply preprocessing
processed_image = preprocess(original_image)

# =========================================
# 4. NORMALIZATION
# =========================================

normalized_image = processed_image / 1.0

print("\nPROCESSED IMAGE INFORMATION")

print("Tensor shape:", processed_image.shape)
print("Data type:", processed_image.dtype)

print(
    "Minimum pixel value:",
    processed_image.min().item()
)

print(
    "Maximum pixel value:",
    processed_image.max().item()
)

print("Normalized tensor shape:", normalized_image.shape)

# =========================================
# 5. DISPLAY COMPARISON
# =========================================

original_gray = original_image.convert("L")

resized_image = original_gray.resize((224, 224))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_gray, cmap="gray")
axes[0].set_title("Original X-ray")
axes[0].axis("off")

axes[1].imshow(resized_image, cmap="gray")
axes[1].set_title("Resized 224 x 224")
axes[1].axis("off")

axes[2].imshow(normalized_image.squeeze(0), cmap="gray")
axes[2].set_title("Normalized Tensor")
axes[2].axis("off")

plt.tight_layout()

output_path = (
    Path(__file__).parent / "preprocessing_comparison.png"
)

plt.savefig(output_path, dpi=200)

plt.show()

print("\nPreprocessing completed!")
print("Saved:", output_path)
