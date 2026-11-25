
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

# =====================================
# 1. DATASET LOCATION
# =====================================

BASE_DIR = Path(__file__).parent / "chest_xray"

normal_folder = BASE_DIR / "train" / "NORMAL"
pneumonia_folder = BASE_DIR / "train" / "PNEUMONIA"

# =====================================
# 2. FIND ACTUAL JPEG IMAGES
# =====================================

normal_images = list(normal_folder.glob("*.jpeg"))
pneumonia_images = list(pneumonia_folder.glob("*.jpeg"))

print("\nMEDICAL IMAGE DATASET")
print("=" * 45)

print("Normal images:", len(normal_images))
print("Pneumonia images:", len(pneumonia_images))

# Check whether images exist

if not normal_images or not pneumonia_images:
    raise FileNotFoundError(
        "Images not found! Check your chest_xray folder structure."
    )

# =====================================
# 3. SELECT SAMPLE IMAGES
# =====================================

random.seed(42)

selected_images = (
    random.sample(normal_images, 4)
    + random.sample(pneumonia_images, 4)
)

# =====================================
# 4. DISPLAY ORIGINAL CHEST X-RAYS
# =====================================

fig, axes = plt.subplots(2, 4, figsize=(16, 9))

for ax, image_path in zip(axes.flat, selected_images):

    with Image.open(image_path) as image:

        print(
            image_path.name,
            "| Original size:",
            image.size,
            "| Format:",
            image.format
        )

        ax.imshow(image.convert("L"), cmap="gray")

        ax.set_title(
            f"{image_path.parent.name}\n"
            f"{image.width} x {image.height} pixels"
        )

        ax.axis("off")

plt.suptitle(
    "Real Chest X-ray Images",
    fontsize=18
)

plt.tight_layout()

# =====================================
# 5. SAVE IMAGE SAMPLES
# =====================================

output_path = (
    Path(__file__).parent / "real_chest_xray_samples.png"
)

plt.savefig(output_path, dpi=200)

plt.show()

print("\nImage inspection completed!")
print("Image samples saved at:", output_path)
