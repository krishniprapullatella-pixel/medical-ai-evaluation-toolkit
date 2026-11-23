
import medmnist
from medmnist import PneumoniaMNIST
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------
# 1. LOAD MEDICAL IMAGE DATASET
# ---------------------------------

print("Loading PneumoniaMNIST dataset...")

train_dataset = PneumoniaMNIST(
    split="train",
    download=True
)

test_dataset = PneumoniaMNIST(
    split="test",
    download=True
)

# ---------------------------------
# 2. DISPLAY DATASET INFORMATION
# ---------------------------------

print("\nDATASET INFORMATION")

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

print("Image shape:", train_dataset.imgs.shape)
print("Label shape:", train_dataset.labels.shape)

print("\nDataset loaded successfully!")

# ---------------------------------
# 3. EXTRACT IMAGES AND LABELS
# ---------------------------------

images = train_dataset.imgs
labels = train_dataset.labels.flatten()

print("\nNumber of images:", len(images))
print("Unique labels:", np.unique(labels))

# ---------------------------------
# 4. DISPLAY SAMPLE MEDICAL IMAGES
# ---------------------------------

plt.figure(figsize=(12, 6))

for i in range(8):

    plt.subplot(2, 4, i + 1)

    plt.imshow(
    images[i],
    cmap="gray",
    interpolation="nearest"
)

    label = int(labels[i])

    class_name = (
        "Normal"
        if label == 0
        else "Pneumonia"
    )

    plt.title(class_name)
    plt.axis("off")

plt.suptitle("PneumoniaMNIST - Sample Chest X-rays")

plt.tight_layout()

plt.savefig(
    "medical_image_samples.png",
    dpi=300
)

plt.show()

print("\nSample images saved successfully!")
