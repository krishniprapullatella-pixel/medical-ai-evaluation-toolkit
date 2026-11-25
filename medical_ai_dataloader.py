
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ==========================================
# 1. DATASET LOCATION
# ==========================================

BASE_DIR = Path(__file__).parent / "chest_xray"

TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([

    # Convert images to grayscale
    transforms.Grayscale(num_output_channels=1),

    # Resize all images
    transforms.Resize((224, 224)),

    # Convert images to tensors
    transforms.ToTensor(),

    # Normalize pixel values
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )

])

# ==========================================
# 3. LOAD TRAINING DATASET
# ==========================================

full_train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=transform
)

print("\nDATASET INFORMATION")
print("=" * 45)

print("Classes:", full_train_dataset.classes)
print("Class mapping:", full_train_dataset.class_to_idx)

print("Total training images:", len(full_train_dataset))
print("Total testing images:", len(test_dataset))

# ==========================================
# 4. SPLIT TRAINING DATASET
# ==========================================

train_size = int(0.8 * len(full_train_dataset))
validation_size = len(full_train_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [train_size, validation_size],
    generator=generator
)

print("\nDATASET SPLIT")
print("=" * 45)

print("Training:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Testing:", len(test_dataset))

# ==========================================
# 5. CREATE DATALOADERS
# ==========================================

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# 6. EXAMINE ONE BATCH
# ==========================================

images, labels = next(iter(train_loader))

print("\nFIRST TRAINING BATCH")
print("=" * 45)

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Image data type:", images.dtype)
print("Label data type:", labels.dtype)

print("First 10 labels:", labels[:10].tolist())

print("Minimum pixel value:", images.min().item())
print("Maximum pixel value:", images.max().item())

print("\nDATALOADER CREATED SUCCESSFULLY!")
