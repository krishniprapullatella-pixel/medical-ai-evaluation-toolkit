
from pathlib import Path
import csv
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "chest_xray"

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device("cpu")

print("Training device:", device)

# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================
# 3. LOAD DATASETS
# ==========================================

full_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=transform
)

print("\nClasses:", full_dataset.classes)
print("Training images:", len(full_dataset))
print("Testing images:", len(test_dataset))

if full_dataset.classes != ["NORMAL", "PNEUMONIA"]:
    raise ValueError("Unexpected class order. Check dataset folders.")

# Split training data into training and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
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
# 4. BUILD CNN MODEL
# ==========================================

class MedicalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Convolution block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = MedicalCNN().to(device)

print("\nCNN MODEL CREATED")
print(model)

# ==========================================
# 5. HANDLE CLASS IMBALANCE
# ==========================================

train_labels = [
    full_dataset.targets[i]
    for i in train_dataset.indices
]

class_counts = torch.bincount(
    torch.tensor(train_labels),
    minlength=2
).float()

class_weights = len(train_labels) / (
    2 * class_counts.clamp(min=1)
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================================
# 6. VALIDATION FUNCTION
# ==========================================

def validate(loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total


# ==========================================
# 7. TRAIN CNN
# ==========================================

best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())

print("\nSTARTING CNN TRAINING")
print("=" * 45)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total

    val_loss, val_accuracy = validate(val_loader)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

# Restore best validation checkpoint
model.load_state_dict(best_state)

# ==========================================
# 8. SAVE TRAINED MODEL
# ==========================================

model_path = BASE_DIR / "medical_ai_cnn.pth"

torch.save(model.state_dict(), model_path)

print("\nBest CNN model saved:", model_path)

# ==========================================
# 9. TEST DATASET EVALUATION
# ==========================================

model.eval()

tp = tn = fp = fn = 0

prediction_rows = []

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1).cpu()

        for i in range(len(labels)):

            actual = int(labels[i])
            predicted = int(predictions[i])

            if actual == 1 and predicted == 1:
                tp += 1
            elif actual == 0 and predicted == 0:
                tn += 1
            elif actual == 0 and predicted == 1:
                fp += 1
            else:
                fn += 1

            image_index = (
                batch_index * BATCH_SIZE + i
            )

            image_path, _ = test_dataset.samples[image_index]

            prediction_rows.append([
                Path(image_path).name,
                "PNEUMONIA" if actual == 1 else "NORMAL",
                "PNEUMONIA" if predicted == 1 else "NORMAL"
            ])

# ==========================================
# 10. CALCULATE PERFORMANCE METRICS
# ==========================================

total = tp + tn + fp + fn

accuracy = (tp + tn) / total * 100 if total else 0
precision = tp / (tp + fp) * 100 if (tp + fp) else 0
recall = tp / (tp + fn) * 100 if (tp + fn) else 0
specificity = tn / (tn + fp) * 100 if (tn + fp) else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0
)

print("\nFINAL TEST RESULTS")
print("=" * 45)

print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"Specificity: {specificity:.2f}%")
print(f"F1-Score: {f1:.2f}%")

# ==========================================
# 11. EXPORT PREDICTIONS TO CSV
# ==========================================

csv_path = BASE_DIR / "medical_ai_cnn_predictions.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "image_id",
        "actual",
        "predicted"
    ])

    writer.writerows(prediction_rows)

print("\nPrediction report saved:", csv_path)

print("\nCNN TRAINING AND EVALUATION COMPLETED!")
