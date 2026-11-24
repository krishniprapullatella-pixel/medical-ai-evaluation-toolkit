
from pathlib import Path
import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "chest_xray"

MODEL_PATH = BASE_DIR / "medical_ai_resnet50.pth"

PREDICTIONS_PATH = (
    BASE_DIR / "medical_ai_resnet50_predictions.csv"
)

PROBABILITY_PATH = (
    BASE_DIR / "medical_ai_resnet50_probability_report.csv"
)

CONFUSION_PATH = (
    BASE_DIR / "medical_ai_resnet50_confusion_matrix.png"
)

ROC_PATH = BASE_DIR / "medical_ai_resnet50_roc_curve.png"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 0.001
RANDOM_SEED = 42
NUM_WORKERS = 0

torch.set_num_threads(12)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cpu")

print("=" * 55)
print("RESNET50 MEDICAL AI TRAINING")
print("=" * 55)

print("Device:", DEVICE)
print("CPU threads:", torch.get_num_threads())


# ============================================================
# 2. CHECK DATASET
# ============================================================

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

if not TRAIN_DIR.exists() or not TEST_DIR.exists():

    raise FileNotFoundError(
        "chest_xray/train and chest_xray/test folders "
        "were not found."
    )


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

image_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),

])


# ============================================================
# 4. LOAD DATASET
# ============================================================

full_train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=image_transform,
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=image_transform,
)

print("\nClasses:", full_train_dataset.classes)
print("Class mapping:", full_train_dataset.class_to_idx)

if full_train_dataset.classes != test_dataset.classes:

    raise ValueError(
        "Training and test dataset classes do not match."
    )

if len(full_train_dataset) == 0 or len(test_dataset) == 0:

    raise ValueError("Dataset is empty.")


# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

validation_size = int(
    0.20 * len(full_train_dataset)
)

training_size = (
    len(full_train_dataset) - validation_size
)

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, validation_dataset = random_split(
    full_train_dataset,
    [training_size, validation_size],
    generator=generator,
)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Test images:", len(test_dataset))


# ============================================================
# 6. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


# ============================================================
# 7. LOAD PRETRAINED RESNET50
# ============================================================

print("\nLoading pretrained ResNet50 weights...")

weights = ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)

# Freeze pretrained feature extractor.

for parameter in model.parameters():

    parameter.requires_grad = False

# Replace final layer for two-class classification.

input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    2,
)

model = model.to(DEVICE)

print("ResNet50 loaded successfully.")
print("Trainable parameters: final classification layer only.")


# ============================================================
# 8. HANDLE CLASS IMBALANCE
# ============================================================

class_counts = np.bincount(
    np.array(full_train_dataset.targets)[
        train_dataset.indices
    ],
    minlength=2,
)

if np.any(class_counts == 0):

    raise ValueError(
        "A class is missing from the training split."
    )

class_weights = (
    class_counts.sum() /
    (2 * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
).to(DEVICE)

print("\nTraining class counts:", class_counts)
print("Class weights:", class_weights.tolist())


# ============================================================
# 9. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 10. TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return (
        total_loss / total,
        correct / total * 100,
    )


# ============================================================
# 11. VALIDATION FUNCTION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        total_loss / total,
        correct / total * 100,
    )


# ============================================================
# 12. MODEL TRAINING
# ============================================================

best_validation_loss = float("inf")

best_model_weights = copy.deepcopy(
    model.state_dict()
)

print("\nStarting ResNet50 training...")

training_start = time.time()

for epoch in range(EPOCHS):

    epoch_start = time.time()

    train_loss, train_accuracy = train_one_epoch()

    validation_loss, validation_accuracy = validate()

    print("\n" + "=" * 45)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss: {train_loss:.4f}"
    )

    print(
        f"Training Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )

    print(
        f"Epoch Time: {time.time() - epoch_start:.1f} seconds"
    )

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_model_weights,
                "classes": full_train_dataset.classes,
                "image_size": IMAGE_SIZE,
            },
            MODEL_PATH,
        )

        print("Best model checkpoint saved.")


print("\nTraining completed.")

print(
    "Total training time:",
    round(time.time() - training_start, 2),
    "seconds",
)


# ============================================================
# 13. LOAD BEST MODEL
# ============================================================

model.load_state_dict(best_model_weights)

model.eval()


# ============================================================
# 14. TEST DATASET EVALUATION
# ============================================================

print("\nEvaluating ResNet50 on test dataset...")

all_actual = []
all_predictions = []
all_probabilities = []
all_image_ids = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = outputs.argmax(dim=1)

        all_actual.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy().tolist()
        )


# Preserve test image order from ImageFolder.

all_image_ids = [
    Path(path).name
    for path, label in test_dataset.samples
]

actual = np.array(all_actual)
predicted = np.array(all_predictions)
probability = np.array(all_probabilities)


# ============================================================
# 15. METRIC CALCULATION
# ============================================================

matrix = confusion_matrix(
    actual,
    predicted,
    labels=[0, 1],
)

tn, fp, fn, tp = matrix.ravel()

total = tp + tn + fp + fn

accuracy = (
    (tp + tn) / total * 100
    if total else 0
)

precision = (
    tp / (tp + fp) * 100
    if (tp + fp) else 0
)

recall = (
    tp / (tp + fn) * 100
    if (tp + fn) else 0
)

specificity = (
    tn / (tn + fp) * 100
    if (tn + fp) else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) else 0
)

auc = roc_auc_score(
    actual,
    probability,
) if len(np.unique(actual)) == 2 else float("nan")


# ============================================================
# 16. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 55)
print("RESNET50 TEST RESULTS")
print("=" * 55)

print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"Specificity: {specificity:.2f}%")
print(f"F1-Score: {f1:.2f}%")
print(f"ROC-AUC: {auc:.4f}")


# ============================================================
# 17. SAVE PREDICTION CSV
# ============================================================

prediction_report = pd.DataFrame({

    "image_id": all_image_ids,

    "actual": actual,

    "predicted": predicted,

})

prediction_report.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# 18. SAVE PROBABILITY REPORT
# ============================================================

probability_report = pd.DataFrame({

    "image_id": all_image_ids,

    "actual": actual,

    "predicted": predicted,

    "probability_healthy": 1 - probability,

    "probability_pneumonia": probability,

})

probability_report.to_csv(
    PROBABILITY_PATH,
    index=False,
)


# ============================================================
# 19. SAVE CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(
    matrix,
    cmap="Blues",
)

ax.set_title(
    "ResNet50 Confusion Matrix",
    fontsize=14,
    fontweight="bold",
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Normal", "Pneumonia"])
ax.set_yticklabels(["Normal", "Pneumonia"])

for row in range(2):

    for col in range(2):

        ax.text(
            col,
            row,
            str(matrix[row, col]),
            ha="center",
            va="center",
            fontsize=15,
            color="black",
        )

fig.colorbar(image, ax=ax)

fig.tight_layout()

fig.savefig(
    CONFUSION_PATH,
    dpi=300,
)

plt.close(fig)


# ============================================================
# 20. SAVE ROC CURVE
# ============================================================

if len(np.unique(actual)) == 2:

    fpr, tpr, thresholds = roc_curve(
        actual,
        probability,
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        fpr,
        tpr,
        label=f"ResNet50 AUC = {auc:.4f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title("ResNet50 ROC Curve")

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        ROC_PATH,
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# 21. SAVE METRIC REPORT
# ============================================================

metric_report = pd.DataFrame([
    {
        "Model": "ResNet50",
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-Score": f1,
        "ROC-AUC": auc,
    }
])

metric_report.to_csv(
    BASE_DIR / "medical_ai_resnet50_metrics.csv",
    index=False,
)


# ============================================================
# 22. COMPLETION
# ============================================================

print("\n" + "=" * 55)
print("RESNET50 TRAINING AND EVALUATION COMPLETED")
print("=" * 55)

print("Saved model:", MODEL_PATH.name)
print("Saved predictions:", PREDICTIONS_PATH.name)
print("Saved probabilities:", PROBABILITY_PATH.name)
print("Saved confusion matrix:", CONFUSION_PATH.name)
print("Saved ROC curve:", ROC_PATH.name)

print("\nNext step: Train EfficientNet-B0.")
