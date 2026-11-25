# medical_ai_efficientnet.py

from pathlib import Path
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# =========================================================
# 1. CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "chest_xray"

MODEL_NAME = "EfficientNet-B0"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_WORKERS = 0

DEVICE = torch.device("cpu")

torch.set_num_threads(12)

print("=" * 60)
print("MEDICAL AI - EFFICIENTNET-B0 TRAINING")
print("=" * 60)

print("Device:", DEVICE)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)


# =========================================================
# 2. IMAGE TRANSFORMATIONS
# =========================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 3. LOAD DATASET
# =========================================================

train_dataset_full = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform
)

classes = train_dataset_full.classes

print("\nClasses:", classes)
print("Class mapping:", train_dataset_full.class_to_idx)
print("Total training images:", len(train_dataset_full))
print("Total test images:", len(test_dataset))


# =========================================================
# 4. TRAIN / VALIDATION SPLIT
# =========================================================

train_size = int(0.80 * len(train_dataset_full))
validation_size = len(train_dataset_full) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, validation_dataset = random_split(
    train_dataset_full,
    [train_size, validation_size],
    generator=generator
)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(validation_dataset))


# =========================================================
# 5. DATA LOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# =========================================================
# 6. CLASS WEIGHTS
# =========================================================

train_indices = train_dataset.indices

train_labels = [
    train_dataset_full.targets[index]
    for index in train_indices
]

class_counts = np.bincount(train_labels)

class_weights = len(train_labels) / (
    len(class_counts) * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("\nTraining class counts:", class_counts)
print("Class weights:", class_weights.tolist())


# =========================================================
# 7. LOAD PRETRAINED EFFICIENTNET-B0
# =========================================================

print("\nLoading pretrained EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=weights)


# =========================================================
# 8. FREEZE PRETRAINED FEATURES
# =========================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# =========================================================
# 9. REPLACE CLASSIFICATION LAYER
# =========================================================

input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    len(classes)
)

model = model.to(DEVICE)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel:", MODEL_NAME)
print("Total parameters:", total_parameters)
print("Trainable parameters:", trainable_parameters)
print("Trainable layer: final classification layer only")


# =========================================================
# 10. LOSS FUNCTION AND OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE)
)

optimizer = optim.Adam(
    model.classifier[1].parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# 11. TRAINING FUNCTION
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# =========================================================
# 12. VALIDATION FUNCTION
# =========================================================

def validate_model(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    validation_loss = running_loss / total
    validation_accuracy = correct / total

    return validation_loss, validation_accuracy


# =========================================================
# 13. TRAINING LOOP
# =========================================================

best_validation_loss = float("inf")

best_model_weights = copy.deepcopy(
    model.state_dict()
)

training_start_time = time.time()

print("\nStarting EfficientNet-B0 training...")

for epoch in range(EPOCHS):

    epoch_start_time = time.time()

    print("\n" + "=" * 45)

    print(f"Epoch {epoch + 1}/{EPOCHS}")

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    validation_loss, validation_accuracy = validate_model(
        model,
        validation_loader,
        criterion
    )

    epoch_time = time.time() - epoch_start_time

    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

    print(f"Epoch Time: {epoch_time:.1f} seconds")

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_model_weights,
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "model_name": MODEL_NAME
            },
            PROJECT_DIR / "medical_ai_efficientnet.pth"
        )

        print("Best model checkpoint saved!")


# =========================================================
# 14. LOAD BEST MODEL
# =========================================================

model.load_state_dict(best_model_weights)

total_training_time = time.time() - training_start_time

print("\nTraining completed.")

print(
    f"Total training time: "
    f"{total_training_time:.2f} seconds"
)


# =========================================================
# 15. TEST EVALUATION
# =========================================================

print("\nEvaluating EfficientNet-B0 on test dataset...")

model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(labels.numpy())

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


all_labels = np.array(all_labels)

all_predictions = np.array(all_predictions)

all_probabilities = np.array(all_probabilities)


# =========================================================
# 16. CONFUSION MATRIX AND METRICS
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

TN, FP, FN, TP = cm.ravel()

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities[:, 1]
)


# =========================================================
# 17. PRINT RESULTS
# =========================================================

print("\n" + "=" * 55)

print("EFFICIENTNET-B0 TEST RESULTS")

print("=" * 55)

print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)

print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Precision: {precision * 100:.2f}%")

print(f"Recall: {recall * 100:.2f}%")

print(f"Specificity: {specificity * 100:.2f}%")

print(f"F1-Score: {f1 * 100:.2f}%")

print(f"ROC-AUC: {roc_auc:.4f}")


# =========================================================
# 18. SAVE METRICS
# =========================================================

metrics = {
    "Model": MODEL_NAME,
    "TP": TP,
    "TN": TN,
    "FP": FP,
    "FN": FN,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity,
    "F1-Score": f1,
    "ROC-AUC": roc_auc,
    "Training Time Seconds": total_training_time
}

metrics_df = pd.DataFrame([metrics])

metrics_df.to_csv(
    PROJECT_DIR / "medical_ai_efficientnet_metrics.csv",
    index=False
)


# =========================================================
# 19. SAVE PREDICTIONS
# =========================================================

predictions_df = pd.DataFrame({
    "Actual": all_labels,
    "Predicted": all_predictions,
    "Correct": all_labels == all_predictions
})

predictions_df.to_csv(
    PROJECT_DIR / "medical_ai_efficientnet_predictions.csv",
    index=False
)


# =========================================================
# 20. SAVE PROBABILITY REPORT
# =========================================================

probability_df = pd.DataFrame({
    "Actual": all_labels,
    "Normal Probability": all_probabilities[:, 0],
    "Pneumonia Probability": all_probabilities[:, 1],
    "Predicted": all_predictions
})

probability_df.to_csv(
    PROJECT_DIR / "medical_ai_efficientnet_probability_report.csv",
    index=False
)


# =========================================================
# 21. SAVE CONFUSION MATRIX VISUALIZATION
# =========================================================

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("EfficientNet-B0 Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    classes
)

plt.yticks(
    [0, 1],
    classes
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "medical_ai_efficientnet_confusion_matrix.png",
    dpi=300
)

plt.close()


# =========================================================
# 22. SAVE ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probabilities[:, 1]
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"EfficientNet-B0 (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("EfficientNet-B0 ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PROJECT_DIR / "medical_ai_efficientnet_roc_curve.png",
    dpi=300
)

plt.close()


# =========================================================
# 23. COMPLETION MESSAGE
# =========================================================

print("\n" + "=" * 55)

print("EFFICIENTNET-B0 TRAINING AND EVALUATION COMPLETED")

print("=" * 55)

print("Saved model: medical_ai_efficientnet.pth")

print("Saved metrics: medical_ai_efficientnet_metrics.csv")

print("Saved predictions: medical_ai_efficientnet_predictions.csv")

print("Saved probabilities: medical_ai_efficientnet_probability_report.csv")

print("Saved confusion matrix: medical_ai_efficientnet_confusion_matrix.png")

print("Saved ROC curve: medical_ai_efficientnet_roc_curve.png")

print("\nNext step: Integrate CNN, ResNet50 and EfficientNet-B0.")