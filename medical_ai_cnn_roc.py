
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

# ------------------------------------------------
# 1. PROJECT SETTINGS
# ------------------------------------------------

BASE_DIR = Path(__file__).parent
TEST_DIR = BASE_DIR / "chest_xray" / "test"
MODEL_PATH = BASE_DIR / "medical_ai_cnn.pth"

IMAGE_SIZE = 128
BATCH_SIZE = 16

device = torch.device("cpu")

# ------------------------------------------------
# 2. IMAGE PREPROCESSING
# Must match the CNN training preprocessing
# ------------------------------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ------------------------------------------------
# 3. LOAD TEST DATASET
# ------------------------------------------------

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes:", test_dataset.classes)
print("Test images:", len(test_dataset))

# ------------------------------------------------
# 4. RECREATE THE TRAINED CNN ARCHITECTURE
# ------------------------------------------------

class MedicalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

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


# ------------------------------------------------
# 5. LOAD SAVED MODEL
# ------------------------------------------------

model = MedicalCNN().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("\nTrained CNN model loaded successfully!")

# ------------------------------------------------
# 6. GENERATE PREDICTION PROBABILITIES
# ------------------------------------------------

actual_labels = []
pneumonia_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        pneumonia_scores = probabilities[:, 1]

        actual_labels.extend(labels.numpy())
        pneumonia_probabilities.extend(
            pneumonia_scores.numpy()
        )

# ------------------------------------------------
# 7. CALCULATE ROC CURVE AND AUC
# ------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    actual_labels,
    pneumonia_probabilities,
    pos_label=1
)

roc_auc = auc(fpr, tpr)

print("\nCNN ROC ANALYSIS")
print("==============================")
print(f"ROC AUC Score: {roc_auc:.4f}")
print(f"ROC AUC Percentage: {roc_auc * 100:.2f}%")

# ------------------------------------------------
# 8. PLOT ROC CURVE
# ------------------------------------------------

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"CNN (AUC = {roc_auc:.4f})",
    linewidth=2
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Medical AI CNN - ROC Curve")

plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.savefig(
    BASE_DIR / "medical_ai_cnn_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nROC curve saved successfully!")

# ------------------------------------------------
# 9. EXPORT PROBABILITY REPORT
# ------------------------------------------------

report = pd.DataFrame({
    "actual": actual_labels,
    "pneumonia_probability": pneumonia_probabilities
})

report.to_csv(
    BASE_DIR / "medical_ai_cnn_probability_report.csv",
    index=False
)

print("Probability report saved successfully!")
print("\nCNN ROC ANALYSIS COMPLETED!")
