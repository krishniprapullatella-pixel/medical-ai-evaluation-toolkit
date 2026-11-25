
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "chest_xray"
MODEL_PATH = BASE_DIR / "medical_ai_cnn.pth"
OUTPUT_FILE = BASE_DIR / "medical_ai_cnn_probability_report.csv"

IMAGE_SIZE = 128
BATCH_SIZE = 16

device = torch.device("cpu")

print("=" * 60)
print("GENERATING CNN PROBABILITY REPORT")
print("=" * 60)


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
# 3. LOAD TEST DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = test_dataset.classes

print("Classes:", class_names)
print("Test images:", len(test_dataset))


# ==========================================
# 4. CNN ARCHITECTURE
# ==========================================

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


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

model = MedicalCNN().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

model.eval()

print("\nTrained CNN loaded successfully!")


# ==========================================
# 6. GENERATE PROBABILITY PREDICTIONS
# ==========================================

results = []
image_index = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        for i in range(len(images)):

            actual_index = labels[i].item()
            predicted_index = predictions[i].item()

            normal_probability = probabilities[i][0].item()
            pneumonia_probability = probabilities[i][1].item()

            confidence = max(
                normal_probability,
                pneumonia_probability
            )

            image_path = test_dataset.samples[image_index][0]
            image_id = Path(image_path).name

            actual_label = class_names[actual_index]
            predicted_label = class_names[predicted_index]

            results.append({

                "image_id": image_id,

                "actual": actual_label,

                "predicted": predicted_label,

                "normal_probability": normal_probability,

                "pneumonia_probability": pneumonia_probability,

                "confidence": confidence,

                "correct": actual_label == predicted_label

            })

            image_index += 1


# ==========================================
# 7. SAVE CSV REPORT
# ==========================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 60)
print("PROBABILITY REPORT CREATED SUCCESSFULLY!")
print("=" * 60)

print("\nFile created:")
print(OUTPUT_FILE)

print("\nFirst 10 predictions:")
print(df.head(10).to_string(index=False))

print("\nTotal images evaluated:", len(df))

print(
    "Test accuracy:",
    f"{df['correct'].mean() * 100:.2f}%"
)

print(
    "Average confidence:",
    f"{df['confidence'].mean() * 100:.2f}%"
)

print("\nCSV columns:")
print(df.columns.tolist())

print("\nDONE!")
