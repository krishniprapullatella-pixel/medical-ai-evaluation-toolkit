
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


# PROJECT DIRECTORY
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "medical_ai_predictions.csv")


# READ DATASET
data = pd.read_csv(INPUT_FILE)

print("\nMEDICAL AI DATASET EVALUATOR")
print("--------------------------------")

print("\nOriginal Dataset:")
print(data)


# DATASET VALIDATION

# CHECK EMPTY DATASET
if data.empty:
    raise ValueError("Dataset is empty.")

# CHECK REQUIRED COLUMNS
required_columns = ["image_id", "actual", "predicted"]

if not all(column in data.columns for column in required_columns):
    raise ValueError("Dataset is missing required columns.")

# CHECK MISSING VALUES
if data[required_columns].isnull().any().any():
    raise ValueError("Dataset contains missing values.")

# CHECK DUPLICATE IMAGE IDS
if data["image_id"].duplicated().any():
    raise ValueError("Duplicate image IDs found.")

# CLEAN LABELS
data["actual"] = data["actual"].astype(str).str.strip().str.lower()
data["predicted"] = data["predicted"].astype(str).str.strip().str.lower()

# CHECK VALID LABELS
valid_labels = {"disease", "healthy"}

if not data["actual"].isin(valid_labels).all():
    raise ValueError("Invalid labels in actual column.")

if not data["predicted"].isin(valid_labels).all():
    raise ValueError("Invalid labels in predicted column.")

print("\nDataset validation successful!")


# CALCULATE CONFUSION MATRIX VALUES

tp = ((data["actual"] == "disease") &
      (data["predicted"] == "disease")).sum()

tn = ((data["actual"] == "healthy") &
      (data["predicted"] == "healthy")).sum()

fp = ((data["actual"] == "healthy") &
      (data["predicted"] == "disease")).sum()

fn = ((data["actual"] == "disease") &
      (data["predicted"] == "healthy")).sum()

total = tp + tn + fp + fn


# CALCULATE METRICS

accuracy = ((tp + tn) / total) * 100 if total else None

precision = (tp / (tp + fp)) * 100 if (tp + fp) else None

recall = (tp / (tp + fn)) * 100 if (tp + fn) else None

specificity = (tn / (tn + fp)) * 100 if (tn + fp) else None

if precision is None or recall is None:
    f1_score = None
elif precision + recall == 0:
    f1_score = None
else:
    f1_score = (2 * precision * recall) / (precision + recall)


# DISPLAY EVALUATION REPORT

print("\nMEDICAL AI EVALUATION REPORT")
print("--------------------------------")

print("Total Images:", total)
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

metrics = [
    ("Accuracy", accuracy),
    ("Precision", precision),
    ("Recall", recall),
    ("Specificity", specificity),
    ("F1-score", f1_score)
]

for name, value in metrics:
    if value is None:
        print(f"{name}: Undefined")
    else:
        print(f"{name}: {value:.2f}%")


# CONFUSION MATRIX VISUALIZATION

cm = np.array([
    [tp, fn],
    [fp, tn]
])

if total > 0:

    percentages = (cm / total) * 100

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")

    plt.title("Medical AI Dataset Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks([0, 1], ["Disease", "Healthy"])
    plt.yticks([0, 1], ["Disease", "Healthy"])

    labels = [
        ["True Positive", "False Negative"],
        ["False Positive", "True Negative"]
    ]

    for i in range(2):
        for j in range(2):

            color = "white" if cm[i, j] > cm.max() / 2 else "black"

            plt.text(
                j, i,
                f"{labels[i][j]}\n"
                f"Count: {cm[i, j]}\n"
                f"{percentages[i, j]:.2f}%",
                ha="center",
                va="center",
                fontsize=11,
                color=color
            )

    plt.colorbar()
    plt.tight_layout()

    image_path = os.path.join(
        BASE_DIR,
        "medical_ai_dataset_confusion_matrix.png"
    )

    plt.savefig(image_path, dpi=300, bbox_inches="tight")

    print("\nConfusion matrix saved:", image_path)

    plt.show()


# EXPORT EVALUATION REPORT

report_path = os.path.join(
    BASE_DIR,
    "medical_ai_dataset_report.csv"
)

with open(report_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow(["Metric", "Value"])

    for name, value in metrics:
        if value is None:
            writer.writerow([name, "Undefined"])
        else:
            writer.writerow([name, f"{value:.2f}%"])

    writer.writerow([])
    writer.writerow(["Total Images", total])
    writer.writerow(["True Positives", tp])
    writer.writerow(["True Negatives", tn])
    writer.writerow(["False Positives", fp])
    writer.writerow(["False Negatives", fn])


print("\nEvaluation report exported:", report_path)
print("\nDATASET EVALUATION COMPLETED!")
