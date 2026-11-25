
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


# PROJECT FOLDER
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# INPUT VALIDATION
def get_valid_count(message):
    while True:
        try:
            value = int(input(message))

            if value < 0:
                print("ERROR: Value cannot be negative.")
            else:
                return value

        except ValueError:
            print("ERROR: Please enter a valid integer.")


# ACCURACY
def calculate_accuracy(tp, tn, fp, fn):
    total = tp + tn + fp + fn

    if total == 0:
        return None

    return ((tp + tn) / total) * 100


# PRECISION
def calculate_precision(tp, fp):
    if tp + fp == 0:
        return None

    return (tp / (tp + fp)) * 100


# RECALL
def calculate_recall(tp, fn):
    if tp + fn == 0:
        return None

    return (tp / (tp + fn)) * 100


# SPECIFICITY
def calculate_specificity(tn, fp):
    if tn + fp == 0:
        return None

    return (tn / (tn + fp)) * 100


# F1 SCORE
def calculate_f1score(precision, recall):
    if precision is None or recall is None:
        return None

    if precision + recall == 0:
        return None

    return (2 * precision * recall) / (precision + recall)


# CONFUSION MATRIX
def display_confusion_matrix(tp, tn, fp, fn):

    cm = np.array([
        [tp, fn],
        [fp, tn]
    ])

    total = cm.sum()

    if total == 0:
        print("Cannot generate matrix: No images available.")
        return

    percentages = (cm / total) * 100

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")

    plt.title("Medical AI Confusion Matrix")
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
        "medical_ai_confusion_matrix.png"
    )

    plt.savefig(image_path, dpi=300, bbox_inches="tight")

    print("\nConfusion matrix saved successfully!")
    print("Location:", image_path)

    plt.show()


# CSV EXPORT
def export_report_to_csv(result, p_result, r_result, s_result, f1_result):

    csv_path = os.path.join(
        BASE_DIR,
        "medical_ai_evaluation_report.csv"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(["Metric", "Value"])

        metrics = [
            ["Accuracy", result],
            ["Precision", p_result],
            ["Recall", r_result],
            ["Specificity", s_result],
            ["F1-score", f1_result]
        ]

        for metric, value in metrics:
            if value is None:
                writer.writerow([metric, "Undefined"])
            else:
                writer.writerow([metric, f"{value:.2f}%"])

    print("\nEvaluation report exported successfully!")
    print("Location:", csv_path)


# MAIN APPLICATION
print("\nMEDICAL AI EVALUATION TOOLKIT")
print("--------------------------------")

tp = get_valid_count("Enter True Positives: ")
tn = get_valid_count("Enter True Negatives: ")
fp = get_valid_count("Enter False Positives: ")
fn = get_valid_count("Enter False Negatives: ")


# CALCULATE METRICS
result = calculate_accuracy(tp, tn, fp, fn)
p_result = calculate_precision(tp, fp)
r_result = calculate_recall(tp, fn)
s_result = calculate_specificity(tn, fp)
f1_result = calculate_f1score(p_result, r_result)


# DISPLAY REPORT
print("\nMEDICAL AI EVALUATION REPORT")
print("--------------------------------")

metrics = [
    ("Accuracy", result),
    ("Precision", p_result),
    ("Recall", r_result),
    ("Specificity", s_result),
    ("F1-score", f1_result)
]

for name, value in metrics:
    if value is None:
        print(f"{name}: Undefined")
    else:
        print(f"{name}: {value:.2f}%")





# EXPORT CSV
export_report_to_csv(
    result,
    p_result,
    r_result,
    s_result,
    f1_result
)

# GENERATE GRAPH
display_confusion_matrix(tp, tn, fp, fn)

print("\nALL FILES GENERATED SUCCESSFULLY!")
