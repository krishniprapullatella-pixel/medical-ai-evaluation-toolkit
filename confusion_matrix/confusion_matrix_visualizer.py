
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: GET INPUTS

tp = int(input("Enter True Positives: "))
tn = int(input("Enter True Negatives: "))
fp = int(input("Enter False Positives: "))
fn = int(input("Enter False Negatives: "))


# STEP 2: VALIDATE INPUTS

if min(tp, tn, fp, fn) < 0:
    print("ERROR: Values cannot be negative.")

else:

    # STEP 3: CREATE CONFUSION MATRIX

    cm = np.array([
        [tp, fn],
        [fp, tn]
    ])

    # STEP 4: CALCULATE TOTAL AND PERCENTAGES

    total = cm.sum()

    if total == 0:
        print("ERROR: Total images cannot be zero.")

    else:
        percentages = (cm / total) * 100

        # STEP 5: DISPLAY CONFUSION MATRIX

        plt.imshow(cm, cmap="Blues")

        plt.title("Medical AI Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        plt.xticks([0, 1], ["Disease", "Healthy"])
        plt.yticks([0, 1], ["Disease", "Healthy"])

        # STEP 6: ADD CELL LABELS

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

        # STEP 7: ADD COLOR BAR

        plt.colorbar()

        # STEP 8: ADJUST GRAPH LAYOUT

        plt.tight_layout()

        # STEP 9: SAVE GRAPH AS PNG

        plt.savefig(
            "medical_ai_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        print("\nConfusion matrix saved successfully!")
        print("File: medical_ai_confusion_matrix.png")

        # STEP 10: DISPLAY GRAPH

        plt.show()
