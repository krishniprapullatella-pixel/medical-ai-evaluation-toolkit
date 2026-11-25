
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# Load CNN predictions
df = pd.read_csv("medical_ai_cnn_predictions.csv")

# Convert labels into binary values
# NORMAL = 0, PNEUMONIA = 1
y_true = df["actual"]
y_pred = df["predicted"]

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("CNN CONFUSION MATRIX")
print(cm)

# Visualize confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("CNN Confusion Matrix")
plt.savefig("medical_ai_cnn_confusion_matrix.png")
plt.show()

print("Confusion matrix saved!")

# ROC curve needs prediction probabilities.
# This CSV currently contains predicted class labels,
# not probability scores, so ROC/AUC cannot be calculated
# correctly from this file alone.
