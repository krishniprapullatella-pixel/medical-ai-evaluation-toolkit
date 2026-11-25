
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, brier_score_loss

st.set_page_config(
    page_title="Medical AI Confidence Analysis",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical AI Confidence & Calibration Analysis")
st.caption("Educational and research evaluation only — not for clinical diagnosis.")

BASE_DIR = Path(__file__).parent

MODEL_FILES = {
    "Custom CNN": "medical_ai_cnn_probability_report.csv",
    "ResNet50": "medical_ai_resnet50_probability_report.csv",
    "EfficientNet-B0": "medical_ai_efficientnet_probability_report.csv",
}

def load_model_data(model_name):
    file_path = BASE_DIR / MODEL_FILES[model_name]

    if not file_path.exists():
        return None, f"File not found: {file_path.name}"

    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    return df, None


def find_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def prepare_data(df):
    actual_col = find_column(
        df,
        ["actual", "actual_label", "true_label", "y_true", "ground_truth", "label"]
    )

    predicted_col = find_column(
        df,
        ["predicted", "prediction", "predicted_label", "y_pred"]
    )

    normal_prob_col = find_column(
        df,
        ["normal_probability", "normal_prob", "prob_normal", "probability_normal"]
    )

    pneumonia_prob_col = find_column(
        df,
        [
            "pneumonia_probability",
            "pneumonia_prob",
            "prob_pneumonia",
            "probability_pneumonia",
            "positive_probability",
            "probability"
        ]
    )

    # Alternative: two probability columns
    if normal_prob_col is None or pneumonia_prob_col is None:
        probability_columns = [
            c for c in df.columns
            if "prob" in c or "confidence" in c
        ]

        if len(probability_columns) >= 2:
            normal_prob_col = probability_columns[0]
            pneumonia_prob_col = probability_columns[1]

    if actual_col is None:
        return None, "Could not identify the actual-label column."

    if normal_prob_col is None or pneumonia_prob_col is None:
        return None, (
            "Could not identify both NORMAL and PNEUMONIA probability columns. "
            "Please send me a screenshot of this CSV's column names."
        )

    actual = df[actual_col].astype(str).str.upper().str.strip()

    # Convert labels to binary:
    # NORMAL = 0
    # PNEUMONIA = 1
    y_true = actual.map(
        lambda x: 1 if "PNEUMONIA" in x else 0
    )

    normal_probability = pd.to_numeric(
        df[normal_prob_col], errors="coerce"
    )

    pneumonia_probability = pd.to_numeric(
        df[pneumonia_prob_col], errors="coerce"
    )

    # Convert percentages to 0-1 if necessary
    if normal_probability.max() > 1:
        normal_probability = normal_probability / 100

    if pneumonia_probability.max() > 1:
        pneumonia_probability = pneumonia_probability / 100

    valid = (
        y_true.notna()
        & normal_probability.notna()
        & pneumonia_probability.notna()
    )

    y_true = y_true[valid].astype(int)
    normal_probability = normal_probability[valid]
    pneumonia_probability = pneumonia_probability[valid]

    # Confidence = probability of the model's predicted class
    predicted = (pneumonia_probability >= 0.5).astype(int)

    confidence = np.where(
        predicted == 1,
        pneumonia_probability,
        normal_probability
    )

    correctness = (predicted == y_true).astype(int)

    return {
        "y_true": y_true.to_numpy(),
        "predicted": predicted.to_numpy(),
        "confidence": confidence,
        "correctness": correctness,
        "pneumonia_probability": pneumonia_probability.to_numpy(),
        "normal_probability": normal_probability.to_numpy()
    }, None


def calculate_ece(confidence, correctness, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece = 0
    bin_results = []

    for i in range(bins):
        lower = edges[i]
        upper = edges[i + 1]

        if i == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = mask.sum()

        if count > 0:
            avg_confidence = confidence[mask].mean()
            accuracy = correctness[mask].mean()

            ece += (count / len(confidence)) * abs(
                avg_confidence - accuracy
            )

            bin_results.append({
                "Bin": f"{lower:.1f}–{upper:.1f}",
                "Samples": count,
                "Confidence": avg_confidence,
                "Accuracy": accuracy,
                "Gap": abs(avg_confidence - accuracy)
            })

    return ece, pd.DataFrame(bin_results)


# Sidebar
st.sidebar.header("Select Model")

selected_model = st.sidebar.selectbox(
    "Choose a model:",
    list(MODEL_FILES.keys())
)

st.sidebar.info(
    "Confidence is calculated from the model's predicted-class probability."
)

# Load data
df, error = load_model_data(selected_model)

if error:
    st.error(error)
    st.stop()

data, error = prepare_data(df)

if error:
    st.error(error)
    st.write("Available CSV columns:")
    st.write(list(df.columns))
    st.stop()

confidence = data["confidence"]
correctness = data["correctness"]
y_true = data["y_true"]
predicted = data["predicted"]

accuracy = accuracy_score(y_true, predicted)

brier_score = brier_score_loss(
    y_true,
    data["pneumonia_probability"]
)

ece, bin_df = calculate_ece(confidence, correctness)

# Metrics
st.subheader(f"📊 {selected_model} Confidence Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Mean Confidence", f"{confidence.mean():.2%}")
col3.metric("Brier Score", f"{brier_score:.4f}")
col4.metric("Expected Calibration Error", f"{ece:.4f}")

st.divider()

# Reliability diagram
st.subheader("🎯 Reliability Diagram")

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

if not bin_df.empty:
    ax.plot(
        bin_df["Confidence"],
        bin_df["Accuracy"],
        marker="o",
        label="Model"
    )

ax.set_xlabel("Average Confidence")
ax.set_ylabel("Actual Accuracy")
ax.set_title(f"{selected_model} Reliability Diagram")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.markdown("""
### How to interpret this graph

- **Diagonal line:** perfect calibration.
- **Above the line:** the model is more accurate than its confidence suggests.
- **Below the line:** the model is overconfident.
- **ECE:** smaller values generally indicate better calibration.
- **Brier Score:** measures probability prediction error; lower is better.
""")

st.divider()

# Confidence distribution
st.subheader("📈 Confidence Distribution")

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.hist(
    confidence,
    bins=10,
    range=(0, 1)
)

ax2.set_xlabel("Prediction Confidence")
ax2.set_ylabel("Number of Images")
ax2.set_title("Distribution of Model Confidence")
ax2.grid(True)

st.pyplot(fig2)

st.divider()

# Calibration table
st.subheader("📋 Calibration Bin Analysis")

if not bin_df.empty:
    display_df = bin_df.copy()

    for column in ["Confidence", "Accuracy", "Gap"]:
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.2%}"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# Confidence correctness breakdown
st.subheader("🔍 Confidence vs Correctness")

correct_confidence = confidence[correctness == 1]
incorrect_confidence = confidence[correctness == 0]

col1, col2 = st.columns(2)

if len(correct_confidence) > 0:
    col1.metric(
        "Average Confidence — Correct Predictions",
        f"{correct_confidence.mean():.2%}"
    )

if len(incorrect_confidence) > 0:
    col2.metric(
        "Average Confidence — Incorrect Predictions",
        f"{incorrect_confidence.mean():.2%}"
    )

# Download
st.download_button(
    "Download Calibration Table",
    bin_df.to_csv(index=False),
    file_name=f"{selected_model}_calibration_analysis.csv",
    mime="text/csv"
)

st.caption(
    "This analysis is experimental and dataset-specific. "
    "Calibration results do not establish clinical reliability."
)
