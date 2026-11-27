from pathlib import Path
from textwrap import dedent

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Custom CNN": "medical_ai_cnn_predictions.csv",
    "ResNet50": "medical_ai_resnet50_predictions.csv",
    "EfficientNet-B0": "medical_ai_efficientnet_predictions.csv",
}

PROB_FILES = {
    "Custom CNN": "medical_ai_cnn_probability_report.csv",
    "ResNet50": "medical_ai_resnet50_probability_report.csv",
    "EfficientNet-B0": "medical_ai_efficientnet_probability_report.csv",
}


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Medical AI Evaluation Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f4f7fb;
}

[data-testid="stSidebar"] {
    background: #111d35;
}

[data-testid="stSidebar"] * {
    color: #edf3ff;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5eaf2;
    padding: 16px;
    border-radius: 13px;
    box-shadow: 0 2px 8px rgba(20,40,70,.04);
}

.main-heading {
    font-family: Manrope, sans-serif;
    color: #17243d;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 2px;
}

.sub-heading {
    color: #718096;
    font-size: .95rem;
}

.hero {
    background: linear-gradient(115deg, #142a49, #205c78);
    border-radius: 17px;
    padding: 28px 32px;
    margin: 18px 0 24px;
    color: white;
}

.hero .eyebrow {
    color: #a9dce9;
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .13em;
    text-transform: uppercase;
}

.hero h1 {
    color: white;
    font-family: Manrope, sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    margin: 9px 0;
}

.hero p {
    color: #dceaf4;
    font-size: .98rem;
    margin-bottom: 0;
}

.section-heading {
    font-family: Manrope, sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #17243d;
    margin: 20px 0 8px;
}

.info-panel {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 13px;
    padding: 18px;
}

.small-note {
    color: #718096;
    font-size: .84rem;
}

footer {
    visibility: hidden;
}
</style>
"""), unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def hero(label, heading, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{label}</div>
            <h1>{heading}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def section(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )


# =========================================================
# RESNET50 MODEL FOR GRAD-CAM
# =========================================================

@st.cache_resource
def load_resnet_for_gradcam():

    checkpoint_path = BASE_DIR / "medical_ai_resnet50.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.name}"
        )

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict):

        checkpoint = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in checkpoint.items()
        }

    model.load_state_dict(checkpoint)

    model.eval()

    return model


# =========================================================
# GRAD-CAM GENERATION
# =========================================================

def create_gradcam(model, tensor):

    activations = {}
    gradients = {}

    target_layer = model.layer4[-1]

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    logits = model(tensor)

    predicted_index = int(
        logits.argmax(dim=1).item()
    )

    model.zero_grad(set_to_none=True)

    logits[0, predicted_index].backward()

    activations_tensor = activations["value"].detach()[0]

    gradients_tensor = gradients["value"].detach()[0]

    weights = gradients_tensor.mean(dim=(1, 2))

    cam = (
        weights[:, None, None] * activations_tensor
    ).sum(dim=0).relu()

    cam = cam - cam.min()

    if cam.max() > 0:
        cam = cam / cam.max()

    cam = torch.nn.functional.interpolate(
        cam[None, None],
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    forward_handle.remove()
    backward_handle.remove()

    probabilities = torch.softmax(
        logits.detach(),
        dim=1
    )[0]

    return (
        predicted_index,
        probabilities.numpy(),
        cam.numpy()
    )


# =========================================================
# GRAD-CAM STREAMLIT PAGE
# =========================================================

def show_gradcam_page():

    hero(
        "EXPLAINABILITY",
        "Grad-CAM Image Analysis",
        "Upload a chest X-ray to visualize the regions that influenced the ResNet50 prediction. Educational use only."
    )

    st.warning(
        "Grad-CAM is an approximate model-activation visualization. "
        "It is not a medical interpretation or diagnostic result."
    )

    uploaded_image = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        key="gradcam_upload"
    )

    if uploaded_image is None:

        st.info("Upload an X-ray image to begin.")

        return

    image = Image.open(uploaded_image).convert("RGB")

    st.subheader("Uploaded image")

    st.image(
        image,
        caption="Uploaded chest X-ray",
        width=400
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image).unsqueeze(0)

    try:

        with st.spinner(
            "Loading ResNet50 and generating Grad-CAM..."
        ):

            model = load_resnet_for_gradcam()

            predicted_index, probabilities, cam = create_gradcam(
                model,
                tensor
            )

        labels = [
            "NORMAL",
            "PNEUMONIA"
        ]

        predicted_class = labels[predicted_index]

        st.subheader(
            f"Model Prediction: {predicted_class}"
        )

        # -------------------------------------------------
        # PROBABILITY METRICS
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        col1.metric(
            "NORMAL probability",
            f"{probabilities[0] * 100:.2f}%"
        )

        col2.metric(
            "PNEUMONIA probability",
            f"{probabilities[1] * 100:.2f}%"
        )

        # -------------------------------------------------
        # VISUALIZATION
        # -------------------------------------------------

        resized_image = image.resize((224, 224))

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(14, 5)
        )

        axes[0].imshow(
            resized_image,
            cmap="gray"
        )

        axes[0].set_title(
            "Original X-ray"
        )

        axes[1].imshow(
            cam,
            cmap="jet"
        )

        axes[1].set_title(
            "Grad-CAM heatmap"
        )

        axes[2].imshow(
            resized_image
        )

        axes[2].imshow(
            cam,
            cmap="jet",
            alpha=0.45
        )

        axes[2].set_title(
            "Grad-CAM overlay"
        )

        for axis in axes:
            axis.axis("off")

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.caption(
            "Grad-CAM highlights regions associated with the model's "
            "prediction. It does not prove that the model focused on "
            "medically meaningful pathology."
        )

    except Exception as error:

        st.error(
            f"Grad-CAM could not be generated: {error}"
        )
        # =========================================================
# CSV FUNCTIONS
# =========================================================

def load_csv(filename):

    path = BASE_DIR / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except Exception as error:

        st.warning(
            f"Could not load {filename}: {error}"
        )

        return None


def find_col(df, candidates):

    columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in columns:

            return columns[candidate.lower()]

    return None


def normalize_labels(values):

    aliases = {
        "0": "normal",
        "0.0": "normal",
        "normal": "normal",
        "healthy": "normal",
        "negative": "normal",

        "1": "pneumonia",
        "1.0": "pneumonia",
        "pneumonia": "pneumonia",
        "disease": "pneumonia",
        "positive": "pneumonia"
    }

    return (
        pd.Series(values)
        .reset_index(drop=True)
        .astype(str)
        .str.strip()
        .str.lower()
        .map(aliases)
    )


def extract_predictions(df):

    actual_column = find_col(
        df,
        [
            "actual",
            "true",
            "actual_label",
            "ground_truth",
            "label",
            "target"
        ]
    )

    predicted_column = find_col(
        df,
        [
            "predicted",
            "prediction",
            "predicted_label",
            "cnn_prediction",
            "model_prediction"
        ]
    )

    if actual_column is None or predicted_column is None:

        raise ValueError(
            f"Could not identify actual/predicted columns in: "
            f"{list(df.columns)}"
        )

    actual = normalize_labels(
        df[actual_column]
    )

    predicted = normalize_labels(
        df[predicted_column]
    )

    if actual.isna().any() or predicted.isna().any():

        raise ValueError(
            "Labels could not be normalized. "
            "Expected 0/1 or NORMAL/PNEUMONIA labels."
        )

    return (
        actual_column,
        predicted_column,
        actual,
        predicted
    )


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(actual, predicted):

    actual = normalize_labels(actual)

    predicted = normalize_labels(predicted)

    TP = int(
        (
            (actual == "pneumonia") &
            (predicted == "pneumonia")
        ).sum()
    )

    TN = int(
        (
            (actual == "normal") &
            (predicted == "normal")
        ).sum()
    )

    FP = int(
        (
            (actual == "normal") &
            (predicted == "pneumonia")
        ).sum()
    )

    FN = int(
        (
            (actual == "pneumonia") &
            (predicted == "normal")
        ).sum()
    )

    total = TP + TN + FP + FN

    def percentage(numerator, denominator):

        if denominator == 0:
            return None

        return (
            100 * numerator / denominator
        )

    precision = percentage(
        TP,
        TP + FP
    )

    recall = percentage(
        TP,
        TP + FN
    )

    specificity = percentage(
        TN,
        TN + FP
    )

    if (
        precision is not None
        and recall is not None
        and precision + recall != 0
    ):

        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

    else:

        f1 = None

    accuracy = percentage(
        TP + TN,
        total
    )

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Total Images": total,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-Score": f1
    }


def fmt(value):

    if value is None or pd.isna(value):

        return "N/A"

    return f"{value:.2f}%"


def metric_cards(metrics, names=None):

    if names is None:

        names = [
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1-Score"
        ]

    columns = st.columns(
        len(names)
    )

    for column, name in zip(columns, names):

        column.metric(
            name,
            fmt(metrics.get(name))
        )


# =========================================================
# CONFUSION MATRIX
# =========================================================

def cm_plot(metrics, title):

    matrix = np.array([
        [
            metrics["TP"],
            metrics["FN"]
        ],
        [
            metrics["FP"],
            metrics["TN"]
        ]
    ])

    fig, ax = plt.subplots(
        figsize=(4.7, 3.8)
    )

    image = ax.imshow(
        matrix,
        cmap="Blues"
    )

    ax.set_title(
        title,
        fontweight="bold",
        pad=12
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "Actual label"
    )

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Pneumonia",
        "Normal"
    ])

    ax.set_yticklabels([
        "Pneumonia",
        "Normal"
    ])

    threshold = (
        matrix.max() / 2
        if matrix.max()
        else 0
    )

    for row in range(2):

        for column in range(2):

            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontweight="bold",
                color=(
                    "white"
                    if matrix[row, column] > threshold
                    else "#17324d"
                )
            )

    fig.colorbar(
        image,
        ax=ax,
        fraction=.046,
        pad=.04
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# =========================================================
# DOWNLOAD FUNCTION - UNIQUE STREAMLIT KEYS
# =========================================================

def download_df(df, filename, label):

    unique_key = f"download_{filename}_{label}"

    unique_key = (
        unique_key
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
    )

    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=unique_key
    )
    # =========================================================
# LOAD MODEL PREDICTION FILES
# =========================================================

model_data = {}

model_metrics = {}

model_errors = {}

for name, filename in MODEL_FILES.items():

    data = load_csv(filename)

    if data is not None:

        model_data[name] = data

        try:

            _, _, actual, predicted = extract_predictions(
                data
            )

            model_metrics[name] = calculate_metrics(
                actual,
                predicted
            )

        except Exception as error:

            model_errors[name] = str(error)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="font-size:2.2rem">🩺</div>
        <div style="font-size:1.35rem;font-weight:800;font-family:Manrope">
            MEDICAL AI
        </div>
        <div style="font-size:.8rem">
            EVALUATION PLATFORM
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption("WORKSPACE")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Model Comparison",
            "Model Analysis",
            "Prediction Explorer",
            "Grad-CAM Explainability",
            "Threshold Analysis",
            "Confidence & Calibration",
            "Advanced Tools"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.caption("LOCAL PROJECT WORKSPACE")

    st.caption("Educational evaluation only")


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="main-heading">
        Medical AI Evaluation Toolkit
    </div>

    <div class="sub-heading">
        Unified workspace for evaluating chest X-ray classification models.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# THRESHOLD ANALYSIS HELPERS
# =========================================================

def load_threshold_data(model_name):

    filename = PROB_FILES.get(model_name)

    if not filename:
        return None, None, None

    probability_df = load_csv(filename)

    if probability_df is None:
        return None, None, None

    actual_col = find_col(
        probability_df,
        [
            "actual",
            "actual_label",
            "true_label",
            "ground_truth",
            "label",
            "target",
            "y_true"
        ]
    )

    probability_col = find_col(
        probability_df,
        [
            "pneumonia_probability",
            "probability_pneumonia",
            "p_pneumonia",
            "pneumonia_prob",
            "prob_pneumonia",
            "probability",
            "prob",
            "positive_probability",
            "positive_prob",
            "prediction_probability"
        ]
    )

    if actual_col is None:

        return probability_df, None, None

    actual = normalize_labels(
        probability_df[actual_col]
    )

    actual_binary = actual.map({
        "normal": 0,
        "pneumonia": 1
    })

    # -----------------------------------------------------
    # FALLBACK PROBABILITY COLUMN DETECTION
    # -----------------------------------------------------

    if probability_col is None:

        numeric_candidates = []

        for column in probability_df.columns:

            if column == actual_col:
                continue

            converted = pd.to_numeric(
                probability_df[column],
                errors="coerce"
            )

            valid_count = converted.notna().sum()

            if valid_count == len(probability_df):

                if (
                    converted.min() >= 0
                    and converted.max() <= 1
                ):

                    numeric_candidates.append(column)

        if numeric_candidates:

            probability_col = numeric_candidates[-1]

    if probability_col is None:

        return probability_df, actual_binary, None

    probabilities = pd.to_numeric(
        probability_df[probability_col],
        errors="coerce"
    )

    valid_mask = (
        actual_binary.notna()
        & probabilities.notna()
        & probabilities.between(0, 1)
    )

    actual_binary = actual_binary[valid_mask].astype(int)

    probabilities = probabilities[valid_mask].astype(float)

    return (
        probability_df,
        actual_binary.reset_index(drop=True),
        probabilities.reset_index(drop=True)
    )


# =========================================================
# THRESHOLD METRICS
# =========================================================

def calculate_threshold_metrics(actual, probabilities, threshold):

    actual = np.asarray(actual).astype(int)

    probabilities = np.asarray(probabilities).astype(float)

    predicted = (
        probabilities >= threshold
    ).astype(int)

    TP = int(
        np.sum((actual == 1) & (predicted == 1))
    )

    TN = int(
        np.sum((actual == 0) & (predicted == 0))
    )

    FP = int(
        np.sum((actual == 0) & (predicted == 1))
    )

    FN = int(
        np.sum((actual == 1) & (predicted == 0))
    )

    total = len(actual)

    accuracy = (
        (TP + TN) / total
        if total else 0
    )

    precision = (
        TP / (TP + FP)
        if (TP + FP) else 0
    )

    sensitivity = (
        TP / (TP + FN)
        if (TP + FN) else 0
    )

    specificity = (
        TN / (TN + FP)
        if (TN + FP) else 0
    )

    f1 = (
        2 * precision * sensitivity /
        (precision + sensitivity)
        if (precision + sensitivity) else 0
    )

    return {
        "Threshold": threshold,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Total": total,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Recall": sensitivity,
        "Specificity": specificity,
        "F1-Score": f1
    }
    # =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    hero(
        "PROJECT OVERVIEW",
        "Three-model evaluation workspace",
        "Real test-set predictions from your trained Custom CNN, ResNet50 and EfficientNet-B0 models."
    )

    if not model_metrics:

        st.warning(
            "No valid prediction reports loaded. "
            "Keep model prediction CSVs in this dashboard folder."
        )

    else:

        section("Available trained models")

        st.success(
            f"Successfully loaded {len(model_metrics)} model report(s): "
            f"{', '.join(model_metrics.keys())}"
        )

        comparison = pd.DataFrame(
            model_metrics
        ).T

        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1-Score"
        ]

        st.dataframe(
            comparison[metric_columns].applymap(fmt),
            use_container_width=True
        )

        section("Accuracy overview")

        st.bar_chart(
            comparison[["Accuracy"]]
        )

        section("Model highlights")

        best_accuracy = comparison["Accuracy"].idxmax()

        best_f1 = comparison["F1-Score"].idxmax()

        best_recall = comparison["Recall"].idxmax()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Highest Accuracy",
            best_accuracy,
            fmt(comparison.loc[best_accuracy, "Accuracy"])
        )

        col2.metric(
            "Highest F1",
            best_f1,
            fmt(comparison.loc[best_f1, "F1-Score"])
        )

        col3.metric(
            "Highest Recall",
            best_recall,
            fmt(comparison.loc[best_recall, "Recall"])
        )

    st.info(
        "This platform is for educational evaluation and research only—"
        "not clinical diagnosis or patient-care decisions."
    )


# =========================================================
# MODEL COMPARISON PAGE
# =========================================================

elif page == "Model Comparison":

    hero(
        "MODEL BENCHMARKING",
        "Compare the trained models",
        "Metrics are calculated directly from each model's actual prediction CSV—not simulated results."
    )

    if len(model_metrics) < 2:

        st.warning(
            "Load at least two valid prediction CSVs to compare models."
        )

    else:

        comparison = pd.DataFrame(
            model_metrics
        ).T

        section("Performance comparison")

        show = comparison[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1-Score"
            ]
        ].copy()

        st.dataframe(
            show.applymap(fmt),
            use_container_width=True
        )

        section("Metric charts")

        chart_metric = st.selectbox(
            "Choose metric",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1-Score"
            ]
        )

        st.bar_chart(
            comparison[[chart_metric]]
        )

        section("Confusion matrices")

        columns = st.columns(
            len(model_metrics)
        )

        for column, (name, metrics) in zip(
            columns,
            model_metrics.items()
        ):

            with column:

                cm_plot(
                    metrics,
                    name
                )

        download_df(
            comparison.reset_index(names="Model"),
            "medical_ai_real_model_comparison.csv",
            "Download comparison CSV"
        )

        st.caption(
            "Model results are based on the available dataset split "
            "and may not generalize to other hospitals or populations."
        )


# =========================================================
# MODEL ANALYSIS PAGE
# =========================================================

elif page == "Model Analysis":

    hero(
        "MODEL DETAILS",
        "Individual model analysis",
        "Inspect model-specific metrics, confusion matrices and prediction records."
    )

    if not model_metrics:

        st.warning(
            "No model prediction reports were loaded."
        )

    else:

        chosen_model = st.selectbox(
            "Select model",
            list(model_metrics.keys())
        )

        metrics = model_metrics[chosen_model]

        data = model_data[chosen_model]

        section(
            f"{chosen_model} · Test-set metrics"
        )

        metric_cards(metrics)

        left, right = st.columns(
            [1, 1.2]
        )

        with left:

            cm_plot(
                metrics,
                f"{chosen_model} confusion matrix"
            )

        with right:

            section("Metric details")

            st.dataframe(
                pd.DataFrame({
                    "Metric": list(metrics.keys()),
                    "Value": [
                        fmt(value)
                        if isinstance(value, (float, np.floating))
                        else value
                        for value in metrics.values()
                    ]
                }),
                hide_index=True,
                use_container_width=True
            )

            section("Class error counts")

            st.write(
                f"**False negatives:** {metrics['FN']} — "
                "pneumonia cases predicted Normal"
            )

            st.write(
                f"**False positives:** {metrics['FP']} — "
                "Normal cases predicted Pneumonia"
            )

        section("Prediction report")

        st.dataframe(
            data,
            use_container_width=True,
            height=350
        )

        download_df(
            data,
            f"{chosen_model.lower().replace(' ', '_')}_predictions.csv",
            "Download predictions"
        )

        probability_report = load_csv(
            PROB_FILES.get(chosen_model, "")
        ) if PROB_FILES.get(chosen_model) else None

        if probability_report is not None:

            section("Probability report")

            st.dataframe(
                probability_report,
                use_container_width=True,
                height=280
            )

            download_df(
                probability_report,
                f"{chosen_model.lower().replace(' ', '_')}_probabilities.csv",
                "Download probability report"
            )


# =========================================================
# PREDICTION EXPLORER PAGE
# =========================================================

elif page == "Prediction Explorer":

    hero(
        "IMAGE-LEVEL EVALUATION",
        "Prediction explorer",
        "Filter correct and incorrect image-level predictions for any loaded model."
    )

    if not model_data:

        st.warning(
            "No prediction reports loaded."
        )

    else:

        chosen_model = st.selectbox(
            "Model",
            list(model_data.keys())
        )

        data = model_data[chosen_model].copy()

        try:

            actual_column, predicted_column, actual, predicted = extract_predictions(
                data
            )

            data["Actual class"] = actual.map({
                "pneumonia": "Pneumonia",
                "normal": "Normal"
            })

            data["Predicted class"] = predicted.map({
                "pneumonia": "Pneumonia",
                "normal": "Normal"
            })

            data["Result"] = np.where(
                actual == predicted,
                "Correct",
                "Incorrect"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total predictions",
                len(data)
            )

            col2.metric(
                "Correct",
                int((data.Result == "Correct").sum())
            )

            col3.metric(
                "Incorrect",
                int((data.Result == "Incorrect").sum())
            )

            selected_filter = st.selectbox(
                "Filter predictions",
                [
                    "All",
                    "Correct",
                    "Incorrect"
                ]
            )

            if selected_filter == "All":

                filtered_data = data

            else:

                filtered_data = data[
                    data.Result == selected_filter
                ]

            section("Image-level records")

            st.dataframe(
                filtered_data,
                use_container_width=True,
                height=400
            )

            download_df(
                filtered_data,
                "prediction_explorer_filtered.csv",
                "Download filtered predictions"
            )

        except Exception as error:

            st.error(
                str(error)
            )
            # =========================================================
# THRESHOLD ANALYSIS PAGE
# =========================================================

elif page == "Threshold Analysis":

    hero(
        "DECISION THRESHOLD ANALYSIS",
        "Threshold optimization workspace",
        "Explore how changing the pneumonia probability threshold affects sensitivity, specificity, precision, accuracy and F1-score."
    )

    st.warning(
        "Threshold analysis is an educational research tool. "
        "A threshold selected using this dataset is not clinically validated "
        "and must not be used for patient-care decisions."
    )

    available_probability_models = []

    threshold_data = {}

    for model_name in PROB_FILES:

        report, actual, probabilities = load_threshold_data(
            model_name
        )

        if (
            report is not None
            and actual is not None
            and probabilities is not None
            and len(actual) > 0
        ):

            available_probability_models.append(
                model_name
            )

            threshold_data[model_name] = {
                "report": report,
                "actual": actual,
                "probabilities": probabilities
            }

    if not available_probability_models:

        st.error(
            "No usable probability reports were found. "
            "Threshold Analysis requires actual labels and pneumonia "
            "probabilities in your probability CSV files."
        )

        section("Expected probability report columns")

        st.markdown("""
        The probability report should contain:

        - **Actual label:** `actual`, `actual_label`, `true_label`, `label`, etc.
        - **Pneumonia probability:** `pneumonia_probability`, `probability_pneumonia`, `p_pneumonia`, or another numeric probability column between 0 and 1.
        """)

    else:

        # -------------------------------------------------
        # MODEL AND THRESHOLD CONTROLS
        # -------------------------------------------------

        control1, control2 = st.columns(2)

        with control1:

            selected_threshold_model = st.selectbox(
                "Select model",
                available_probability_models,
                key="threshold_model"
            )

        with control2:

            selected_threshold = st.slider(
                "Pneumonia decision threshold",
                min_value=0.01,
                max_value=0.99,
                value=0.50,
                step=0.01,
                key="threshold_slider"
            )

        selected_data = threshold_data[
            selected_threshold_model
        ]

        actual = selected_data["actual"]

        probabilities = selected_data["probabilities"]

        current_metrics = calculate_threshold_metrics(
            actual,
            probabilities,
            selected_threshold
        )

        # -------------------------------------------------
        # CURRENT THRESHOLD METRICS
        # -------------------------------------------------

        section(
            f"Performance at threshold {selected_threshold:.2f}"
        )

        metric_columns = st.columns(5)

        metric_columns[0].metric(
            "Accuracy",
            f"{current_metrics['Accuracy'] * 100:.2f}%"
        )

        metric_columns[1].metric(
            "Precision",
            f"{current_metrics['Precision'] * 100:.2f}%"
        )

        metric_columns[2].metric(
            "Sensitivity / Recall",
            f"{current_metrics['Sensitivity'] * 100:.2f}%"
        )

        metric_columns[3].metric(
            "Specificity",
            f"{current_metrics['Specificity'] * 100:.2f}%"
        )

        metric_columns[4].metric(
            "F1-Score",
            f"{current_metrics['F1-Score'] * 100:.2f}%"
        )

        # -------------------------------------------------
        # CONFUSION MATRIX COUNTS
        # -------------------------------------------------

        section("Confusion matrix counts")

        count_columns = st.columns(4)

        count_columns[0].metric(
            "True Positive",
            current_metrics["TP"]
        )

        count_columns[1].metric(
            "True Negative",
            current_metrics["TN"]
        )

        count_columns[2].metric(
            "False Positive",
            current_metrics["FP"]
        )

        count_columns[3].metric(
            "False Negative",
            current_metrics["FN"]
        )

        # -------------------------------------------------
        # CONFUSION MATRIX VISUALIZATION
        # -------------------------------------------------

        cm_plot(
            current_metrics,
            f"{selected_threshold_model} · Threshold {selected_threshold:.2f}"
        )

        # -------------------------------------------------
        # THRESHOLD EXPLANATION
        # -------------------------------------------------

        section("What this threshold means")

        st.markdown(
            f"""
            The selected decision threshold is **{selected_threshold:.2f}**.

            - If pneumonia probability is **greater than or equal to {selected_threshold:.2f}**,
              the model predicts **PNEUMONIA**.
            - If pneumonia probability is **below {selected_threshold:.2f}**,
              the model predicts **NORMAL**.
            """
        )

        st.info(
            "Lowering the threshold generally increases sensitivity but may "
            "increase false positives. Raising the threshold generally "
            "increases specificity but may increase false negatives. "
            "Actual results depend on the model and dataset."
        )

        # -------------------------------------------------
        # THRESHOLD SWEEP
        # -------------------------------------------------

        section("Threshold sweep: 0.10 to 0.90")

        thresholds = np.round(
            np.arange(0.10, 0.91, 0.01),
            2
        )

        sweep_records = []

        for threshold in thresholds:

            metrics = calculate_threshold_metrics(
                actual,
                probabilities,
                threshold
            )

            sweep_records.append(
                metrics
            )

        sweep_df = pd.DataFrame(
            sweep_records
        )

        sweep_display = sweep_df.copy()

        percentage_columns = [
            "Accuracy",
            "Precision",
            "Sensitivity",
            "Recall",
            "Specificity",
            "F1-Score"
        ]

        for column in percentage_columns:

            sweep_display[column] = (
                sweep_display[column] * 100
            )

        st.dataframe(
            sweep_display,
            hide_index=True,
            use_container_width=True,
            height=350
        )

        download_df(
            sweep_df,
            f"{selected_threshold_model.lower().replace(' ', '_')}_threshold_sweep.csv",
            "Download threshold sweep CSV"
        )

        # -------------------------------------------------
        # SENSITIVITY AND SPECIFICITY
        # -------------------------------------------------

        section("Sensitivity and specificity across thresholds")

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            sweep_df["Threshold"],
            sweep_df["Sensitivity"] * 100,
            label="Sensitivity / Recall",
            linewidth=2
        )

        ax.plot(
            sweep_df["Threshold"],
            sweep_df["Specificity"] * 100,
            label="Specificity",
            linewidth=2
        )

        ax.axvline(
            selected_threshold,
            linestyle="--",
            label=f"Selected threshold = {selected_threshold:.2f}"
        )

        ax.set_xlabel(
            "Decision threshold"
        )

        ax.set_ylabel(
            "Percentage"
        )

        ax.set_title(
            "Sensitivity and Specificity vs Threshold"
        )

        ax.set_ylim(
            0,
            100
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # -------------------------------------------------
        # PRECISION AND F1
        # -------------------------------------------------

        section("Precision and F1-score across thresholds")

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            sweep_df["Threshold"],
            sweep_df["Precision"] * 100,
            label="Precision",
            linewidth=2
        )

        ax.plot(
            sweep_df["Threshold"],
            sweep_df["F1-Score"] * 100,
            label="F1-Score",
            linewidth=2
        )

        ax.axvline(
            selected_threshold,
            linestyle="--",
            label=f"Selected threshold = {selected_threshold:.2f}"
        )

        ax.set_xlabel(
            "Decision threshold"
        )

        ax.set_ylabel(
            "Percentage"
        )

        ax.set_title(
            "Precision and F1-Score vs Threshold"
        )

        ax.set_ylim(
            0,
            100
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # -------------------------------------------------
        # ACCURACY AND F1
        # -------------------------------------------------

        section("Accuracy and F1-score across thresholds")

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            sweep_df["Threshold"],
            sweep_df["Accuracy"] * 100,
            label="Accuracy",
            linewidth=2
        )

        ax.plot(
            sweep_df["Threshold"],
            sweep_df["F1-Score"] * 100,
            label="F1-Score",
            linewidth=2
        )

        ax.axvline(
            selected_threshold,
            linestyle="--",
            label=f"Selected threshold = {selected_threshold:.2f}"
        )

        ax.set_xlabel(
            "Decision threshold"
        )

        ax.set_ylabel(
            "Percentage"
        )

        ax.set_title(
            "Accuracy and F1-Score vs Threshold"
        )

        ax.set_ylim(
            0,
            100
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)
                # -------------------------------------------------
        # BEST THRESHOLDS
        # -------------------------------------------------

        section("Threshold recommendations from this dataset")

        best_f1_row = sweep_df.loc[
            sweep_df["F1-Score"].idxmax()
        ]

        best_accuracy_row = sweep_df.loc[
            sweep_df["Accuracy"].idxmax()
        ]

        best_sensitivity_row = sweep_df.loc[
            sweep_df["Sensitivity"].idxmax()
        ]

        best_specificity_row = sweep_df.loc[
            sweep_df["Specificity"].idxmax()
        ]

        recommendation_columns = st.columns(4)

        recommendation_columns[0].metric(
            "Best F1 threshold",
            f"{best_f1_row['Threshold']:.2f}",
            f"F1: {best_f1_row['F1-Score'] * 100:.2f}%"
        )

        recommendation_columns[1].metric(
            "Best accuracy threshold",
            f"{best_accuracy_row['Threshold']:.2f}",
            f"Accuracy: {best_accuracy_row['Accuracy'] * 100:.2f}%"
        )

        recommendation_columns[2].metric(
            "Highest sensitivity threshold",
            f"{best_sensitivity_row['Threshold']:.2f}",
            f"Sensitivity: {best_sensitivity_row['Sensitivity'] * 100:.2f}%"
        )

        recommendation_columns[3].metric(
            "Highest specificity threshold",
            f"{best_specificity_row['Threshold']:.2f}",
            f"Specificity: {best_specificity_row['Specificity'] * 100:.2f}%"
        )

        st.warning(
            "These thresholds are mathematical results from the current "
            "test dataset. They are not medical recommendations and should "
            "not be selected for clinical deployment without independent "
            "validation, calibration assessment and clinical review."
        )

        # -------------------------------------------------
        # CROSS-MODEL THRESHOLD COMPARISON
        # -------------------------------------------------

        section("Cross-model comparison at selected threshold")

        cross_model_records = []

        for model_name in available_probability_models:

            model_actual = threshold_data[
                model_name
            ]["actual"]

            model_probabilities = threshold_data[
                model_name
            ]["probabilities"]

            metrics = calculate_threshold_metrics(
                model_actual,
                model_probabilities,
                selected_threshold
            )

            metrics["Model"] = model_name

            cross_model_records.append(
                metrics
            )

        cross_model_df = pd.DataFrame(
            cross_model_records
        )

        cross_model_df = cross_model_df[
            [
                "Model",
                "Threshold",
                "TP",
                "TN",
                "FP",
                "FN",
                "Total",
                "Accuracy",
                "Precision",
                "Sensitivity",
                "Specificity",
                "F1-Score"
            ]
        ]

        cross_model_display = cross_model_df.copy()

        cross_model_percentage_columns = [
            "Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1-Score"
        ]

        for column in cross_model_percentage_columns:

            if column in cross_model_display.columns:

                cross_model_display[column] = (
                    cross_model_display[column] * 100
                )

        st.dataframe(
            cross_model_display,
            hide_index=True,
            use_container_width=True
        )

        download_df(
            cross_model_df,
            "cross_model_threshold_comparison.csv",
            "Download cross-model comparison"
        )

        # -------------------------------------------------
        # CROSS-MODEL GRAPH
        # -------------------------------------------------

        section("Cross-model metric comparison")

        selected_comparison_metric = st.selectbox(
            "Choose comparison metric",
            [
                "Accuracy",
                "Precision",
                "Sensitivity",
                "Specificity",
                "F1-Score"
            ],
            key="threshold_cross_model_metric"
        )

        chart_data = cross_model_df.set_index(
            "Model"
        )[
            [selected_comparison_metric]
        ]

        st.bar_chart(
            chart_data * 100
        )

        # -------------------------------------------------
        # DOWNLOAD CURRENT THRESHOLD METRICS
        # -------------------------------------------------

        current_metrics_df = pd.DataFrame([
            current_metrics
        ])

        download_df(
            current_metrics_df,
            f"{selected_threshold_model.lower().replace(' ', '_')}_selected_threshold_metrics.csv",
            "Download selected-threshold metrics"
        )

        st.caption(
            "Threshold analysis uses the probability reports generated "
            "by your model evaluation scripts. The selected threshold "
            "changes the classification decision, not the model weights."
        )
        # =========================================================
# CONFIDENCE & CALIBRATION PAGE
# =========================================================

elif page == "Confidence & Calibration":

    hero(
        "PROBABILITY QUALITY",
        "Confidence & Calibration",
        "Evaluate probability quality, confidence, Brier score, expected calibration error and reliability across the trained models."
    )

    st.warning(
        "Calibration results are dataset-specific and educational. "
        "They do not establish clinical reliability or readiness for patient care."
    )

    calibration_models = {}

    for model_name in PROB_FILES:

        report, actual, probabilities = load_threshold_data(
            model_name
        )

        if (
            report is not None
            and actual is not None
            and probabilities is not None
            and len(actual)
        ):

            calibration_models[model_name] = {
                "report": report,
                "actual": np.asarray(actual, dtype=int),
                "probabilities": np.asarray(probabilities, dtype=float)
            }

    if not calibration_models:

        st.error(
            "No usable probability reports were found. Calibration requires "
            "actual labels and pneumonia probabilities in the probability CSV files."
        )

        st.info(
            "Check the probability report files in the Project File Status "
            "section of Advanced Tools."
        )

    else:

        selected_calibration_model = st.selectbox(
            "Select model for calibration analysis",
            list(calibration_models.keys()),
            key="calibration_model"
        )

        calibration_data = calibration_models[
            selected_calibration_model
        ]

        y_true = calibration_data["actual"]

        y_prob = calibration_data["probabilities"]

        y_pred = (
            y_prob >= 0.50
        ).astype(int)

        confidence = np.where(
            y_pred == 1,
            y_prob,
            1 - y_prob
        )

        correctness = (
            y_pred == y_true
        ).astype(int)

        accuracy = float(
            np.mean(correctness)
        )

        mean_confidence = float(
            np.mean(confidence)
        )

        brier_score = float(
            np.mean((y_prob - y_true) ** 2)
        )

        bin_edges = np.linspace(
            0,
            1,
            11
        )

        bin_ids = np.digitize(
            confidence,
            bin_edges[1:-1],
            right=False
        )

        bin_records = []

        ece = 0.0

        for bin_index in range(10):

            mask = bin_ids == bin_index

            count = int(
                mask.sum()
            )

            if count:

                bin_conf = float(
                    confidence[mask].mean()
                )

                bin_accuracy = float(
                    correctness[mask].mean()
                )

                ece += (
                    count / len(confidence)
                ) * abs(
                    bin_accuracy - bin_conf
                )

            else:

                bin_conf = np.nan

                bin_accuracy = np.nan

            bin_records.append({
                "Bin": (
                    f"{bin_edges[bin_index]:.1f}–"
                    f"{bin_edges[bin_index + 1]:.1f}"
                ),
                "Lower bound": bin_edges[bin_index],
                "Upper bound": bin_edges[bin_index + 1],
                "Count": count,
                "Mean confidence": bin_conf,
                "Accuracy": bin_accuracy,
                "Calibration gap": (
                    abs(bin_accuracy - bin_conf)
                    if count
                    else np.nan
                )
            })

        bins_df = pd.DataFrame(
            bin_records
        )

        # -------------------------------------------------
        # CALIBRATION METRICS
        # -------------------------------------------------

        section(
            "Calibration metrics at the 0.50 decision threshold"
        )

        metric_cols = st.columns(5)

        metric_cols[0].metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        metric_cols[1].metric(
            "Mean confidence",
            f"{mean_confidence * 100:.2f}%"
        )

        metric_cols[2].metric(
            "Brier score",
            f"{brier_score:.4f}"
        )

        metric_cols[3].metric(
            "ECE",
            f"{ece:.4f}"
        )

        metric_cols[4].metric(
            "Images",
            f"{len(y_true):,}"
        )

        # -------------------------------------------------
        # RELIABILITY DIAGRAM
        # -------------------------------------------------

        section("Reliability diagram")

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        valid_bins = bins_df.dropna(
            subset=[
                "Mean confidence",
                "Accuracy"
            ]
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfect calibration"
        )

        ax.plot(
            valid_bins["Mean confidence"],
            valid_bins["Accuracy"],
            marker="o",
            linewidth=2,
            label="Model"
        )

        ax.set_xlabel(
            "Mean confidence"
        )

        ax.set_ylabel(
            "Accuracy"
        )

        ax.set_title(
            f"Reliability diagram · {selected_calibration_model}"
        )

        ax.set_xlim(
            0,
            1
        )

        ax.set_ylim(
            0,
            1
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # -------------------------------------------------
        # CONFIDENCE DISTRIBUTION
        # -------------------------------------------------

        section("Confidence distribution")

        fig, ax = plt.subplots(
            figsize=(9, 4.5)
        )

        ax.hist(
            confidence,
            bins=np.linspace(0, 1, 11),
            edgecolor="black"
        )

        ax.set_xlabel(
            "Prediction confidence"
        )

        ax.set_ylabel(
            "Number of images"
        )

        ax.set_title(
            f"Confidence histogram · {selected_calibration_model}"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # -------------------------------------------------
        # CALIBRATION BIN DETAILS
        # -------------------------------------------------

        section("Calibration bin details")

        display_bins = bins_df.copy()

        for col in [
            "Mean confidence",
            "Accuracy",
            "Calibration gap"
        ]:

            display_bins[col] = display_bins[col].apply(
                lambda value: (
                    f"{value * 100:.2f}%"
                    if pd.notna(value)
                    else "—"
                )
            )

        st.dataframe(
            display_bins,
            hide_index=True,
            use_container_width=True
        )

        download_df(
            bins_df,
            f"{selected_calibration_model.lower().replace(' ', '_')}_calibration_bins.csv",
            "Download calibration bins CSV"
        )

        # -------------------------------------------------
        # CROSS-MODEL CALIBRATION COMPARISON
        # -------------------------------------------------

        section("Cross-model calibration comparison")

        comparison_records = []

        for model_name, data in calibration_models.items():

            actual_values = data["actual"]

            probability_values = data["probabilities"]

            predictions = (
                probability_values >= 0.50
            ).astype(int)

            conf_values = np.where(
                predictions == 1,
                probability_values,
                1 - probability_values
            )

            correct_values = (
                predictions == actual_values
            ).astype(int)

            brier = float(
                np.mean(
                    (probability_values - actual_values) ** 2
                )
            )

            ece_value = 0.0

            ids = np.digitize(
                conf_values,
                bin_edges[1:-1],
                right=False
            )

            for b in range(10):

                mask = ids == b

                if mask.any():

                    ece_value += (
                        mask.sum() / len(conf_values)
                    ) * abs(
                        float(correct_values[mask].mean())
                        -
                        float(conf_values[mask].mean())
                    )

            comparison_records.append({
                "Model": model_name,
                "Accuracy": float(
                    correct_values.mean()
                ),
                "Mean confidence": float(
                    conf_values.mean()
                ),
                "Brier score": brier,
                "ECE": ece_value,
                "Images": len(actual_values)
            })

        calibration_comparison = pd.DataFrame(
            comparison_records
        )

        calibration_display = calibration_comparison.copy()

        for col in [
            "Accuracy",
            "Mean confidence"
        ]:

            calibration_display[col] = (
                calibration_display[col].map(
                    lambda value: f"{value * 100:.2f}%"
                )
            )

        st.dataframe(
            calibration_display,
            hide_index=True,
            use_container_width=True
        )

        metric_to_chart = st.selectbox(
            "Compare calibration metric",
            [
                "Accuracy",
                "Mean confidence",
                "Brier score",
                "ECE"
            ],
            key="calibration_compare_metric"
        )

        st.bar_chart(
            calibration_comparison.set_index(
                "Model"
            )[
                [metric_to_chart]
            ]
        )

        download_df(
            calibration_comparison,
            "cross_model_calibration_comparison.csv",
            "Download cross-model calibration CSV"
        )

        st.caption(
            "Brier score and ECE are lower-is-better. Accuracy and mean "
            "confidence should be interpreted together; confidence alone "
            "is not proof of correctness."
        )
        # =========================================================
# GRAD-CAM EXPLAINABILITY PAGE
# =========================================================

elif page == "Grad-CAM Explainability":

    show_gradcam_page()


# =========================================================
# ADVANCED TOOLS PAGE
# =========================================================

elif page == "Advanced Tools":

    hero(
        "ADVANCED WORKSPACE",
        "Custom evaluation tools",
        "Upload external prediction reports or check which project reports are available."
    )

    # -----------------------------------------------------
    # SINGLE MODEL CSV EVALUATION
    # -----------------------------------------------------

    with st.expander(
        "Single-model CSV evaluation"
    ):

        uploaded_csv = st.file_uploader(
            "Upload CSV containing actual and predicted labels",
            type=["csv"],
            key="single"
        )

        if uploaded_csv:

            try:

                dataframe = pd.read_csv(
                    uploaded_csv
                )

                _, _, actual, predicted = extract_predictions(
                    dataframe
                )

                metrics = calculate_metrics(
                    actual,
                    predicted
                )

                metric_cards(
                    metrics
                )

                cm_plot(
                    metrics,
                    "Uploaded dataset confusion matrix"
                )

                download_df(
                    pd.DataFrame([metrics]),
                    "custom_evaluation_metrics.csv",
                    "Download metrics"
                )

            except Exception as error:

                st.error(
                    str(error)
                )

    # -----------------------------------------------------
    # PROJECT FILE STATUS
    # -----------------------------------------------------

    with st.expander(
        "Project file status",
        expanded=True
    ):

        paths = (
            list(MODEL_FILES.values())
            +
            list(PROB_FILES.values())
        )

        status = pd.DataFrame([
            {
                "File": filename,
                "Status": (
                    "Available"
                    if (BASE_DIR / filename).exists()
                    else "Missing"
                )
            }
            for filename in paths
        ])

        st.dataframe(
            status,
            hide_index=True,
            use_container_width=True
        )

        if model_errors:

            st.warning(
                "Some files were found but could not be evaluated: "
                + str(model_errors)
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Medical AI Evaluation Toolkit | Educational research project | "
    "Not intended for clinical diagnosis or patient-care decisions."
)