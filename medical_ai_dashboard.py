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

        col1, col2 = st.columns(2)

        col1.metric(
            "NORMAL probability",
            f"{probabilities[0] * 100:.2f}%"
        )

        col2.metric(
            "PNEUMONIA probability",
            f"{probabilities[1] * 100:.2f}%"
        )

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
            "Grad-CAM Heatmap"
        )

        axes[2].imshow(
            resized_image
        )

        axes[2].imshow(
            cam,
            cmap="jet",
            alpha=0.42
        )

        axes[2].set_title(
            f"Grad-CAM Overlay · {predicted_class}"
        )

        for axis in axes:
            axis.axis("off")

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        section("How to read the Grad-CAM visualization")

        st.markdown("""
        - 🔴 **Red / Yellow:** Strong model activation
        - 🟢 **Green:** Moderate model activation
        - 🔵 **Blue:** Lower model activation

        The visualization highlights areas that contributed to the model's
        prediction. It does **not** prove that the model is focusing on
        medically correct regions.
        """)

        st.info(
            "Educational and research use only. This model is not validated "
            "for clinical diagnosis or patient-care decisions."
        )

    except Exception as error:

        st.error(
            f"Grad-CAM could not run: {error}"
        )

        st.caption(
            "Check that medical_ai_resnet50.pth is in the same folder "
            "as this dashboard and is the trained ResNet50 checkpoint."
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
# DOWNLOAD FUNCTION
# =========================================================

def download_df(df, filename, label):

    st.download_button(
        label,
        df.to_csv(index=False).encode(),
        file_name=filename,
        mime="text/csv"
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
# GRAD-CAM EXPLAINABILITY PAGE
# =========================================================

elif page == "Grad-CAM Explainability":

    show_gradcam_page()


# =========================================================
# CONFIDENCE & CALIBRATION PAGE
# =========================================================

elif page == "Confidence & Calibration":

    hero(
        "MODEL UNCERTAINTY",
        "Confidence & calibration analysis",
        "Study whether prediction confidence aligns with observed accuracy across the three models."
    )

    calibration_results = []

    for model_name, filename in PROB_FILES.items():

        report_path = BASE_DIR / filename

        if not report_path.exists():
            st.warning(f"{model_name}: probability report not found ({filename}).")
            continue

        try:
            df = pd.read_csv(report_path)

            # Flexible column lookup for probability reports
            column_lookup = {str(c).strip().lower(): c for c in df.columns}

            actual_col = next((column_lookup[c] for c in ["actual", "actual_label", "true_label", "y_true"] if c in column_lookup), None)
            prob_col = next((column_lookup[c] for c in ["pneumonia_probability", "probability_pneumonia", "p_pneumonia", "probability"] if c in column_lookup), None)
            confidence_col = next((column_lookup[c] for c in ["confidence", "prediction_confidence", "max_probability"] if c in column_lookup), None)
            correct_col = next((column_lookup[c] for c in ["correct", "is_correct", "prediction_correct"] if c in column_lookup), None)

            if actual_col is None or prob_col is None:
                st.warning(f"{model_name}: report needs actual-label and pneumonia-probability columns.")
                continue

            actual = pd.to_numeric(df[actual_col], errors="coerce")
            probability = pd.to_numeric(df[prob_col], errors="coerce")

            valid = actual.notna() & probability.notna()
            actual = actual[valid].astype(int)
            probability = probability[valid].clip(0, 1)
            clean_df = df.loc[valid].copy()

            if len(actual) == 0:
                st.warning(f"{model_name}: no valid rows found.")
                continue

            predicted = (probability >= 0.5).astype(int)
            correct = (predicted == actual).astype(int)
            confidence = np.maximum(probability, 1 - probability)

            if confidence_col is not None:
                reported_confidence = pd.to_numeric(clean_df[confidence_col], errors="coerce")
                confidence = reported_confidence.fillna(pd.Series(confidence, index=reported_confidence.index)).to_numpy()

            if correct_col is not None:
                reported_correct = clean_df[correct_col]
                if reported_correct.dtype == object:
                    reported_correct = reported_correct.astype(str).str.lower().map({"true": 1, "false": 0, "yes": 1, "no": 0})
                reported_correct = pd.to_numeric(reported_correct, errors="coerce")
                correct = reported_correct.fillna(pd.Series(correct, index=reported_correct.index)).astype(int).to_numpy()

            confidence = np.asarray(confidence, dtype=float)
            correct = np.asarray(correct, dtype=int)

            accuracy = float(correct.mean())
            mean_confidence = float(confidence.mean())
            correct_confidence = float(confidence[correct == 1].mean()) if np.any(correct == 1) else float("nan")
            incorrect_confidence = float(confidence[correct == 0].mean()) if np.any(correct == 0) else float("nan")
            brier_score = float(np.mean((probability.to_numpy() - actual.to_numpy()) ** 2))

            bin_edges = np.linspace(0, 1, 11)
            calibration_rows = []
            ece = 0.0

            for bin_index in range(10):
                lower = bin_edges[bin_index]
                upper = bin_edges[bin_index + 1]

                if bin_index == 9:
                    mask = (confidence >= lower) & (confidence <= upper)
                else:
                    mask = (confidence >= lower) & (confidence < upper)

                if mask.sum() == 0:
                    continue

                bin_conf = float(confidence[mask].mean())
                bin_acc = float(correct[mask].mean())
                bin_count = int(mask.sum())

                ece += (bin_count / len(confidence)) * abs(bin_acc - bin_conf)

                calibration_rows.append({
                    "Confidence Bin": f"{lower:.1f}–{upper:.1f}",
                    "Samples": bin_count,
                    "Mean Confidence": bin_conf,
                    "Accuracy": bin_acc,
                    "Calibration Gap": abs(bin_acc - bin_conf)
                })

            calibration_df = pd.DataFrame(calibration_rows)

            calibration_results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Mean Confidence": mean_confidence,
                "Correct Prediction Confidence": correct_confidence,
                "Incorrect Prediction Confidence": incorrect_confidence,
                "Brier Score": brier_score,
                "ECE": ece
            })

            st.divider()
            st.subheader(f"{model_name} analysis")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Accuracy",
                f"{accuracy:.2%}"
            )

            m2.metric(
                "Mean Confidence",
                f"{mean_confidence:.2%}"
            )

            m3.metric(
                "Brier Score",
                f"{brier_score:.4f}"
            )

            m4.metric(
                "Expected Calibration Error",
                f"{ece:.4f}"
            )

            m5, m6 = st.columns(2)

            m5.metric(
                "Confidence on Correct Predictions",
                f"{correct_confidence:.2%}"
                if not np.isnan(correct_confidence)
                else "N/A"
            )

            m6.metric(
                "Confidence on Incorrect Predictions",
                f"{incorrect_confidence:.2%}"
                if not np.isnan(incorrect_confidence)
                else "N/A"
            )

            # -------------------------------------------------
            # RELIABILITY DIAGRAM
            # -------------------------------------------------

            if not calibration_df.empty:

                fig, ax = plt.subplots(
                    figsize=(7, 5)
                )

                ax.plot(
                    [0, 1],
                    [0, 1],
                    linestyle="--",
                    label="Perfect calibration"
                )

                ax.plot(
                    calibration_df["Mean Confidence"],
                    calibration_df["Accuracy"],
                    marker="o",
                    label=model_name
                )

                ax.set_xlabel(
                    "Mean confidence"
                )

                ax.set_ylabel(
                    "Accuracy"
                )

                ax.set_title(
                    f"Reliability Diagram — {model_name}"
                )

                ax.set_xlim(0, 1)

                ax.set_ylim(0, 1)

                ax.grid(
                    True,
                    alpha=0.3
                )

                ax.legend()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

                # -------------------------------------------------
                # CONFIDENCE DISTRIBUTION
                # -------------------------------------------------

                st.subheader(
                    "Confidence distribution"
                )

                fig, ax = plt.subplots(
                    figsize=(7, 4)
                )

                ax.hist(
                    confidence,
                    bins=10,
                    edgecolor="black"
                )

                ax.set_xlabel(
                    "Prediction confidence"
                )

                ax.set_ylabel(
                    "Number of predictions"
                )

                ax.set_title(
                    f"Confidence Histogram — {model_name}"
                )

                ax.grid(
                    axis="y",
                    alpha=0.3
                )

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

                # -------------------------------------------------
                # CALIBRATION TABLE
                # -------------------------------------------------

                st.subheader(
                    "Calibration bins"
                )

                st.dataframe(
                    calibration_df,
                    hide_index=True,
                    use_container_width=True
                )

                download_df(
                    calibration_df,
                    f"{model_name.lower().replace(' ', '_')}_calibration_bins.csv",
                    "Download calibration bins CSV"
                )

        except Exception as error:

            st.error(
                f"Could not analyze {model_name}: {error}"
            )

    # =====================================================
    # CROSS-MODEL CALIBRATION COMPARISON
    # =====================================================

    if calibration_results:

        st.divider()

        st.header(
            "Cross-model calibration comparison"
        )

        comparison_df = pd.DataFrame(
            calibration_results
        ).set_index("Model")

        st.dataframe(
            comparison_df.style.format({
                "Accuracy": "{:.2%}",
                "Mean Confidence": "{:.2%}",
                "Correct Prediction Confidence": "{:.2%}",
                "Incorrect Prediction Confidence": "{:.2%}",
                "Brier Score": "{:.4f}",
                "ECE": "{:.4f}"
            }),
            use_container_width=True
        )

        c1, c2 = st.columns(2)

        with c1:

            st.subheader(
                "Brier score comparison"
            )

            st.bar_chart(
                comparison_df[["Brier Score"]]
            )

        with c2:

            st.subheader(
                "Expected calibration error comparison"
            )

            st.bar_chart(
                comparison_df[["ECE"]]
            )

        download_df(
            comparison_df.reset_index(),
            "medical_ai_calibration_comparison.csv",
            "Download comparison CSV"
        )

    st.info(
        "Educational and research use only. These models are not clinically validated "
        "and must not be used for diagnosis or patient-care decisions."
    )


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