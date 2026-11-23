
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Medical AI Evaluation Toolkit",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical AI Evaluation Toolkit")
st.write("Evaluate and compare medical AI classification models.")

# ==========================================
# 2. METRIC CALCULATION FUNCTION
# ==========================================

def calculate_metrics(actual, predicted):

    actual = actual.astype(str).str.strip().str.lower()
    predicted = predicted.astype(str).str.strip().str.lower()

    tp = int(((actual == "disease") & (predicted == "disease")).sum())
    tn = int(((actual == "healthy") & (predicted == "healthy")).sum())
    fp = int(((actual == "healthy") & (predicted == "disease")).sum())
    fn = int(((actual == "disease") & (predicted == "healthy")).sum())

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total * 100 if total else None
    precision = tp / (tp + fp) * 100 if (tp + fp) else None
    recall = tp / (tp + fn) * 100 if (tp + fn) else None
    specificity = tn / (tn + fp) * 100 if (tn + fp) else None

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Total Images": total,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-Score": f1
    }


# ==========================================
# 3. DATASET VALIDATION
# ==========================================

def validate_dataset(data, required_columns):

    if data.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if data[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if data["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs found.")

    valid_labels = {"disease", "healthy"}

    for col in required_columns[1:]:

        labels = (
            data[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if not set(labels).issubset(valid_labels):
            raise ValueError(
                f"Invalid labels found in {col}. "
                "Use Disease or Healthy."
            )


# ==========================================
# 4. CONFUSION MATRIX FUNCTION
# ==========================================

def show_confusion_matrix(metrics, title):

    cm = np.array([
        [metrics["TP"], metrics["FN"]],
        [metrics["FP"], metrics["TN"]]
    ])

    fig, ax = plt.subplots(figsize=(5, 4))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Disease", "Healthy"])
    ax.set_yticklabels(["Disease", "Healthy"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color="black"
            )

    fig.colorbar(image, ax=ax)
    st.pyplot(fig)
    plt.close(fig)


# ==========================================
# 5. DASHBOARD TABS
# ==========================================

tab1, tab2 = st.tabs([
    "Single Model Evaluation",
    "Multiple Model Comparison"
])


# ==========================================
# TAB 1: SINGLE MODEL EVALUATION
# ==========================================

with tab1:

    st.header("Single Medical AI Model Evaluation")

    uploaded_file = st.file_uploader(
        "Upload single-model CSV dataset",
        type=["csv"],
        key="single_upload"
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            required = [
                "image_id",
                "actual",
                "predicted"
            ]

            validate_dataset(data, required)

            st.success("Dataset validation successful!")

            metrics = calculate_metrics(
                data["actual"],
                data["predicted"]
            )

            st.subheader("Performance Metrics")

            columns = st.columns(5)

            metric_names = [
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1-Score"
            ]

            for col, name in zip(columns, metric_names):

                value = metrics[name]

                display = (
                    f"{value:.2f}%"
                    if value is not None
                    else "Undefined"
                )

                col.metric(name, display)

            st.subheader("Confusion Matrix")

            show_confusion_matrix(
                metrics,
                "Single Model Confusion Matrix"
            )

            st.subheader("Dataset Preview")

            st.dataframe(data, use_container_width=True)

            report = pd.DataFrame({
                "Metric": list(metrics.keys()),
                "Value": [
                    round(value, 2)
                    if isinstance(value, (int, float))
                    and value is not None
                    else "Undefined"
                    for value in metrics.values()
                ]
            })

            st.download_button(
                "Download Single Model Report",
                report.to_csv(index=False),
                file_name="single_model_report.csv",
                mime="text/csv"
            )

        except Exception as error:

            st.error(str(error))


# ==========================================
# TAB 2: MULTIPLE MODEL COMPARISON
# ==========================================

with tab2:

    st.header("Compare Multiple Medical AI Models")

    st.write(
        "Compare CNN, ResNet50 and EfficientNet "
        "using the same actual image labels."
    )

    comparison_file = st.file_uploader(
        "Upload medical_ai_model_comparison.csv",
        type=["csv"],
        key="comparison_upload"
    )

    if comparison_file is not None:

        try:

            data = pd.read_csv(comparison_file)

            model_columns = {
                "CNN": "cnn_prediction",
                "ResNet50": "resnet_prediction",
                "EfficientNet": "efficientnet_prediction"
            }

            required = [
                "image_id",
                "actual"
            ] + list(model_columns.values())

            validate_dataset(data, required)

            st.success("Comparison dataset validation successful!")

            st.subheader("Dataset Information")

            st.metric("Total Images", len(data))

            st.dataframe(data, use_container_width=True)

            # Calculate metrics for each model

            results = {}

            for model_name, prediction_column in model_columns.items():

                results[model_name] = calculate_metrics(
                    data["actual"],
                    data[prediction_column]
                )

            # ----------------------------------
            # COMPARISON TABLE
            # ----------------------------------

            st.divider()

            st.subheader("Model Performance Comparison")

            comparison = pd.DataFrame(results).T

            metric_columns = [
                "Accuracy",
                "Precision",
                "Recall",
                "Specificity",
                "F1-Score"
            ]

            formatted_comparison = comparison[metric_columns].copy()

            for col in metric_columns:

                formatted_comparison[col] = (
                    formatted_comparison[col].apply(
                        lambda x: f"{x:.2f}%"
                        if pd.notna(x)
                        else "Undefined"
                    )
                )

            st.dataframe(
                formatted_comparison,
                use_container_width=True
            )

            # ----------------------------------
            # BEST MODEL
            # ----------------------------------

            st.divider()

            st.subheader("Best Performing Model")

            selected_metric = st.selectbox(
                "Choose evaluation metric",
                metric_columns
            )

            valid_results = {
                name: values[selected_metric]
                for name, values in results.items()
                if values[selected_metric] is not None
            }

            if valid_results:

                best_model = max(
                    valid_results,
                    key=valid_results.get
                )

                st.success(
                    f"{best_model} has the highest "
                    f"{selected_metric}: "
                    f"{valid_results[best_model]:.2f}%"
                )

            else:

                st.warning(
                    "No valid values available for this metric."
                )

            # ----------------------------------
            # ACCURACY COMPARISON CHART
            # ----------------------------------

            st.divider()

            st.subheader("Accuracy Comparison Chart")

            accuracy_values = {
                name: values["Accuracy"]
                for name, values in results.items()
            }

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.bar(
                list(accuracy_values.keys()),
                [
                    value if value is not None else 0
                    for value in accuracy_values.values()
                ]
            )

            ax.set_title("Medical AI Model Accuracy")
            ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 100)

            for i, value in enumerate(accuracy_values.values()):

                label = (
                    f"{value:.2f}%"
                    if value is not None
                    else "Undefined"
                )

                ax.text(
                    i,
                    (value if value is not None else 0) + 1,
                    label,
                    ha="center"
                )

            st.pyplot(fig)
            plt.close(fig)

            # ----------------------------------
            # INDIVIDUAL CONFUSION MATRICES
            # ----------------------------------

            st.divider()

            st.subheader("Individual Confusion Matrices")

            matrix_columns = st.columns(3)

            for col, (model_name, metrics) in zip(
                matrix_columns,
                results.items()
            ):

                with col:

                    st.write(f"**{model_name}**")

                    show_confusion_matrix(
                        metrics,
                        f"{model_name} Confusion Matrix"
                    )

            # ----------------------------------
            # DOWNLOAD COMPARISON REPORT
            # ----------------------------------

            st.divider()

            st.subheader("Download Comparison Report")

            report = comparison.reset_index()
            report.rename(
                columns={"index": "Model"},
                inplace=True
            )

            csv_report = report.to_csv(index=False)

            st.download_button(
                "Download Model Comparison Report",
                data=csv_report,
                file_name="medical_ai_model_comparison_report.csv",
                mime="text/csv"
            )

            st.success(
                "Medical AI model comparison completed!"
            )

        except Exception as error:

            st.error(str(error))

st.divider()

st.caption(
    "Medical AI Evaluation Toolkit | Educational project | "
    "Not intended for clinical diagnosis."
)
