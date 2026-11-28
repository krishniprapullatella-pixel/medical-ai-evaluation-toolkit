# 🩺 Medical AI Evaluation Toolkit

An interactive dashboard for evaluating, comparing, analyzing, and interpreting machine learning models for **chest X-ray pneumonia classification**.

The project brings multiple model evaluation techniques into a single Streamlit application, helping users understand not only model accuracy, but also sensitivity, specificity, confidence, calibration, threshold behaviour, robustness, clinical risk, and explainability.

---

## 📌 Project Overview

Medical AI models can achieve promising performance metrics, but evaluating them using accuracy alone may hide important limitations—particularly in healthcare applications.

This project provides an interactive evaluation environment for comparing multiple pneumonia classification models and exploring their performance from different analytical perspectives.

The dashboard currently supports:

- Custom CNN
- ResNet50
- EfficientNet-B0

The goal is to build a comprehensive evaluation toolkit that demonstrates how medical AI models can be assessed beyond conventional accuracy metrics.

---

## 🎯 Problem Statement

Medical AI classification systems must be evaluated carefully before they can be considered useful in clinical settings.

A model with high accuracy may still:

- Miss positive cases
- Produce false-positive predictions
- Perform poorly at different decision thresholds
- Make overconfident predictions
- Show poor calibration
- Be sensitive to image perturbations
- Lack explainability

This project addresses these challenges by providing a multi-dimensional model evaluation dashboard.

---

## 🧠 Models Evaluated

| Model | Description |
|---|---|
| Custom CNN | Convolutional Neural Network developed for image classification |
| ResNet50 | Deep residual network architecture |
| EfficientNet-B0 | Efficient convolutional neural network architecture |

The dashboard allows model performance to be examined individually and compared across models.

---

## 📊 Evaluation Metrics

The toolkit supports analysis of several important classification metrics.

### Classification Metrics

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1 Score
- Confusion Matrix

### Clinical Performance

- False Negative analysis
- False Positive analysis
- Sensitivity and specificity trade-offs
- Clinical risk analysis
- Threshold-dependent performance

### Model Confidence

- Prediction confidence
- Confidence distribution
- Probability analysis
- Calibration analysis

---

## 🚀 Dashboard Features

The dashboard contains 12 analytical modules.

### 1. Overview

Provides a high-level summary of model performance and project evaluation results.

### 2. Model Comparison

Compare Custom CNN, ResNet50, and EfficientNet-B0 using multiple performance metrics.

### 3. Model Analysis

Explore individual model performance in greater detail.

### 4. Prediction Explorer

Explore individual model predictions and prediction-related information.

### 5. Grad-CAM Explainability

Visualize model attention using Grad-CAM.

Currently implemented for **ResNet50**.

The module displays:

- Original X-ray image
- Grad-CAM heatmap
- Heatmap overlay
- Prediction probabilities

Grad-CAM is intended to help explore which image regions contribute to the model's prediction.

### 6. Threshold Analysis

Explore how changing the decision threshold affects:

- Sensitivity
- Specificity
- Classification performance

This helps demonstrate the trade-off between identifying positive cases and reducing false positives.

### 7. Confidence & Calibration

Analyze prediction confidence and examine model calibration behaviour.

### 8. Advanced Tools

Access additional analytical utilities for deeper model evaluation.

### 9. Medical AI Scorecard

Provides a structured evaluation summary of model performance.

### 10. Clinical Risk Analysis

Explore model errors and their potential clinical implications.

Particular attention is given to:

- False negatives
- False positives
- Sensitivity
- Specificity
- Potential risk associated with classification errors

### 11. Robustness Stress Test

Evaluate how model performance may change under stress or perturbation scenarios supported by the dashboard.

### 12. Automated Research Report

Generate an automated evaluation report summarizing model analysis and results.

---

## 🛠️ Technology Stack

- **Python**
- **Streamlit**
- **PyTorch**
- **Torchvision**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Pillow**
- **Tabulate**

---

## 📁 Project Structure

```text
Medical AI Model Evaluator/
│
├── medical_ai_dashboard.py
│
├── medical_ai_cnn_predictions.csv
├── medical_ai_resnet50_predictions.csv
├── medical_ai_efficientnet_predictions.csv
│
├── medical_ai_cnn_probability_report.csv
├── medical_ai_resnet50_probability_report.csv
├── medical_ai_efficientnet_probability_report.csv
│
├── medical_ai_resnet50.pth
│
├── README.md
│
└── Other project files
```

*The exact file structure may vary depending on the files maintained in the repository.*

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd "Medical AI Model Evaluator"
```

### 3. Install required packages

```bash
pip install streamlit pandas numpy matplotlib scikit-learn torch torchvision pillow tabulate
```

---

## ▶️ Running the Dashboard

Run the following command from the project directory:

```bash
streamlit run medical_ai_dashboard.py
```

Streamlit will launch the application in your browser.

---

## 📈 Model Evaluation Workflow

```text
Prediction Data
      │
      ▼
Load Model Results
      │
      ▼
Custom CNN ── ResNet50 ── EfficientNet-B0
      │
      ▼
Performance Evaluation
      │
      ├── Classification Metrics
      ├── Model Comparison
      ├── Threshold Analysis
      ├── Confidence Analysis
      ├── Calibration
      ├── Clinical Risk
      ├── Robustness Testing
      └── Explainability
              │
              ▼
       Interactive Dashboard
              │
              ▼
       Automated Research Report
```

---

## 🔬 Research & Learning Objectives

This project was developed to explore practical aspects of Medical AI evaluation, including:

- Comparing deep learning architectures
- Understanding classification metrics
- Evaluating sensitivity and specificity
- Understanding threshold selection
- Exploring prediction confidence
- Studying calibration
- Investigating model robustness
- Understanding explainability techniques
- Evaluating potential clinical risks
- Building interactive AI evaluation tools

---

## ⚠️ Limitations

This project is an educational and research-oriented evaluation toolkit.

The dashboard's results depend on the datasets, prediction files, model outputs, and evaluation methods used.

The current implementation should not be interpreted as:

- A clinically validated diagnostic system
- A substitute for radiologists or medical professionals
- A medical device
- A system approved for clinical deployment

Further validation using appropriate datasets, external validation cohorts, clinical evaluation, regulatory review, and rigorous testing would be required before any clinical application.

---

## 🔮 Future Improvements

Potential future development includes:

- Grad-CAM support for all evaluated models
- Additional deep learning architectures
- External dataset validation
- Automated model comparison pipelines
- Model drift monitoring
- Improved calibration techniques
- Additional explainability methods
- Interactive image-level evaluation
- Exportable evaluation reports
- Integration with experiment tracking tools
- Advanced robustness evaluation
- Improved clinical risk assessment frameworks

---

## 👨‍💻 Author

Developed as an AI/ML and Medical AI learning project focused on model evaluation, explainability, and responsible healthcare AI.

---

## ⭐ Project Goal

The ultimate goal is to move beyond simply asking:

**"How accurate is the model?"**

And instead ask:

**"How reliable, explainable, calibrated, robust, and potentially safe is this AI model for a medical classification task?"**

---

## 📜 Disclaimer

This project is intended for **educational, research, and demonstration purposes only**.

It is not intended for diagnosis, treatment decisions, or clinical use. The outputs should not be used as a substitute for professional medical judgment.