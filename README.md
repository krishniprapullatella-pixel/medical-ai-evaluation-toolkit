# Medical AI Evaluation Toolkit

A Python-based toolkit for evaluating medical AI model performance using essential classification metrics.

## Project Overview

This project is designed to understand and evaluate how well an AI model performs when analysing medical datasets.

It includes Python programs for calculating model accuracy and confusion matrix-based evaluation metrics.

## Features

- Medical AI Accuracy Calculator
- Confusion Matrix Calculator
- True Positives (TP)
- True Negatives (TN)
- False Positives (FP)
- False Negatives (FN)
- Accuracy calculation
- Precision calculation

## Technologies Used

- Python 3
- VS Code
- Git
- GitHub

## Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures how reliable positive predictions are |
| Recall | Measures how many actual positive cases are identified |
| Specificity | Measures how many actual negative cases are identified |
| F1-Score | Harmonic mean of precision and recall |

*Recall, specificity and F1-score are planned improvements.*

## Project Structure

```text
medical-ai-evaluation-toolkit/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── accuracy_calculator/
│   └── medical_ai_calculator.py
│
└── confusion_matrix/
    └── medical_ai_confusion_matrix.py
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/krishniprapullatella-pixel/medical-ai-evaluation-toolkit.git
```

### 2. Navigate into the project

```bash
cd medical-ai-evaluation-toolkit
```

### 3. Run the Accuracy Calculator

```bash
python accuracy_calculator/medical_ai_calculator.py
```

### 4. Run the Confusion Matrix Calculator

```bash
python confusion_matrix/medical_ai_confusion_matrix.py
```

## Learning Objectives

- Understand medical AI model evaluation.
- Learn confusion matrix concepts.
- Implement evaluation metrics using Python.
- Practise Python programming and problem-solving.
- Develop practical skills for Medical Imaging AI.

## Future Improvements

- Recall and Specificity calculations
- F1-Score calculation
- Confusion Matrix visualization
- CSV dataset evaluation
- Graphical user interface
- Streamlit dashboard

## Disclaimer

This is an educational project for learning and experimentation. It is not intended for clinical diagnosis or medical decision-making.

## Author

Developed as part of my journey towards becoming a Medical Imaging AI Engineer.

