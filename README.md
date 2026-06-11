# 🩺 Diabetes Risk Prediction — ML Pipeline & REST API

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-86.7%25-brightgreen?style=for-the-badge)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.91-blue?style=for-the-badge)

> An **end-to-end ML classification pipeline** for diabetes risk prediction — custom feature engineering, SMOTE oversampling, Random Forest classifier, and a production-ready **Flask REST API** ready for health-tech integration.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Pipeline Design](#pipeline-design)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Results & Evaluation](#results--evaluation)
- [Author](#author)

---

## Overview

This project builds a **reliable, production-ready diabetes risk classifier** — not just a notebook experiment. The full pipeline handles data preprocessing, class imbalance, feature engineering, model training, cross-validation, and API deployment with strict input validation.

The REST API accepts patient health metrics and returns a **risk probability + binary classification**, documented with a full Postman schema for direct integration into health-tech frontends or mobile apps.

---

## Model Performance

| Metric | Score |
|---|---|
| ✅ Accuracy | **86.7%** |
| 🎯 F1-Score | **0.84** |
| 📈 AUC-ROC | **0.91** |
| 🔁 Validation | 10-Fold Cross-Validation |
| ⚖️ Class Balance | SMOTE oversampling applied |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn |
| Model | Random Forest Classifier |
| Class Balancing | SMOTE (imbalanced-learn) |
| API Framework | Flask |
| Data Processing | Pandas, NumPy |
| Evaluation | Matplotlib, Seaborn |
| API Testing | Postman |

---

## Pipeline Design

Raw Dataset (CSV)
│
▼
┌─────────────────────────────────┐
│  Data Preprocessing             │
│  - Missing value imputation     │
│  - Outlier capping (IQR)        │
│  - MinMax scaling               │
└──────────────┬──────────────────┘
│
▼
┌─────────────────────────────────┐
│  Feature Engineering            │
│  - BMI-age interaction terms    │
│  - Glucose-insulin ratio        │
│  - Risk score composite feature │
└──────────────┬──────────────────┘
│
Train/Test Split (80/20)
│
┌────────┴────────┐
│                 │
Train Set         Test Set
│              (held out — no leakage)
▼
┌─────────────────────────────────┐
│  SMOTE Oversampling             │
│  (applied only on train set)    │
└──────────────┬──────────────────┘
│
▼
┌─────────────────────────────────┐
│  Random Forest Classifier       │
│  Grid Search Hyperparameter     │
│  Tuning (CV=10)                 │
└──────────────┬──────────────────┘
│
▼
┌─────────────────────────────────┐
│  Evaluation                     │
│  Accuracy / F1 / AUC-ROC        │
│  Confusion Matrix               │
└──────────────┬──────────────────┘
│
▼
┌─────────────────────────────────┐
│  Model Serialisation (joblib)   │
│  Flask REST API Deployment      │
└─────────────────────────────────┘
---

## Key Features

- 🧪 **Custom Feature Engineering** — BMI-age interaction terms, glucose-insulin ratio, and composite risk score
- ⚖️ **SMOTE Oversampling** — synthetic minority samples generated only on training split, no data leakage
- 🔁 **10-Fold Cross-Validation** — robust generalisation metrics beyond a single train/test split
- 🌲 **Random Forest + Grid Search** — tuned over `n_estimators`, `max_depth`, and `min_samples_split`
- 🛡️ **Strict Input Validation** — API rejects malformed or out-of-range inputs with descriptive errors
- 📄 **Postman-Documented API** — full JSON request/response schema for plug-in integration

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/siramkrishnapriya-cell/Diabetes-Prediction-Model.git
cd Diabetes-Prediction-Model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Start the Flask API
python app.py
API runs at http://localhost:5000

API Reference

POST /predict

Request Body:
{
  "pregnancies": 2,
  "glucose": 138,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree": 0.627,
  "age": 45
}
Response:
{
  "prediction": 1,
  "risk_probability": 0.84,
  "risk_label": "High Risk",
  "confidence": "84%"
}
Error Response:
Diabetes-Prediction-Model/
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── eda_and_training.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── model.py
│   └── validate.py
├── app.py
├── model.pkl
├── requirements.txt
├── docs/
│   └── postman_collection.json
└── README.md
Project Structure
Diabetes-Prediction-Model/
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── eda_and_training.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── model.py
│   └── validate.py
├── app.py
├── model.pkl
├── requirements.txt
├── docs/
│   └── postman_collection.json
└── README.md
Results & Evaluation
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.89      0.90       130
           1       0.81      0.83      0.84        84

    accuracy                           0.87       214
   macro avg       0.85      0.86      0.87       214
weighted avg       0.87      0.87      0.87       214

AUC-ROC Score: 0.91

Author
Siram Krishna Priya
📧 siramkrishnapriya@gmail.com
