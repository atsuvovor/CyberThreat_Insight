# 🚨 Production Simulation – Cyber Threat Detection Engine

## Stacked Anomaly Detection Classifier (Inference Only)

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_anomaly_detector2.png"
       alt="Stacked Anomaly Detection Architecture"
       style="width: 1000px; height: auto;">
</p>

<p align="center">
  <strong>Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI</strong>
</p>

**Toronto, Sept 17, 2024**
**Author:** Atsu Vovor

> Master of Management in Artificial Intelligence
> Consultant – Data Analytics | Machine Learning | Data Science | Quantitative Analysis
> French & English Bilingual

---

### Project Overview

**Project:** Stacked Anomaly Detection Classifier
**Model Type:** Stacked Supervised Classifier with Unsupervised Anomaly Features
**Execution Mode:** Inference (Pretrained Models Only)
**Target Users:** Security Analysts, Data Analysts, AI/ML Practitioners

---

## 1. Introduction

The **Stacked Anomaly Detection Classifier** is a production-ready cybersecurity analytics engine designed to **detect anomalous and malicious behavior** in operational security data.

The solution combines:

### 🔹 Unsupervised Anomaly Detectors

* Isolation Forest
* One-Class SVM
* Local Outlier Factor (LOF)
* DBSCAN
* K-Means
* Dense Autoencoder
* LSTM Autoencoder

### 🔹 Supervised Ensemble (Stacked Learning)

* **Random Forest** (base learner)
* **Gradient Boosting** (meta learner)

Unsupervised anomaly scores are **engineered as features** and stacked with normalized input data to generate robust, explainable **Threat Level predictions**.

---

## 2. Threat Level Definitions

| Threat Level | Numeric Code | Meaning                     |
| ------------ | ------------ | --------------------------- |
| Low          | 0            | Normal activity             |
| Medium       | 1            | Suspicious but non-critical |
| High         | 2            | Confirmed anomaly           |
| Critical     | 3            | Severe malicious behavior   |

For operational use, **High & Critical** are treated as anomalies.

---

## 3. Execution Environment

### Recommended Platform

✔ **Google Colab** (fully supported)
✔ Jupyter Notebook (local execution possible with minor path adjustments)

### Software Requirements

* Python **3.9 – 3.12**
* Required libraries:

```bash
pip install numpy pandas scikit-learn tensorflow joblib matplotlib seaborn
```

No training is required — **models are preloaded**.

---

## 4. Model Artifacts (Required)

All pretrained artifacts must exist in:

```
CyberThreat_Insight/stacked_models_deployment/
```

### Loaded at Runtime

* `scaler.joblib` – Feature standardization
* `rf_base.joblib` – Random Forest base classifier
* `gb_meta.joblib` – Gradient Boosting meta learner
* `iso.joblib`, `ocsvm.joblib`, `lof.joblib`
* `dbscan.joblib`, `kmeans.joblib`
* `dense_autoencoder.keras`
* `lstm_autoencoder.keras`
* `train_X_scaled.npy` – Reference feature space

⚠️ **Missing artifacts will cause inference failure.**

---

## 5. Input Data Requirements (Critical)

### 5.1 Operational Dataset (New Data)

* Loaded **directly from Google Drive**
* Must contain the same feature columns used during training
* May optionally include `"Threat Level"` for evaluation

Example (configured in script):

```python
NEW_DATA_URL = "https://drive.google.com/file/d/1Nr9PymyvLfDh3qTfaeKNVbvLwt7lNX6l/view"
```

---

### 5.2 Augmented Reference Dataset (Mandatory)

An **augmented training dataset** is required to:

* Align feature columns
* Preserve correct input dimensionality

Example:

```python
AUGMENTED_DATA_URL = "https://drive.google.com/file/d/10UYplPdqse328vu1S1tdUAlYMN_TJ8II/view"
```

This dataset **is not used for prediction**, only for schema consistency.

---

## 6. Inference Workflow

The entire inference process is encapsulated in:

```python
predict_new_data(
    NEW_DATA_URL,
    AUGMENTED_DATA_URL,
    MODEL_DIR,
    label_col="Threat Level"
)
```

### What Happens Internally

1. Models and scalers are loaded
2. New operational data is fetched from Google Drive
3. Features are aligned using augmented reference data
4. Anomaly scores are generated
5. Features are stacked and classified
6. Results are appended to the original dataset

---

## 7. Running the Model (Colab Example)

```python
results_df = predict_new_data(
    NEW_DATA_URL,
    AUGMENTED_DATA_URL,
    MODEL_DIR
)

display(results_df.head())
```

No manual preprocessing is required.

---

## 8. Output Schema

The returned DataFrame contains all original fields plus:

| Column                     | Description                                 |
| -------------------------- | ------------------------------------------- |
| **Predicted Threat Level** | Final model prediction                      |
| **true_anomaly**           | Ground truth anomaly flag (if labels exist) |
| **anomaly_score**          | Mean anomaly score across detectors         |
| **predicted_anomaly**      | Binary flag (High/Critical = 1)             |

### Binary Mapping Logic

```text
Low / Medium → 0 (Normal)
High / Critical → 1 (Anomaly)
```

---

## 9. Interpretation Guidelines

* **Predicted Threat Level** is the primary operational output
* **anomaly_score** helps rank suspicious events
* **predicted_anomaly** supports alerting and dashboards
* **true_anomaly** is for validation only (if labels exist)

---

## 10. Deployment Considerations

### Suitable For

✔ Batch analysis of cybersecurity logs
✔ Post-incident forensic analysis
✔ AI-assisted SOC analytics
✔ Executive risk dashboards

### Production Extensions

* Wrap `predict_new_data()` in **FastAPI**
* Schedule batch inference jobs
* Integrate with SIEM tools
* Monitor drift and retrain periodically

---

## 11. Troubleshooting

| Issue             | Resolution                                       |
| ----------------- | ------------------------------------------------ |
| Model not found   | Verify `MODEL_DIR`                               |
| Drive load fails  | Ensure file is shared publicly                   |
| Feature mismatch  | Ensure augmented dataset matches training schema |
| TensorFlow errors | Restart Colab runtime                            |

---

## 12. Summary

This production pipeline enables **advanced cyber threat detection** using a **hybrid AI architecture** that combines:

* Statistical anomaly detection
* Deep learning autoencoders
* Supervised ensemble learning

End users can run **high-fidelity inference** on new cybersecurity data **without retraining**, making this solution ideal for operational analytics, research, and executive reporting.

