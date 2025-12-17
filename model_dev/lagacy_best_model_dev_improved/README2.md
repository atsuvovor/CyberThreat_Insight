Below is a **complete, publication-grade `README.md`** you can place at the **root of the `model_development/` (or equivalent) folder** of **CyberThreat-Insight**.

It is written to clearly explain **how unsupervised models are adapted to generate informative features for supervised multiclass threat classification**, using *your exact pipeline and code logic*.

You can paste this **as-is** into `README.md`.

---

# 🚨 CyberThreat-Insight

## Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI

**Toronto, September 08, 2025**

**Author:** Atsu Vovor
**Master of Management in Artificial Intelligence**
**Consultant – Data Analytics & Machine Learning**
**Data Science | Quantitative Analysis | SOC Analytics**
**Bilingual: French & English**

---

## 📌 Project Overview

Traditional anomaly detection systems excel at identifying *outliers* but fail at answering a critical SOC question:

> **“How severe is this threat?”**

Most unsupervised models only return:

* Binary anomaly flags (normal vs anomalous), or
* Unlabeled clusters with no operational meaning.

This repository introduces a **hybrid unsupervised–supervised learning architecture** that **adapts anomaly detection models into severity-aware feature generators**, enabling **multiclass cyber threat classification**:

**Low | Medium | High | Critical**

---

## 🎯 Core Problem Statement

> **Unsupervised models alone cannot classify multiple threat levels.**

They:

* Lack semantic labels
* Produce scores or clusters without business meaning
* Cannot directly support SOC prioritization or escalation workflows

This project solves that limitation by **aligning unsupervised outputs with labeled threat severity using weak supervision**, and then **leveraging those transformed signals for supervised learning**.

---

## 🧠 Key Insight: How the Algorithm Adapts Unsupervised Models

Instead of using unsupervised models *as final classifiers*, we treat them as:

> **Risk signal generators**

Each unsupervised model produces:

* An **anomaly score** (continuous risk)
* A **binary or cluster indicator** (structural behavior)

These outputs are then **mapped to known threat levels** using labeled training data.

---

## 🧩 Architecture Summary

```
Raw Cybersecurity Features
        │
        ▼
Unsupervised Models
(Isolation Forest, LOF, OCSVM, DBSCAN, KMeans, Autoencoder)
        │
        ▼
Anomaly Scores + Cluster / Binary Signals
        │
        ▼
Weak Supervision Label Alignment
(Majority Vote Mapping to Threat Levels)
        │
        ▼
Severity-Aware Features
        │
        ▼
Supervised Multiclass Classifiers
(Random Forest, Gradient Boosting, LSTM)
        │
        ▼
Threat Severity Prediction
(Low | Medium | High | Critical)
```

---

## 🔬 Step-by-Step Methodology

### 1️⃣ Feature Preparation

* Input features represent behavioral, temporal, and statistical signals
* Target variable: **Threat Level (0–3)**

  * 0 = Low
  * 1 = Medium
  * 2 = High
  * 3 = Critical

---

### 2️⃣ Unsupervised Risk Signal Extraction

The pipeline trains multiple unsupervised models:

| Model                | Risk Signal Produced          |
| -------------------- | ----------------------------- |
| Isolation Forest     | Decision score + anomaly flag |
| Local Outlier Factor | Local density deviation       |
| One-Class SVM        | Boundary distance             |
| DBSCAN               | Noise vs cluster membership   |
| KMeans               | Distance to centroid          |
| Autoencoder          | Reconstruction error          |

These models **do not predict threat levels directly**.

They answer instead:

> *“How unusual is this behavior?”*

---

### 3️⃣ Weak Supervision: Label Alignment

To convert unsupervised outputs into meaningful signals:

1. Run the unsupervised model on **training data**
2. Compare each cluster / anomaly flag with **true threat labels**
3. Assign each cluster or binary output the **majority threat level**
4. Store this mapping:

   ```python
   cluster_id → threat_level
   ```

This transforms raw anomaly signals into **severity-aware indicators**.

---

### 4️⃣ Severity-Aware Feature Construction

Each event now contains:

* Original cybersecurity features
* Anomaly score(s)
* Severity-mapped anomaly indicators

This creates a **hybrid feature space** combining:

* Statistical deviation
* Structural behavior
* Semantic risk meaning

---

### 5️⃣ Supervised Threat Classification

Final classifiers are trained on this enriched feature space:

* **Random Forest**
* **Gradient Boosting**
* **LSTM Neural Network**

These models learn **non-linear relationships between anomaly behavior and threat severity**, enabling robust multiclass prediction.

---

## 📊 Model Evaluation & Selection

The pipeline evaluates **all models consistently** using:

* Overall Accuracy
* Precision (Macro & Weighted)
* Recall (Macro & Weighted)
* F1-Score (Macro & Weighted)
* Confusion Matrices
* ROC & Precision-Recall Curves
* SOC-oriented performance insights

The **best model is automatically selected** and deployed.

---

## 🧠 Why This Works (and Why It’s Different)

| Traditional Approach | This Architecture           |
| -------------------- | --------------------------- |
| Binary anomaly flags | Severity-aware risk signals |
| No SOC context       | SOC-aligned threat levels   |
| Static thresholds    | Learned decision boundaries |
| Poor explainability  | Interpretable mappings      |
| Hard to govern       | Model-risk friendly         |

This design is **bank-grade**, **audit-ready**, and **SOC-operational**.

---

## 🏦 Model Explainability & Governance

This architecture supports regulatory expectations by:

* Explicit cluster-to-label mappings
* Transparent decision logic
* Separable unsupervised and supervised stages
* Reproducible metrics and artifacts
* Human-reviewable severity rationale

Ideal for:

* Financial institutions
* Model Risk Management (MRM)
* Internal audit & compliance
* SOC escalation governance

---

## 🚀 Deployment Output

The pipeline automatically produces:

* `model_metrics_summary.csv`
* Serialized best model (`.joblib` or TensorFlow format)
* Visual diagnostics
* Performance narratives for SOC teams

---

## 👨‍💻 Target Audience

* SOC Analysts
* Cybersecurity Data Scientists
* ML Engineers
* Risk & Governance Teams
* Financial Institutions & Regulators

---

## ✍️ Author

**Atsu Vovor**
Data & Analytics Consultant | Cybersecurity | AI Reporting
📧 [atsu.vovor@bell.net](mailto:atsu.vovor@bell.net)
🌐 [https://atsuvovor.github.io/projects_portfolio.github.io/#](https://atsuvovor.github.io/projects_portfolio.github.io/#)

---

## ⭐ Final Takeaway

> **Unsupervised models do not fail at cybersecurity — they fail at interpretation.**
> This project fixes that by turning anomaly detection into **actionable, severity-aware intelligence**.

---

If you want, next I can:

* Split this into **per-folder READMEs**
* Add a **SOC quick-start guide**
* Create an **executive one-page summary**
* Generate **architecture diagrams (PNG/SVG)**

Just say the word.
