

<h1 align="center"> Cyber Threat Detection Engine: A Stacked Ensemble Approach</h1>

\<div align="center"\>

\<img src="https://github.com/atsuvovor/CyberThreat\_Insight/blob/main/images/lagacy\_model\_dev\_github.png" alt="Cyber Threat Detection Dashboard" width="900px"\>

**Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI & Stacked Generalization**

\</div\>

## 📋 Executive Summary

This project documents the end-to-end development of a **Cyber Threat Detection Engine** designed to identify anomalous behavior in security log data and classify threats by severity (**Low, Medium, High, Critical**).

Moving beyond traditional signature-based detection, this engine leverages a **Stacked Generalization (Stacking)** approach. It synthesizes the strengths of unsupervised anomaly detectors (to catch novel "zero-day" deviations) with the precision of supervised classifiers (to assign specific risk levels).

**Key Performance Highlights:**

  * **Best Base Model:** Random Forest with **98.12% Accuracy**.
  * **Critical Detection:** Identification of high-severity vectors with **94.7% Precision**.
  * **Architecture:** Hybrid pipeline utilizing unsupervised anomaly scores as latent features for a supervised meta-learner.

-----

## 🛠 Project Architecture & Roadmap

This repository is structured around three distinct stages of model maturity:

1.  **Baseline Development:** Comparative analysis of Supervised vs. Unsupervised models.
2.  **Hybrid Analysis:** Deep dive into feature importance and the limitations of unsupervised learning for multi-class severity.
3.  **Stacked Ensemble:** The final architecture using unsupervised outputs as inputs for a Gradient Boosting meta-learner.

### 📂 Quick Links

| Component | Description | Status |
| :--- | :--- | :--- |
| **Data Generation** | Synthetic log generation and preprocessing pipeline. | [](https://www.google.com/search?q=%5Bhttps://github.com/atsuvovor/CyberThreat_Insight/blob/main/datagen/README.md%5D\(https://github.com/atsuvovor/CyberThreat_Insight/blob/main/datagen/README.md\)) |
| **Feature Engineering** | Normalization, encoding, and selection. | [](https://github.com/atsuvovor/CyberThreat_Insight/blob/main/feature_engineering/README.md) |
| **Model Dev (Baseline)** | Initial multi-model benchmark. | [](https://colab.research.google.com/github/atsuvovor/CyberThreat_Insight/blob/main/model_dev/lagacy_best_model_dev/lagacy_model_dev_github.ipynb) |
| **Stacked Model** | Final ensemble implementation. | [](https://github.com/atsuvovor/CyberThreat_Insight/blob/main/model_dev/stacked_model/README.md) |

-----

## 🔍 Stage 1: The Multi-Class Challenge

**Objective:** Model cyber risk using a multi-class target variable defined as:

  * **0 – Low** (Normal background noise)
  * **1 – Medium** (Suspicious activity)
  * **2 – High** (Probable attack)
  * **3 – Critical** (Active breach/Severe Impact)

### The Unsupervised Gap

We initially tested pure unsupervised models (Isolation Forest, One-Class SVM, Autoencoders) to detect deviations. While these models excelled at binary classification (Normal vs. Anomaly), they failed to distinguish between threat severities.

  * **Observation:** Unsupervised models generalize outliers into a single "Anomaly" class (mapped typically to Class 1), missing the nuance between a *Medium* risk and a *Critical* breach.
  * **Result:** Models like Isolation Forest achieved only **\~58% accuracy** on the multi-class target because they inherently lack label context.

-----

## 📊 Stage 2: Supervised Success & Metrics

We pivoted to supervised learning to capture the decision boundaries between risk levels. The **Random Forest** and **Gradient Boosting** algorithms were the top performers.

### Comparative Metrics

| Algorithm | Type | Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Supervised | **0.9812** | 0.9704 | 0.8939 | 0.9265 |
| **Gradient Boosting** | Supervised | **0.9786** | 0.9738 | 0.8842 | 0.9190 |
| **LSTM Classifier** | Deep Learning | 0.7756 | 0.3770 | 0.4076 | 0.3917 |
| **KMeans** | Unsupervised | 0.7243 | 0.3530 | 0.3641 | 0.3544 |
| **Isolation Forest** | Unsupervised | 0.5889 | 0.1472 | 0.2500 | 0.1853 |

### Deep Dive: Random Forest Performance

While Random Forest achieved 98.12% accuracy, a deeper look at the **Critical** class (Class 3) revealed a specific challenge:

  * **Precision (Critical):** 0.9474 (High trust in positive alerts)
  * **Recall (Critical):** 0.6667 (Missed \~33% of critical events)

\<p align="center"\>
\<img src="[https://github.com/atsuvovor/CyberThreat\_Insight/blob/main/images/lagacy\_model\_improved\_confusion](https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_improved_confusion) Matrix2.png" alt="Confusion Matrix" width="600px"\>
\</p\>

**Business Insight:** The model is highly accurate but conservative. It minimizes False Positives (alert fatigue) but currently poses a risk of False Negatives for the most severe attacks. This specific gap drove the development of Stage 3.

-----

## 🚀 Stage 3: The Solution — Stacked Generalization

To bridge the gap between anomaly detection and specific classification, we implemented a **Stacked Supervised Model**.

### Methodology

Instead of discarding the unsupervised models, we utilized them as **Feature Generators**. The logic follows:

1.  **Level 0 (Unsupervised):** Pass data through Isolation Forest, LOF, and KMeans.
2.  **Feature Augmentation:** Extract anomaly scores and cluster distances.
    $$X_{new} = X_{original} \cup \{Score_{Iso}, Score_{LOF}, Dist_{KMeans}\}$$
3.  **Level 1 (Supervised Meta-Learner):** Train a Gradient Boosting classifier on $X_{new}$.

This approach allows the classifier to use "statistical weirdness" (from unsupervised models) as a heavily weighted feature to help distinguish between *High* and *Critical* threats that might look similar in raw feature space.

\<p align="center"\>
\<img src="[https://github.com/atsuvovor/CyberThreat\_Insight/blob/main/images/stacked\_anomaly\_detector2.png](https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_anomaly_detector2.png)" alt="Stacked Architecture" width="100%"\>
\</p\>

-----

## 📉 Visualizing Anomaly Detection (KMeans)

To understand *why* we included KMeans in the stack, we visualized its decision boundaries.

\<p align="center"\>
\<img src="[https://github.com/atsuvovor/CyberThreat\_Insight/blob/main/images/lagacy\_model\_improved\_metrics\_curves.png](https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_improved_metrics_curves.png)" alt="ROC and Precision-Recall Curves" width="100%"\>
\</p\>

  * **ROC AUC = 1.00:** In the latent space, KMeans provides perfect discrimination between the cluster centers (normal) and outliers.
  * **Scatter Plot:** Shows clear separation, confirming that distance-from-center is a high-value feature for the stacked model.

-----

## 🏁 Conclusion & Future Directions

This engine demonstrates that **hybridizing ML approaches** outperforms single-model strategies in cybersecurity. By stacking unsupervised anomaly scores into a supervised learner, we create a system that is robust against known threats while remaining sensitive to statistical anomalies.

### Future Work

  * **Oversampling:** Implement SMOTE to address the class imbalance in the 'Critical' category and improve Recall \> 0.80.
  * **Real-time Streaming:** Adapt the pipeline for Apache Kafka/Spark Streaming.
  * **Explainability:** Integrate SHAP (SHapley Additive exPlanations) values to tell SOC analysts *why* a log was flagged as Critical.

-----

**Author:** [Your Name/GitHub Handle]
**Domain:** Cybersecurity Analytics, Machine Learning, Data Science

