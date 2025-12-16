<div align="center">

 <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/cyber_threat_detection_engine4.png" 
       alt="Cyber Threat Detection Engine" 
       style="width: 600px; height: 40%;">

**Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI & Stacked Generalization**

</div>

**Toronto, September 08 2025**  
**Author: Atsu Vovor**

---

## Abstract

Cybersecurity threats are increasingly sophisticated and stealthy, rendering traditional signature-based detection methods insufficient. This white paper presents the **Cyber Threat Detection Engine**, a three-stage anomaly detection framework leveraging both supervised and unsupervised machine learning models enhanced by generative AI.

The methodology consists of:

1. **Stage 1 – Baseline Modeling:** Traditional supervised classifiers (Random Forest, Gradient Boosting) and unsupervised anomaly detection algorithms (Isolation Forest, LOF, DBSCAN) were implemented and evaluated.
2. **Stage 2 – Unsupervised Feature Adaptation:** Recognizing the limitations of unsupervised methods in multi-class classification, anomaly scores and cluster assignments were extracted from unsupervised models and included as additional features in supervised learning pipelines.
3. **Stage 3 – Stacked Hybrid Ensemble:** A two-layer stacked ensemble was developed, combining Random Forest as the base learner with Gradient Boosting as the meta-learner, leveraging both original features and anomaly-derived features.

Data augmentation techniques (SMOTE, GANs, Cholesky-based perturbations) were applied to balance the dataset. Evaluation demonstrates improved detection of rare and critical threat levels (Classes 2 and 3), outperforming individual supervised or unsupervised models. The framework provides a **robust, interpretable, and deployable solution** for real-time cybersecurity threat analytics.

---

## Introduction

Modern cybersecurity landscapes are characterized by the **rarity and subtlety of malicious events**, the **volume of benign activity**, and the **rapid evolution of attack patterns**. Traditional detection approaches relying solely on known signatures are insufficient for:

* Detecting zero-day attacks
* Identifying insider threats
* Recognizing anomalous patterns across multiple threat levels

The goal of this project is to develop a **robust multi-class cyber threat detection engine** capable of predicting threat levels from 0 (Low) to 3 (Critical). To address this, we adopted a **three-stage model evolution**, each stage building upon insights from the previous stage:

1. **Stage 1 – Baseline Modeling:** Implemented standard supervised and unsupervised models to establish performance benchmarks.
2. **Stage 2 – Unsupervised Feature Adaptation:** Converted unsupervised anomaly outputs into features for supervised learning, addressing gaps in multi-class prediction.
3. **Stage 3 – Stacked Ensemble:** Integrated original features, anomaly features, and base learner probabilities in a two-layer stacking architecture for optimal multi-class detection.

This paper provides **comprehensive documentation**, including dataset preparation, modeling choices, feature engineering, evaluation, deployment considerations, and future research directions.

---
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

| Component | Description | Link |
| :--- | :--- | :--- |
| **Data Generation** | Synthetic log generation and preprocessing pipeline. | [View README](https://github.com/atsuvovor/CyberThreat_Insight/blob/main/datagen/README.md) |
| **Feature Engineering** | Normalization, encoding, selection, and data augmentation (Cholesky, SMOTE, GANs). | [View README](https://github.com/atsuvovor/CyberThreat_Insight/blob/main/feature_engineering/README.md) |
| **Model Dev (Baseline)** | Initial supervised and unsupervised model benchmarks. | [View README](https://github.com/atsuvovor/CyberThreat_Insight/blob/main/model_dev/lagacy_best_model_dev/README.md) |
| **Hybrid Approach** | Integrated supervised + unsupervised threat detection. | [View README](https://github.com/atsuvovor/CyberThreat_Insight/blob/main/model_dev/lagacy_best_model_dev_improved/README.md) |
| **Stacked Model** | Final anomaly-augmented ensemble architecture. | [View README](https://github.com/atsuvovor/CyberThreat_Insight/blob/main/model_dev/stacked_model/README.md) |


---

## 🔍 Stage 1: Baseline Models - The Multi-Class Challenge
## Models Implemented

### Unsupervised Models (Anomaly Detection)

| Model | Purpose |
|------|--------|
| Isolation Forest | Outlier isolation |
| One-Class SVM | Boundary-based anomaly detection |
| Local Outlier Factor (LOF) | Density deviation |
| DBSCAN | Density-based clustering |
| KMeans | Cluster distance analysis |
| Autoencoder | Reconstruction error |
| LSTM Autoencoder | Temporal anomaly detection |

### Supervised Models (Threat Classification)

| Model | Purpose |
|------|--------|
| Random Forest | Robust multi-class classification |
| Gradient Boosting | High-precision meta learner |
| Logistic Regression | Baseline comparison |
| Stacked Ensemble | Final production model |



**Objective:** Model cyber risk using a multi-class target variable defined as:

  * **0 – Low** (Normal background noise)
  * **1 – Medium** (Suspicious activity)
  * **2 – High** (Probable attack)
  * **3 – Critical** (Active breach/Severe Impact)

### The Unsupervised Gap

We initially tested pure unsupervised models (Isolation Forest, One-Class SVM, Autoencoders) to detect deviations. While these models excelled at binary classification (Normal vs. Anomaly), they failed to distinguish between threat severities.

**Observation:**
  * **Supervised models** correctly classify all four threat levels using labeled data.  
  *  **Unsupervised models** generalize outliers into a single "Anomaly" class (mapped typically to Class 1), missing the nuance between a *Medium* risk and a *Critical* breach.
**Result:**
  * Models like Isolation Forest achieved only **\~58% accuracy** on the multi-class target because they inherently lack label context.
  * Stage 1 establishes **baseline performance metrics** and highlights the need for hybrid methods.

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/models_confusion_matrix.png"
       alt="Unsupervised Model Confusion Matrix"
       style="width: 100%; height: auto;">
</p>


**Key Challenge: Unsupervised Model Limitations**  

* Predict **binary outputs only** (normal vs anomaly)
* Fail to distinguish between **High (2)** and **Critical (3)** threats
* Treat all anomalies as a single class

---

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

<p align="center">
<img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_improved_confusion Matrix2.png" alt="Confusion Matrix"  width="600px">
</p>

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

<p align="center">
<img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_anomaly_detector2.png" alt="Stacked Architecture" width="100%">
</p>

-----

## 📉 Visualizing Anomaly Detection (KMeans)

To understand *why* we included KMeans in the stack, we visualized its decision boundaries.

<p align="center">
<img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_Kmeans_roc_recall.png" alt="ROC and Precision-Recall Curves" width="100%">
</p>


  * **ROC AUC = 1.00:** In the latent space, KMeans provides perfect discrimination between the cluster centers (normal) and outliers.
  * **Scatter Plot:** Shows clear separation, confirming that distance-from-center is a high-value feature for the stacked model.

-----

## 🏁 Conclusion & Future Directions

This engine demonstrates that **hybridizing ML approaches** outperforms single-model strategies in cybersecurity. By stacking unsupervised anomaly scores into a supervised learner, we create a system that is robust against known threats while remaining sensitive to statistical anomalies.

### Future Work

  * **Oversampling:** Implement SMOTE to address the class imbalance in the 'Critical' category and improve Recall > 0.80.
  * **Real-time Streaming:** Adapt the pipeline for Apache Kafka/Spark Streaming.
  * **Explainability:** Integrate SHAP (SHapley Additive exPlanations) values to tell SOC analysts *why* a log was flagged as Critical.

-----

## 🤝 Connect with me
I am always open to collaboration and discussion about new projects or technical roles.

Atsu Vovor  
Consultant, Data & Analytics    
Ph: 416-795-8246 | ✉️ atsu.vovor@bell.net    
🔗 <a href="https://www.linkedin.com/in/atsu-vovor-mmai-9188326/" target="_blank">LinkedIn</a> | <a href="https://atsuvovor.github.io/projects_portfolio.github.io/" target="_blank">GitHub</a> | <a href="https://public.tableau.com/app/profile/atsu.vovor8645/vizzes" target="_blank">Tableau Portfolio</a>    
📍 Mississauga ON      

### Thank you for visiting!🙏

