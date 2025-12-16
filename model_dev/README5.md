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

## Data Preparation and Augmentation

<h3>
  <a href="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/datagen/README.md">
    🔄 Data Generation
  </a>
</h3>


The dataset consists of cybersecurity event logs with a target variable **“Threat Level”**:

* 0 = Low
* 1 = Medium
* 2 = High
* 3 = Critical

### Data Augmentation

To balance class distribution and improve model performance, the following techniques were applied:

* **Cholesky-based perturbation:** Generates correlated noise for features
* **SMOTE (Synthetic Minority Oversampling Technique):** Augments minority classes
* **GANs (Generative Adversarial Networks):** Produces realistic synthetic threat patterns

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

def data_splitting(X_augmented, y_augmented, p_features_engineering_columns, target_column='Threat Level'):
    x_features = [col for col in p_features_engineering_columns if col != target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented[x_features],
        y_augmented,
        test_size=0.2,
        random_state=42,
        stratify=y_augmented
    )
    return X_train, X_test, y_train, y_test
```

* **Training Set:** 80%
* **Testing Set:** 20%
* **Stratified sampling** to preserve class ratios
* **Reproducibility:** `random_state=42`

---

## Stage 1 – Baseline Models

### Implemented Models

| Algorithm                  | Type                    | Description                                                                           |
| -------------------------- | ----------------------- | ------------------------------------------------------------------------------------- |
| Isolation Forest           | Unsupervised            | Detects anomalies via recursive partitioning of data.                                 |
| One-Class SVM              | Unsupervised            | Finds boundaries around normal points to detect anomalies.                            |
| Local Outlier Factor (LOF) | Unsupervised            | Compares local density deviations relative to neighbors.                              |
| DBSCAN                     | Unsupervised            | Density-based clustering; identifies outliers as noise.                               |
| Autoencoder                | Unsupervised            | Neural network reconstructing inputs; large reconstruction error indicates anomalies. |
| K-means Clustering         | Unsupervised            | Partitions data into clusters; distant points are flagged as anomalies.               |
| Random Forest              | Supervised              | Ensemble decision trees for multi-class classification.                               |
| Gradient Boosting          | Supervised              | Sequential ensemble of trees optimizing classification accuracy.                      |
| LSTM                       | Supervised/Unsupervised | Detects sequence anomalies via prediction error.                                      |

### Commentary

* **Supervised models** correctly classify all four threat levels using labeled data.
* **Unsupervised models** detect anomalies as **binary labels only**, failing to capture nuanced threat levels.
* Stage 1 establishes **baseline performance metrics** and highlights the need for hybrid methods.

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/models_confusion_matrix.png" 
       alt="Stage 1 Confusion Matrix" 
       style="width: 100%; height: auto;">
</p>  

---

## Stage 2 – Unsupervised Models as Feature Generators

### Rationale

Unsupervised models alone cannot classify multiple threat levels. Stage 2 adapts them to **generate informative features** for supervised learning.

### Feature Extraction

| Algorithm        | Feature Extracted               |
| ---------------- | ------------------------------- |
| Isolation Forest | Anomaly score                   |
| One-Class SVM    | Binary anomaly prediction       |
| LOF              | Local density deviation score   |
| DBSCAN           | Cluster membership / outlier    |
| Autoencoder      | Reconstruction error            |
| KMeans           | Cluster assignment              |
| LSTM             | Time-series anomaly probability |

### Commentary

* Enhances **signal for rare and critical threats** (Class 2 and 3).
* Converts anomaly sensitivity into **auxiliary features**.
* Stage 2 demonstrates **hybridization benefits**, enabling unsupervised models to contribute to multi-class prediction.

---

## Stage 3 – Stacked Hybrid Ensemble

### Architecture

* **Base Learner:** Random Forest
* **Meta Learner:** Gradient Boosting
* **Input Features:** Original + anomaly-derived features + base learner predict_proba

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_anomaly_classifier_flowchart.png" 
       alt="Stacked Model Architecture" 
       style="width: 80%; height: auto;">
</p>  

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y, test_size=0.2, stratify=y)

# Base & meta learners
base_model = RandomForestClassifier(n_estimators=200)
meta_model = GradientBoostingClassifier(n_estimators=200)

# Stacking ensemble
stacked_model = StackingClassifier(
    estimators=[('rf', base_model)],
    final_estimator=meta_model
)

# Train & evaluate
stacked_model.fit(X_train, y_train)
y_pred = stacked_model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Commentary

* Integrates **anomaly features** with supervised predictions.
* Improves **multi-class accuracy**, particularly for rare/high-risk threats.
* Fully **deployable with serialized artifacts**.

---

## Evaluation and Results

| Model                        | Accuracy | F1-Score (Class 3) | Recall (Class 3) |
| ---------------------------- | -------- | ------------------ | ---------------- |
| Random Forest Only           | 84%      | 0.51               | 0.48             |
| Gradient Boosting Only       | 83%      | 0.49               | 0.46             |
| **Stacked w/ Anomaly Feat.** | **88%**  | **0.61**           | **0.59**         |

**Observations:**

* Stage 2 feature engineering boosts rare class detection.
* Stage 3 stacked model shows **best overall performance**, particularly for critical threats.

---

## Discussion: Limitations and Future Work

**Limitations:**

1. **Model Complexity:** Stacked ensemble increases computational cost.
2. **Dependence on Augmentation:** Synthetic anomalies may introduce artifacts.
3. **Threshold Selection:** Anomaly feature thresholds may require fine-tuning.

**Future Work:**

* Experiment with **deep learning ensembles** for temporal security logs.
* Explore **real-time streaming implementation** for continuous threat detection.
* Implement **dynamic thresholding** for anomaly-based features.
* Integrate **explainability modules** for actionable cybersecurity insights.

---

## Deployment Considerations

* **Artifacts:** `scaler.joblib`, `rf_base.joblib`, `gb_meta.joblib`, `iso.joblib`, `ocsvm.joblib`, `lof.joblib`, `dbscan.joblib`, `kmeans.joblib`, `dense_autoencoder.keras`, `lstm_autoencoder.keras`
* **Integration:** SIEM platforms, real-time analytics engines
* **Monitoring:** Concept drift, periodic retraining, anomaly thresholds

---

## Conclusion

The **three-stage evolution** demonstrates a robust methodology for multi-class cyber threat detection:

1. Stage 1 establishes baseline metrics.
2. Stage 2 enriches data with unsupervised anomaly signals.
3. Stage 3 integrates these signals in a stacked ensemble for **superior multi-class performance**.

The approach provides a **deployable, interpretable, and effective solution** suitable for cybersecurity, fraud detection, and other anomaly-sensitive applications.

---
## 🤝 Connect with me
I am always open to collaboration and discussion about new projects or technical roles.

Atsu Vovor  
Consultant, Data & Analytics    
Ph: 416-795-8246 | ✉️ atsu.vovor@bell.net    
🔗 <a href="https://www.linkedin.com/in/atsu-vovor-mmai-9188326/" target="_blank">LinkedIn</a> | <a href="https://atsuvovor.github.io/projects_portfolio.github.io/" target="_blank">GitHub</a> | <a href="https://public.tableau.com/app/profile/atsu.vovor8645/vizzes" target="_blank">Tableau Portfolio</a>    
📍 Mississauga ON      

### Thank you for visiting!🙏

