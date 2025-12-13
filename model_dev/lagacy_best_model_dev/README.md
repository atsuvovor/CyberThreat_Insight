<h2 align="center">
Cyber Threat Detection Engine using Supervised and Unsupervised Models
</h2> 
<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_dev_github.png" 
       alt="Centered Image" 
       style="width: 1000px; height: Auto;">
</p>  

<p align="center">
    Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI
</p> 


## Introduction

The objective of this Model Development section is to design and implement a robust cyber threat detection engine capable of identifying anomalous behavior within security log data and accurately assessing its severity. To achieve this, we model cyber risk using a multi-class target variable, **Threat Level**, defined as:

* **0 – Low**
* **1 – Medium**
* **2 – High**
* **3 – Critical**

Accurately distinguishing between these threat levels is essential for enabling timely incident response, prioritizing security alerts, and supporting risk-informed decision-making.

This section presents a comprehensive exploration of both **supervised and unsupervised machine learning approaches** for cyber threat detection. Multiple algorithms are implemented, evaluated, and compared to determine their effectiveness in identifying malicious activity across varying levels of severity. Supervised models are trained to directly predict threat levels using labeled data, while unsupervised models are applied to detect deviations from normal behavior in the absence of labels.

Through empirical evaluation, this section highlights the strengths of supervised classification techniques in multi-class threat prediction, as well as the inherent limitations of traditional unsupervised anomaly detection methods. Building on these findings, we examine how unsupervised models can be adapted and augmented to better support multi-class threat classification, laying the groundwork for more resilient and scalable cyber defense systems.


## Models Implemented  


| Algorithm                   | Type           | Description                                                                                          |
|-----------------------------|----------------|------------------------------------------------------------------------------------------------------|
| **Isolation Forest**        | Unsupervised   | Anomaly detection by isolating outliers through random partitioning of data.                         |
| **One-Class SVM**           | Unsupervised   | Anomaly detection by identifying a region containing normal data points without labeled data.        |
| **Local Outlier Factor (LOF)** | Unsupervised   | Detects outliers by comparing local data density with that of neighboring points.                     |
| **DBSCAN**                  | Unsupervised   | Density-based clustering, also identifies outliers as noise.                                         |
| **Autoencoder**             | Unsupervised   | A neural network used to learn compressed representations, often for anomaly detection.              |
| **K-means Clustering**      | Unsupervised   | Clustering algorithm that partitions data into clusters without labels based on distance metrics.    |
| **Random Forest**           | Supervised     | An ensemble of decision trees used for classification or regression with labeled data.               |
| **Gradient Boosting**       | Supervised     | An ensemble method that builds sequential trees to improve prediction accuracy in classification or regression. |
| **LSTM (Long Short-Term Memory)** | Supervised/Unsupervised | Typically supervised for sequence prediction tasks, but can also be used in unsupervised anomaly detection. |

  
  

## Model Evaluation

While traditional classification metrics like accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC are primarily designed for binary classification problems, anomaly detection presents a unique challenge. In anomaly detection, the goal is to identify instances that deviate significantly from the normal pattern, rather than classifying them into predefined categories.

**That said, we can adapt some of these metrics to evaluate anomaly detection models**  

#### Applicable Metrics for Anomaly Detection

1. **Precision, Recall, and F1-Score:**
   - These metrics can be calculated by considering the true positive (TP), false positive (FP), true negative (TN), and false negative (FN) rates.
   - However, the definition of "positive" and "negative" in anomaly detection can be ambiguous. Often, the minority class (anomalies) is considered positive.
   - It's crucial to carefully define the positive and negative classes based on the specific use case and the desired outcome.

2. **ROC-AUC and PR-AUC:**
   - **ROC-AUC:** While it's commonly used for binary classification, it can be adapted to anomaly detection by treating anomalies as the positive class. However, the interpretation might be different.
   - **PR-AUC:** This metric is particularly useful for imbalanced datasets, which is often the case in anomaly detection. It focuses on the precision-recall trade-off.

3. **Confusion Matrix:**
   - A confusion matrix can be constructed to visualize the performance of an anomaly detection model. However, the interpretation might differ from traditional classification.

#### **Specific Considerations for Each Model**

1. **Isolation Forest, OneClassSVM, Local Outlier Factor, DBSCAN:**
   - These models directly output anomaly scores or labels.
   - You can set a threshold to classify instances as anomalies or normal.
   - Once you have the predicted labels, you can calculate the standard metrics.

2. **Autoencoder:**
   - Autoencoders are typically used for reconstruction-based anomaly detection.
   - You can calculate the reconstruction error for each instance.
   - A higher reconstruction error often indicates an anomaly.
   - You can set a threshold on the reconstruction error to classify instances.
   - Once you have the predicted labels, you can calculate the standard metrics.

3. **LSTM:**
   - LSTMs can be used for time series anomaly detection.
   - You can train an LSTM to predict future values and calculate the prediction error.
   - A higher prediction error often indicates an anomaly.
   - You can set a threshold on the prediction error to classify instances.
   - Once you have the predicted labels, you can calculate the standard metrics.

4. **Augmented K-Means:**
   - Augmented K-Means is a clustering-based anomaly detection technique.
   - Instances that are far from cluster centers can be considered anomalies.
   - You can set a distance threshold to classify instances.
   - Once you have the predicted labels, you can calculate the standard metrics.

## What Are the Models Predicting?  

Supervised models were evaluated using classification metrics: accuracy, precision, recall, F1-score, and confusion matrices. We noticed that Random Forest and Gradient Boosting both predicted all 4 classes accurately.  
Unsupervised models were originally evaluated by converting anomaly scores into binary labels (normal vs anomaly). However, they were only able to predict binary classes (typically class 0), failing to capture nuanced threat levels (2 and 3).  


### Supervised Models  

The supervised models directly predict the 'Threat Level' label and were able to classify all four
categories correctly. Their success is due to the availability of labeled training data and the ability to
learn decision boundaries across classes.

* **Objective**: Learn to predict the threat level (`Risk Level`: Class 0–3) directly from labeled training data.
* **Algorithms Used**:

  * Random Forest
  * Gradient Boosting
  * Logistic Regression
  * Stacking (Random Forest + Gradient Boosting)
* **Target**: `Risk Level` (0: No Threat → 3: High Threat)
* **Input**: Normalized features (numeric behavioral and system indicators)

### Unsupervised Models  

Unsupervised models like Isolation Forest, One-Class SVM, LOF, and DBSCAN are designed to distinguish anomalies from normal observations but not multiclass labels. These models predict binary labels (0 or 1). Class 0 indicates normal, class 1 indicates anomaly. When mapped against the threat
levels, they mostly capture only class 0 or 1.

* **Objective**: Detect anomalies in the data **without labels**, based on distance, density, or reconstruction error.
* **Algorithms Used**:

  * Isolation Forest
  * One-Class SVM
  * Local Outlier Factor (LOF)
  * DBSCAN
  * KMeans Clustering
  * Autoencoder (Neural Network)
  * LSTM (for sequential anomaly detection)
* **Output**: Binary anomaly scores (0 = normal, 1 = anomaly), not multiclass predictions

---

## Class Prediction Gaps in Unsupervised Models

### Observation:

All unsupervised models **fail to distinguish between threat levels (Class 1, 2, 3)**. Most anomaly detection models only predict **Class 0** or flag minority of samples as "anomalies", making it difficult to classify **subtle threat patterns**.

### Why Do Unsupervised Models Predict Only Class 0 for Class 2 and 3?

Unsupervised anomaly models fail to predict higher threat levels because:
- They are not trained with class labels and cannot distinguish among multiple classes.
- Anomalies are rare, and severe anomalies (high threat) are even rarer.
- These models generalize outliers as a single anomaly class (often mapped to class 1), unable to differentiate between moderate and critical threats.


<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/models_confusion_matrix.png" 
       alt="Centered Image" 
       style="width: 100%; height: Auto;">
</p>  



## Conclusion

This Model Development effort demonstrates that **supervised learning models**—particularly ensemble-based methods such as Random Forest and Gradient Boosting—are highly effective at predicting multi-class cyber threat levels when labeled data is available. These models successfully learn complex decision boundaries and consistently differentiate between low, medium, high, and critical threats, making them well-suited for operational security environments where historical labels exist.

In contrast, **unsupervised anomaly detection models** excel at identifying deviations from normal behavior but are fundamentally limited to **binary outcomes** (normal vs. anomalous). As observed, models such as Isolation Forest, One-Class SVM, LOF, DBSCAN, Autoencoders, and LSTM-based detectors are unable to distinguish between higher-severity threat classes. This limitation stems from their design, which treats all anomalies as a single category without contextual or severity-based differentiation.

These findings underscore a critical insight: while unsupervised models are valuable for early anomaly detection and zero-day threat discovery, they are insufficient on their own for nuanced, multi-level threat classification. Effective cyber threat intelligence systems must therefore either rely on supervised learning where labels exist or incorporate **hybrid and adaptive strategies** that combine anomaly detection with downstream classification, clustering, or risk-scoring mechanisms.

Overall, this section establishes a clear, evidence-based framework for selecting and adapting machine learning models in cybersecurity analytics, bridging the gap between anomaly detection and actionable threat severity classification.
