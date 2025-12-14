<h1 align="center">
Stacked Supervised Model using Unsupervised Anomaly Features
</h1> 


<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_anomaly_detector2.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>  

<p align="center">
    Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI**
</p> 

### Abstract

This project develops and evaluates a novel stacked anomaly detection model for cybersecurity analytics. Leveraging unsupervised anomaly detection techniques as feature generators, the model augments original data with anomaly scores and flags. A supervised meta-learner (Gradient Boosting) is then trained on this enhanced feature set, including outputs from a Random Forest base model. The approach aims to improve the accuracy and interpretability of threat detection by combining the strengths of various anomaly detection methods with a supervised classification framework. The project includes data loading, preprocessing, feature engineering, model training, comprehensive evaluation using standard classification and anomaly detection metrics, visualization of model performance, and a discussion of real-world deployment considerations.  

### Introduction

In the rapidly evolving landscape of cybersecurity threats, the ability to accurately and promptly identify malicious or anomalous activities is paramount. Traditional signature-based detection methods often fall short against novel and sophisticated attacks. Anomaly detection, which focuses on identifying deviations from normal behavior, offers a promising alternative. However, single anomaly detection techniques can have limitations in capturing the diverse nature of anomalies and providing actionable insights. This project addresses these challenges by proposing a stacked ensemble model that integrates multiple unsupervised anomaly detection methods. The output of these unsupervised models serves as crucial input features for a supervised classifier, enabling a more robust and nuanced approach to classifying different levels of cyber threats. This hybrid methodology aims to enhance the overall effectiveness and reliability of anomaly-based threat detection in complex cybersecurity environments.  

### Scope

The scope of this project includes:

1.  **Data Processing:** Loading, splitting, and scaling a provided augmented cybersecurity dataset (`x_y_augmented_data_google_drive.csv`).
2.  **Unsupervised Anomaly Feature Engineering:** Applying and training various unsupervised anomaly detection models (Isolation Forest, One-Class SVM, Local Outlier Factor, DBSCAN, KMeans, Dense Autoencoder, LSTM Autoencoder) on the training data to generate anomaly scores and flags as new features.
3.  **Stacked Model Development:** Training a Random Forest classifier as a base model and a Gradient Boosting classifier as a meta-learner on the extended feature set (original scaled features + unsupervised anomaly features + Random Forest `predict_proba`).
4.  **Model Evaluation:** Performing comprehensive evaluation of the final stacked model using multi-class classification metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report).
5.  **Unsupervised Model Analysis:** Evaluating and visualizing the performance of individual unsupervised anomaly detection models to understand their contributions.
6.  **Artifact Saving:** Saving the trained scaler, all unsupervised models, and the supervised stacked models (Random Forest base and Gradient Boosting meta) for potential future deployment.
7.  **Documentation:** Providing technical specifications and real-world usage guidelines for the developed stacked model.

This project focuses on the model development and evaluation phases using a predefined dataset and does not include real-time data ingestion, production deployment infrastructure development, or extensive hyperparameter tuning.  

## Stacked Supervised Model Technical Specifications  


**Architecture:**  

Stacked Anomaly Detection Classifier Model flowchart  
     
<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_anomaly_classifier_flowchart.png" 
       alt="Centered Image" 
       style="width: 100%; height: 150%;">
</p>  


The model employs a two-layer stacked architecture:
1.  **Base Learners:** A set of unsupervised anomaly detection models are used as feature generators. These models process the input data and produce continuous anomaly scores or binary anomaly flags. The models used are:
    *   Isolation Forest
    *   One-Class SVM
    *   Local Outlier Factor (LOF)
    *   DBSCAN (with nearest neighbor assignment for test data)
    *   KMeans (distance to centroid)
    *   Dense Autoencoder (reconstruction Mean Squared Error - MSE)
    *   LSTM Autoencoder (reconstruction MSE)

2.  **Meta Learner:** A supervised classification model is trained on an augmented feature set. This augmented set includes the original scaled features, the anomaly features generated by the base learners, and the `predict_proba` outputs from a Random Forest model trained on the extended feature set (original + anomaly features).
    *   **Base Model for Proba:** Random Forest Classifier
    *   **Meta Model:** Gradient Boosting Classifier

**Input Data:**
*   Augmented numeric dataset.
*   Assumes `Threat Level` is encoded as integers (0: Low, 1: Medium, 2: High, 3: Critical).
*   Requires features to be preprocessed (no missing values, categorical features encoded).

**Preprocessing Steps:**
1.  **Data Loading:** Load data from a specified CSV file (`x_y_augmented_data_google_drive.csv`).
2.  **Data Splitting:** Split data into training and testing sets using `train_test_split` with stratification on the target variable.
3.  **Feature Scaling:** Standardize features using `StandardScaler`.

**Feature Engineering (Anomaly Features):**
*   Each unsupervised model is fitted on the scaled training data.
*   Anomaly scores or flags are generated for both the scaled training and testing data.
*   These unsupervised features are concatenated with the original scaled features.

**Training Process:**
1.  **Unsupervised Models:** Each unsupervised model is trained independently on the scaled training data (Autoencoders are trained on a subset of "normal" data).
2.  **Random Forest (Base for Proba):** A Random Forest Classifier is trained on the concatenated original scaled features and unsupervised anomaly features.
3.  **Meta Learner (Gradient Boosting):** A Gradient Boosting Classifier is trained on a stacked feature set consisting of:
    *   Original scaled features
    *   Unsupervised anomaly features
    *   `predict_proba` outputs from the trained Random Forest model.

**Hyperparameters:**
*   `TEST_SIZE`: 0.20
*   `RANDOM_STATE`: 42
*   `CONTAMINATION`: 0.05 (for Isolation Forest, One-Class SVM, LOF)
*   `DBSCAN_EPS`: 0.5
*   `DBSCAN_MIN_SAMPLES`: 5
*   `KMEANS_CLUSTERS`: 4
*   `AUTOENCODER_EPOCHS`: 50
*   `LSTM_EPOCHS`: 50
*   `AUTOENCODER_BATCH`: 32
*   `LSTM_BATCH`: 32
*   Random Forest: `n_estimators=200`
*   Gradient Boosting: `n_estimators=200`
*   Autoencoder/LSTM Early Stopping: `patience=5` based on validation loss.

**Output:**
*   Predicted `Threat Level` (0, 1, 2, or 3).
*   Evaluation metrics: Accuracy, Precision, Recall, F1-score (macro and weighted), Confusion Matrix, Classification Report, ROC Curve, Precision-Recall Curve, Anomaly Detection Performance Metrics (AUC, Average Precision, F1).

**Model Artifacts Saved:**
*   `StandardScaler` object (`scaler.joblib`)
*   Trained Random Forest base model (`rf_base.joblib`)
*   Trained Gradient Boosting meta model (`gb_meta.joblib`)
*   Trained Isolation Forest model (`iso.joblib`)
*   Trained One-Class SVM model (`ocsvm.joblib`)
*   Trained Local Outlier Factor model (`lof.joblib`)
*   Trained DBSCAN model (`dbscan.joblib`)
*   Trained KMeans model (`kmeans.joblib`)
*   Scaled training data (for DBSCAN mapping) (`train_X_scaled.npy`)
*   Trained Dense Autoencoder model (`dense_autoencoder.keras`)
*   Trained LSTM Autoencoder model (`lstm_autoencoder.keras`)
*   Evaluation metrics (`metrics.json`)

**Performance Evaluation:**
*   The stacked model's performance is evaluated using standard multi-class classification metrics.
*   Individual unsupervised model performance for anomaly detection is evaluated using metrics like AUC and Average Precision (where applicable) against a binary anomaly ground truth derived from the `Threat Level`.

**Key Features:**
*   Leverages multiple unsupervised anomaly detection techniques to generate rich anomaly features.
*   Uses a stacked ensemble approach to combine the signals from unsupervised models and original features for improved classification.
*   Provides comprehensive evaluation metrics and visualizations for both the final stacked model and the individual unsupervised components.

**Potential Enhancements:**
*   Hyperparameter tuning for all models (unsupervised and supervised).
*   Experimentation with different unsupervised models or combinations.
*   Investigation into more sophisticated methods for combining unsupervised outputs.
*   Advanced techniques for handling class imbalance if necessary.
*   Integration with real-time data streams for continuous monitoring.  
  
  
## Real-World Usage of the Stacked Anomaly Detection Classifier

To use the trained Stacked Anomaly Detection Classifier in a real-world cybersecurity analytics environment, the following steps would typically be involved:

1.  **Model Deployment:**
    *   The saved model artifacts (`scaler.joblib`, `rf_base.joblib`, `gb_meta.joblib`, unsupervised models, `train_X_scaled.npy`) need to be deployed to a production environment. This could be a dedicated server, a cloud platform (like Google Cloud Platform, AWS, Azure), or an edge device, depending on the application's requirements and scale.
    *   The chosen environment should have the necessary dependencies installed (Python, scikit-learn, TensorFlow, joblib, numpy, pandas).

2.  **Data Ingestion and Preprocessing:**
    *   Real-world security data (e.g., network logs, system events, user activity logs) needs to be collected and processed.
    *   This data must be transformed into the same format as the training data, ensuring the same features are present and in the correct order.
    *   Any categorical features in the new data must be encoded using the same encoding scheme used during training (e.g., if label encoding was used, apply the saved label encoders).
    *   Missing values in the new data must be handled in the same way as during training (e.g., imputation).

3.  **Data Scaling:**
    *   The preprocessed real-world data must be scaled using the *saved* `StandardScaler` object (`scaler.joblib`). It's crucial to use the scaler fitted on the training data to ensure consistency.

4.  **Anomaly Feature Generation:**
    *   For each new data instance (or batch of instances), the scaled data is passed through the *saved* unsupervised models (`iso.joblib`, `ocsvm.joblib`, `lof.joblib`, `dbscan.joblib`, `kmeans.joblib`, `dense_autoencoder.keras`, `lstm_autoencoder.keras`).
    *   The anomaly scores or flags are extracted from these models, similar to how it was done for the test set during training.
    *   For DBSCAN, the nearest neighbor approach using the saved `train_X_scaled.npy` would be applied to assign anomaly flags to new data points.
    *   For Autoencoders, the reconstruction error (MSE) is calculated for the new scaled data.

5.  **Feature Stacking:**
    *   The generated anomaly features are concatenated with the original scaled features of the new data instances.
    *   The *saved* Random Forest base model (`rf_base.joblib`) is used to predict the `predict_proba` for the new extended feature set.
    *   These `predict_proba` outputs are then stacked with the extended features (original scaled + anomaly features) to create the final input for the meta-learner.

6.  **Threat Level Prediction:**
    *   The stacked features for the new data instances are fed into the *saved* Gradient Boosting meta-learner (`gb_meta.joblib`).
    *   The meta-learner outputs the predicted `Threat Level` (0, 1, 2, or 3) for each new data instance.

7.  **Actionable Insights and Response:**
    *   The predicted `Threat Level` for each event can be used to trigger alerts, prioritize security incidents, or initiate automated response actions.
    *   For example, events classified as "High" or "Critical" might automatically trigger an investigation by a security analyst or block suspicious network traffic.
    *   The anomaly scores generated by the unsupervised models can also be used to provide additional context or confidence levels for the predictions.

**Workflow Summary:**

Real-World Data -> Preprocessing -> Scale (using saved scaler) -> Generate Anomaly Features (using saved unsupervised models) -> Stack Features + RF Proba (using saved RF base model) -> Predict Threat Level (using saved GB meta-learner) -> Action/Alerting  

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>  


**Considerations for Production:**

*   **Scalability:** The deployment environment should be able to handle the volume and velocity of real-world security data.
*   **Latency:** For real-time threat detection, the model inference process needs to be fast.
*   **Monitoring:** Continuous monitoring of the model's performance in production is essential to detect concept drift or changes in threat patterns.
*   **Maintenance:** The model may need to be retrained periodically with new data to maintain its effectiveness.
*   **Integration:** The model needs to be integrated with existing security information and event management (SIEM) systems or other security tools.  


## Conclusion  


This project successfully developed and evaluated a stacked anomaly detection classifier leveraging unsupervised anomaly features for enhanced cybersecurity threat detection. The integrated pipeline demonstrated the process from data loading and preprocessing through unsupervised feature engineering, stacked model training, and comprehensive evaluation. The results indicate the potential of this hybrid approach to improve the accuracy and interpretability of anomaly detection compared to using individual unsupervised methods alone. While the model shows promising performance, particularly in distinguishing various threat levels, the analysis also highlighted areas for potential improvement, such as enhancing the recall for critical threats and further optimizing individual unsupervised components. The saved model artifacts and detailed documentation provide a solid foundation for potential real-world deployment and further research into advanced anomaly detection techniques in cybersecurity. 



---
## Stacked Supervised Model  Performance Details  
**Run the code:**
<a 
  href="https://colab.research.google.com/github/atsuvovor/CyberThreat_Insight/blob/main/model_dev/stacked_model/Stacked_Anomaly_Detection_Classifier_GitHub.ipynb"
  target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<details>

<summary><strong>Click to view the code</strong></summary>

```python

"""
#Stacked Supervised Model using Unsupervised Anomaly Features  

**Toronto, Sept 17 202**  
**Autor : Atsu Vovor**
>Master of Management in Artificial Intelligence    
>Consultant Data Analytics Specialist | Machine Learning |  
Data science | Quantitative Analysis |French & English Bilingual  

**Model Name:** Stacked Anomaly Detection Classifier

**Model Type:** Stacked Ensemble Model

## Stacked Supervised Model Code
"""

#-------------------
#  Import Library
#------------------

import os
import numpy as np
import pandas as pd
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score, roc_auc_score)
from sklearn.decomposition import PCA
import tensorflow as tf
#from google.colab import drive
#drive.mount("/content/drive")
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.colors import LinearSegmentedColormap


# -----------
# PARAMETERS
# -----------

LABEL_COL = "Threat Level"
TEST_SIZE = 0.20
RANDOM_STATE = 42

AUTOENCODER_EPOCHS = 50
LSTM_EPOCHS = 50
AUTOENCODER_BATCH = 32
LSTM_BATCH = 32
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
KMEANS_CLUSTERS = 4
CONTAMINATION = 0.05
DATA_PATH =  "CyberThreat_Insight/cybersecurity_data"
DATA_PATH = DATA_PATH + "/x_y_augmented_data_google_drive.csv"
MODEL_OUTPUT_DIR = "CyberThreat_Insight/stacked_models_deployment"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

#------------------
# Color Mapping
#------------------
def get_color_map():
    # Define the colors
    #colors = ["darkred", "red", "orangered", "orange", "yelloworange", "lightyellow", "yellow", "greenyellow", "green"]
    colors = ["#8B0000", "#FF0000", "#FF4500", "#FFA500", "#FFB347", "#FFFFE0", "#FFFF00", "#ADFF2F", "#008000"]

    # Create a colormap
    custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", colors)

    return custom_cmap
custom_cmap = get_color_map()

def log(msg):
    print(f"[INFO] {msg}")
    with open(os.path.join(MODEL_OUTPUT_DIR, "log.txt"), "a") as f:
        f.write(f"{msg}\n")

def get_label_class_name(model_y_test, y_model_pred):
    """Get class names based on labels."""
    labels = sorted(list(set(model_y_test) | set(y_model_pred)))
    level_mapping = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
    class_names = [level_mapping.get(label) for label in labels]

    return class_names, labels

#----------------------------------------Model performance report-----------------------------------
def get_metrics(y_true, y_pred, report):
    """Get model performance metrics from classification report."""
    class_names = list(y_true.unique())
    #report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics_dic = {
        "Precision (Macro)": report['macro avg']['precision'],
        "Recall (Macro)": report['macro avg']['recall'],
        "F1 Score (Macro)": report['macro avg']['f1-score'],
        "Precision (Weighted)": report['weighted avg']['precision'],
        "Recall (Weighted)": report['weighted avg']['recall'],
        "F1 Score (Weighted)": report['weighted avg']['f1-score'],
        "Accuracy": accuracy_score(y_true, y_pred),
        "Overall Model Accuracy ": report['accuracy'],

    }
    return metrics_dic

def print_model_performance_report_(model_name, y_true, y_pred):
    """Prints a classification report and confusion matrix for a given model."""
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred))

    print(f"\n{model_name} Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap= custom_cmap, cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

#------------------------------
def plot_metrics_bell_curve(model_name, y_true, y_pred):
    """
    Plots PPV, Sensitivity, NPV, Specificity for multi-class classification
    on separate bell-curve style subplots for clarity.
    """

    cm = confusion_matrix(y_true, y_pred)
    # Assuming class names are 0, 1, 2, 3
    class_names = ["Low", "Medium", "High", "Critical"]


    # Compute TP, FP, FN, TN
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    # Metrics
    PPV = TP / (TP + FP)
    Sensitivity = TP / (TP + FN)
    NPV = TN / (TN + FN)
    Specificity = TN / (TN + FP)

    metrics_df = pd.DataFrame({
        "Class": class_names,
        "PPV": PPV,
        "Sensitivity": Sensitivity,
        "NPV": NPV,
        "Specificity": Specificity
    })

    print(f"\n=== {model_name} Classification Metrics ===")
    print(metrics_df)

    # Plot bell curves in subplots
    metrics_to_plot = ["PPV", "Sensitivity", "NPV", "Specificity"]
    colors = ["skyblue", "lightgreen", "salmon", "orange", "purple", "cyan", "pink"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    x = np.linspace(0, 1, 500)
    sigma = 0.05  # width of bell curve

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        for j, value in enumerate(metrics_df[metric]):
            y = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x - value)/sigma)**2)
            ax.plot(x, y, color=colors[j % len(colors)], alpha=0.7, label=class_names[j])
        ax.set_title(metric)
        ax.set_xlabel("Metric Value (0-1)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle(f"{model_name} - Multi-class Metrics Bell Curves", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return metrics_df
#---------------------------------------
def create_scatter_plot(data, x, y, hue, ax, x_label=None, y_label=None):
    """Generate scatter plot for anomalies vs normal points."""
    sns.scatterplot(x=x, y=y, hue=hue, palette={0: 'blue', 1: 'red'}, data=data, ax=ax)
    ax.set_title("Anomalies (Red) vs Normal Points (Blue)")
    ax.set_xlabel(x_label or x)
    ax.set_ylabel(y_label or y)


def create_roc_curve(data, anomaly_score, is_anomaly, ax):
    """Generate ROC curve and calculate AUC."""
    fpr, tpr, _ = roc_curve(data[is_anomaly], data[anomaly_score])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")


def create_precision_recall_curve(data, anomaly_score, is_anomaly, ax):
    """Generate Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(data[is_anomaly], data[anomaly_score])
    ax.plot(recall, precision, color='purple', lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
#--------------------------
def generate_performance_insight(model_name, report, y_true, y_pred):
    """
    Generates a dynamic performance report insight based on classification metrics.

    Args:
        model_name (str): The name of the model.
        report (dict): The classification report as a dictionary.
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    """
    print(f"\n--- Performance Insight for {model_name} ---")

    class_names, labels = get_label_class_name(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Compute TP, FP, FN, TN for all classes
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    print("\nMetrics per Class:")
    for i, class_name in enumerate(class_names):
        label_str = str(labels[i]) # Get the string representation of the label
        print(f"\n  Class: {class_name}")
        print(f"    Precision: {report[label_str]['precision']:.4f} - Of all instances predicted as '{class_name}', this is the proportion that were actually '{class_name}'.")
        print(f"    Recall: {report[label_str]['recall']:.4f} - Of all actual '{class_name}' instances, this is the proportion that the model correctly identified.")
        print(f"    F1-Score: {report[label_str]['f1-score']:.4f} - A balanced measure of Precision and Recall for this class.")
        print(f"    Support: {report[label_str]['support']} - The number of actual instances of this class in the test set.")
        # PPV, NPV, Sensitivity (Recall), Specificity
        ppv = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
        npv = TN[i] / (TN[i] + FN[i]) if (TN[i] + FN[i]) > 0 else 0
        sensitivity = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0 # Same as Recall
        specificity = TN[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0

        print(f"    PPV (Positive Predictive Value): {ppv:.4f} - The probability that a positive prediction is a true positive.")
        print(f"    NPV (Negative Predictive Value): {npv:.4f} - The probability that a negative prediction is a true negative.")
        print(f"    Sensitivity (True Positive Rate): {sensitivity:.4f} - Ability to correctly identify positive instances.")
        print(f"    Specificity (True Negative Rate): {specificity:.4f} - Ability to correctly identify negative instances.")


    print("\nAggregated Metrics:")
    print(f"  Accuracy: {report['accuracy']:.4f} - The overall proportion of correctly classified instances.")
    print(f"  Precision (Macro): {report['macro avg']['precision']:.4f} - Average Precision across all classes, unweighted.")
    print(f"  Recall (Macro): {report['macro avg']['recall']:.4f} - Average Recall across all classes, unweighted.")
    print(f"  F1 Score (Macro): {report['macro avg']['f1-score']:.4f} - Average F1-Score across all classes, unweighted.")
    print(f"  Precision (Weighted): {report['weighted avg']['precision']:.4f} - Average Precision across all classes, weighted by support.")
    print(f"  Recall (Weighted): {report['weighted avg']['recall']:.4f} - Average Recall across all classes, weighted by support.")
    print(f"  F1 Score (Weighted): {report['weighted avg']['f1-score']:.4f} - Average F1-Score across all classes, weighted by support.")

    print("\nBusiness Insights:")
    print(f"- The model '{model_name}' achieved an overall accuracy of {report['accuracy']:.4f}.")

    # Identify classes with lower performance (example: based on F1-score)
    # Fix: Iterate through class labels directly from the report dictionary
    low_performance_classes = []
    for label_str in report:
        if label_str in ['accuracy', 'macro avg', 'weighted avg']:
            continue # Skip aggregated rows
        if report[label_str]['f1-score'] < 0.7: # Example threshold
            # Find the corresponding class name
            try:
                label_int = int(label_str)
                class_index = labels.index(label_int)
                low_performance_classes.append(class_names[class_index])
            except ValueError:
                 # Handle cases where label might not be an integer (e.g., if class_report includes other keys)
                 continue


    if low_performance_classes:
        print(f"- Pay close attention to the model's performance on classes: {', '.join(low_performance_classes)}. Lower F1-scores here might indicate challenges in correctly identifying these specific threat levels.")
    else:
        print("- The model demonstrates strong overall performance across all major metrics and classes.")

    # Highlight critical class performance
    if "Critical" in class_names:
        critical_index = class_names.index("Critical")
        critical_label_str = str(labels[critical_index])
        if critical_label_str in report: # Ensure 'Critical' label is in the report keys
            critical_recall = report[critical_label_str]['recall']
            critical_precision = report[critical_label_str]['precision']
            print(f"- For 'Critical' threats (Recall: {critical_recall:.4f}, Precision: {critical_precision:.4f}), the model is able to capture a significant portion of actual critical events (Recall), and when it flags an event as critical, it is often correct (Precision). However, further investigation into missed critical events might be warranted to minimize risk.")
            if critical_recall < 0.8: # Example threshold
                 print("- Improving recall for Critical threats should be a priority to ensure all high-severity events are detected.")
        else:
            print("- 'Critical' class metrics not found in the classification report.")


    # General recommendations
    print("- Consider deploying this model for automated threat detection, but maintain human oversight, especially for high-severity alerts.")
    print("- Continuously monitor model performance in a production environment as threat patterns can evolve.")
    print("- Investigate misclassified instances to understand the limitations and potential areas for model improvement or data enhancement.")

def generate_visualization_insight(data, anomaly_score_col, is_anomaly_col, model_name):
    """
    Generates dynamic comments and business insights based on visualization data.

    Args:
        data (pd.DataFrame): DataFrame containing the data used for visualization.
        anomaly_score_col (str): Column name for the anomaly score.
        is_anomaly_col (str): Column name for the binary anomaly flag (0 or 1).
        model_name (str): The name of the model being visualized.
    """
    print(f"\n--- Visualization Insights for {model_name} ---")

    if data.empty:
        print("  No data available for visualization insights.")
        return

    # Analyze Scatter Plot
    num_anomalies = data[is_anomaly_col].sum()
    num_normal = len(data) - num_anomalies
    print("\nScatter Plot Insight:")
    print(f"- The scatter plot visually separates points identified as anomalies ({num_anomalies} points, typically shown in red) from normal points ({num_normal} points, typically shown in blue).")
    print(f"- A clear separation between the red and blue points indicates that the model is effectively distinguishing between normal and anomalous behavior based on the chosen features.")

    # Analyze ROC Curve (requires ground truth in is_anomaly_col)
    if len(data[is_anomaly_col].unique()) > 1: # Check if there are both normal and anomaly points
        try:
            fpr, tpr, _ = roc_curve(data[is_anomaly_col], data[anomaly_score_col])
            roc_auc = auc(fpr, tpr)
            print("\nROC Curve Insight:")
            print(f"- The ROC curve shows the trade-off between the True Positive Rate (ability to detect actual anomalies) and the False Positive Rate (incorrectly flagging normal points as anomalies) at various threshold settings.")
            print(f"- An Area Under the ROC Curve (AUC) of {roc_auc:.4f} indicates the overall ability of the model to discriminate between positive (anomaly) and negative (normal) classes.")
            if roc_auc > 0.8: # Example threshold for good performance
                print("- An AUC value above 0.8 suggests good discriminative power.")
            elif roc_auc > 0.5:
                 print("- An AUC value above 0.5 suggests the model performs better than random guessing.")
            else:
                 print("- An AUC value below 0.5 suggests the model performs worse than random guessing.")

        except Exception as e:
            print(f"  Could not generate ROC curve insight: {e}")
    else:
        print("\nROC Curve Insight: Not applicable (requires both normal and anomaly ground truth labels).")


    # Analyze Precision-Recall Curve (requires ground truth in is_anomaly_col)
    if len(data[is_anomaly_col].unique()) > 1: # Check if there are both normal and anomaly points
        try:
            precision, recall, _ = precision_recall_curve(data[is_anomaly_col], data[anomaly_score_col])
            print("\nPrecision-Recall Curve Insight:")
            print(f"- The Precision-Recall curve highlights the trade-off between Precision (accuracy of positive predictions) and Recall (completeness of positive predictions) as the decision threshold is varied.")
            print("- This curve is particularly informative for imbalanced datasets, where the number of anomalies is much smaller than normal instances.")
            print("- A curve that stays high as recall increases indicates that the model can achieve high precision even when identifying a large proportion of actual anomalies.")

        except Exception as e:
            print(f"  Could not generate Precision-Recall curve insight: {e}")
    else:
        print("\nPrecision-Recall Curve Insight: Not applicable (requires both normal and anomaly ground truth labels).")


    print("\nBusiness Insights from Visualization:")
    if len(data[is_anomaly_col].unique()) > 1:
        if roc_auc > 0.8: # Example threshold for good discriminative power
            print(f"- The visualization for the {model_name} suggests that the model has good potential for identifying anomalous behavior. The clear separation in the scatter plot and the high AUC value are positive indicators.")
            print("- This model could be used to prioritize events for further investigation by security analysts.")
        elif roc_auc > 0.5:
             print(f"- The visualization for the {model_name} indicates that the model is somewhat effective at identifying anomalies, performing better than random guessing. However, there might be room for improvement in separating normal and anomalous instances.")
             print("- Further refinement of features or model parameters could enhance its utility in practice.")
        else:
             print(f"- The visualization for the {model_name} suggests that the model's ability to discriminate anomalies is limited. The scatter plot may show significant overlap, and the AUC is low.")
             print("- This model might not be suitable for direct deployment without significant improvements or a different approach.")

        # Comment on Precision-Recall based on the curve shape (qualitative for now)
        print("- The shape of the Precision-Recall curve provides insight into how many anomalies the model can find before its positive predictions become unreliable. A curve that drops sharply indicates that increasing recall comes at a high cost of precision (more false alarms).")

    else:
        print("- As the visualization data currently only contains one class (either all normal or all anomalies), it's difficult to provide detailed business insights based on the discriminative performance (ROC and Precision-Recall curves). Ensure your test data for visualization includes a representative mix of normal and anomalous instances with their ground truth labels.")

    print(f"\nEnd Visualization Insights for {model_name}.")
#--------------------------------------------
def visualizing_model_performance_pipeline(data, x, y,
                                           anomaly_score, is_anomaly,  best_unsupervised_model_name, title=None):
    """Pipeline to visualize scatter plot, ROC curve, and Precision-Recall curve."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model Performance Visualization\n")

    # Generate Scatter Plot
    create_scatter_plot(data, x, y, hue=is_anomaly, ax=ax1, x_label=x, y_label=y)

    # Generate ROC Curve
    create_roc_curve(data, anomaly_score, is_anomaly, ax=ax2)

    # Generate Precision-Recall Curve
    create_precision_recall_curve(data, anomaly_score, is_anomaly, ax=ax3)

    # Adjust layout and set title
    plt.tight_layout()
    if title:
       plt.suptitle(title)
    plt.show()
    generate_visualization_insight(data, anomaly_score, is_anomaly, best_unsupervised_model_name)
#----------------------------------------------------
def print_model_performance_report(model_name, model_y_test, y_model_pred):
    """Print comprehensive model performance report including classification report, confusion matrix, and aggregated metrics."""

    print("\n" + model_name + " classification_report:\n")
    report = classification_report(model_y_test, y_model_pred, output_dict=True)
    class_names, labels = get_label_class_name(model_y_test, y_model_pred)

    # Create DataFrame from classification report and rename index
    report_df = pd.DataFrame(report).T
    # Only rename the class labels (0, 1, 2, 3), not the aggregated rows
    class_labels_to_rename = [str(label) for label in labels]
    rename_dict = {label: class_names[i] for i, label in enumerate(class_labels_to_rename)}

    report_df = report_df.rename(index=rename_dict)

    display(round(report_df,4))


    cm = confusion_matrix(model_y_test, y_model_pred, labels=labels)
    confusion_matrix_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("\n" + model_name + " Confusion Matrix:\n")
    plt.figure(figsize=(4, 3))
    heatmap = sns.heatmap(
            round(confusion_matrix_df,2),
            annot=True,
            fmt='d',
            cmap=custom_cmap,
            xticklabels=class_names,
            yticklabels=class_names
    )

    # Get the axes object
    ax = heatmap.axes
    # Set the x-axis label
    ax.set_xlabel("Predicted Class")

    # Move the x-axis label to the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    #Set the y-axis label (title)
    ax.set_ylabel("Actual Class")

    # Set the overall plot title
    plt.title("Confusion Matrix\n")

    # Adjust subplot parameters to give more space at the top
    plt.subplots_adjust(top=0.85)
    # Display the plot
    plt.show()

    print("\n" + model_name + " Agreggated Peformance Metrics:\n")
    metrics_dic = get_metrics(model_y_test, y_model_pred, report)
    metrics_df = pd.DataFrame(metrics_dic.items(), columns=['Metric', 'Value'])
    display(metrics_df)
    plot_metrics_bell_curve(model_name, model_y_test, y_model_pred)
    print("\nOverall Model Accuracy : ", metrics_dic.get("Overall Model Accuracy ", 0))
    # Call the generate_performance_insight function for the best model
    generate_performance_insight(model_name, report, model_y_test, y_model_pred )

    #select tbe best unsupersised model end generate the reports insight

    #generate_visualization_insight(data, anomaly_score, is_anomaly, best_unsupervised_model_name)
    return  metrics_dic
#-----------------------------------------------------
def build_dense_autoencoder(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(input_dim, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    # explicit loss object to be safe when serializing
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model


def build_lstm_autoencoder(timesteps, features):
    inputs = Input(shape=(timesteps, features))
    encoded = LSTM(128, activation='relu', return_sequences=False)(inputs)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(features, activation='linear', return_sequences=True)(decoded)
    model = Model(inputs, decoded)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return model


def load_dataset():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset.")
    # ensure integer labels
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]
    return X, y

def get_best_unsupervised_model_corrected(X_test_scaled, y_test, unsupervised_models):
    """
    Evaluates all unsupervised models and identifies the best one based on a chosen metric (AUC or F1).

    Args:
        X_test_scaled (np.ndarray): Scaled test features.
        y_test (pd.Series): True labels for the test set.
        unsupervised_models (dict): Dictionary containing all trained unsupervised models and train_X.

    Returns:
        tuple: (best_model_name, best_anomaly_score, best_is_anomaly)
               best_model_name (str): Name of the best performing model.
               best_anomaly_score (np.ndarray): Anomaly scores from the best model.
               best_is_anomaly (np.ndarray): Binary anomaly labels (1 for anomaly, 0 for normal) from the best model.
    """
    performance_data = get_unsupervised_model_performance_data_corrected(X_test_scaled, y_test, unsupervised_models)

    best_model_name = None
    best_performance = -1 # Initialize with a value lower than any expected metric
    best_anomaly_score = None
    best_is_anomaly = None

    for name, (anomaly_score, is_anomaly, performance_metric) in performance_data.items():
        if performance_metric is not None:
            if performance_metric > best_performance:
                best_performance = performance_metric
                best_model_name = name
                best_anomaly_score = anomaly_score
                best_is_anomaly = is_anomaly

    if best_model_name:
        log(f"\nBest unsupervised model: {best_model_name} with performance: {best_performance:.4f}")
    else:
        log("\nCould not identify the best unsupervised model.")
    print("\n--- End of Unsupervised Model Visualizations and Insights ---")
    return best_model_name, best_anomaly_score, best_is_anomaly

def evaluate_unsupervised_model_corrected(name, model, X_test_scaled, y_test, unsupervised_models):
    """
    Evaluates an unsupervised model for anomaly detection.

    Args:
        name (str): Name of the model.
        model: The trained unsupervised model.
        X_test_scaled (np.ndarray): Scaled test features.
        y_test (pd.Series): True labels for the test set.
        unsupervised_models (dict): Dictionary containing all trained unsupervised models and train_X.

    Returns:
        tuple: (anomaly_score, is_anomaly, performance_metric)
               anomaly_score (np.ndarray): Anomaly scores from the model.
               is_anomaly (np.ndarray): Binary anomaly labels (1 for anomaly, 0 for normal).
               performance_metric (float): A chosen metric for evaluation (AUC or F1).
                                           Returns None if evaluation is not possible.
    """
    anomaly_score = None
    is_anomaly = None
    performance_metric = None
    y_test_binary_anomaly = (y_test.reset_index(drop=True) >= 2).astype(int) # Define anomaly based on Threat Level

    try:
        if hasattr(model, 'decision_function'):
            # For models with decision function, invert for consistency: higher score = more anomalous
            anomaly_score = -model.decision_function(X_test_scaled)
            # For models with decision function, we can also get predicted labels based on contamination
            if hasattr(model, 'predict'):
                 is_anomaly = (model.predict(X_test_scaled) == -1).astype(int) # -1 is outlier for IsolationForest, LOF, OneClassSVM

        elif hasattr(model, 'predict'):
             # DBSCAN: predicts cluster labels. -1 is noise (anomaly)
             if name == "DBSCAN":
                 # For DBSCAN on test data, use nearest neighbor assignment and map -1 to anomaly
                 # Need to re-calculate this as it's not stored in the model object
                 train_X = unsupervised_models.get('train_X')
                 if train_X is not None and hasattr(model, 'labels_'):
                      nbrs = NearestNeighbors(n_neighbors=1).fit(train_X)
                      nn_idx = nbrs.kneighbors(X_test_scaled, return_distance=False)[:, 0]
                      db_labels_train = model.labels_
                      assigned_train_labels = db_labels_train[nn_idx]
                      is_anomaly = (assigned_train_labels == -1).astype(int)
                      # Using binary label as score proxy for plotting if no continuous score
                      anomaly_score = is_anomaly.copy().astype(float)
                 else:
                    log(f"Warning: Cannot evaluate DBSCAN on test set without train_X and model labels_.")
                    return None, None, None # Cannot evaluate without necessary components
             # KMeans: distance to centroid as anomaly score
             elif name == "KMeans":
                 test_k_labels = model.predict(X_test_scaled)
                 anomaly_score = np.linalg.norm(X_test_scaled - model.cluster_centers_[test_k_labels], axis=1)
                 # No direct binary prediction from KMeans predict
                 is_anomaly = None

        # Handle Autoencoders (Dense and LSTM)
        if name in ["Autoencoder", "LSTM(Classifier)"]:
            if name == "LSTM(Classifier)":
                timesteps = 1 # Assuming timestep=1 as in training
                input_dim = X_test_scaled.shape[1]
                X_test_seq = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, input_dim))
                reconstructions = model.predict(X_test_seq, verbose=0)
                anomaly_score = np.mean((X_test_seq - reconstructions) ** 2, axis=(1, 2))
            elif name == "Autoencoder": # Dense Autoencoder
                reconstructions = model.predict(X_test_scaled, verbose=0)
                anomaly_score = np.mean((X_test_scaled - reconstructions) ** 2, axis=1)
            # No direct binary prediction from Autoencoders predict
            is_anomaly = None

        # Evaluate performance if possible
        # Check if y_test_binary_anomaly has at least two unique classes before calculating metrics
        if anomaly_score is not None and len(np.unique(y_test_binary_anomaly)) > 1:
            # Use Average Precision Score for models with continuous anomaly scores
            if name not in ["DBSCAN"] and anomaly_score is not None: # DBSCAN has binary prediction primarily
                try:
                    performance_metric = average_precision_score(y_test_binary_anomaly, anomaly_score)
                    log(f"  {name} performance (Average Precision): {performance_metric:.4f}")
                except Exception as e:
                    log(f"  Could not calculate Average Precision for {name}: {e}")
                    # Fallback to AUC if Average Precision fails and AUC is applicable
                    if name not in ["DBSCAN"] and anomaly_score is not None:
                        try:
                            performance_metric = roc_auc_score(y_test_binary_anomaly, anomaly_score)
                            log(f"  {name} performance (AUC - Fallback): {performance_metric:.4f}")
                        except Exception as e_auc:
                            log(f"  Could not calculate AUC for {name}: {e_auc}")
                            performance_metric = None # Cannot evaluate with AUC or Average Precision


            # Use F1-score for models with binary anomaly predictions (like DBSCAN)
            if is_anomaly is not None:
                 # Check if both classes exist in predictions and true labels for F1
                 if len(np.unique(is_anomaly)) > 1 and len(np.unique(y_test_binary_anomaly)) > 1:
                     try:
                         f1 = f1_score(y_test_binary_anomaly, is_anomaly)
                         # For DBSCAN, F1 is the primary metric here
                         if name == "DBSCAN":
                             performance_metric = f1
                             log(f"  {name} performance (F1): {performance_metric:.4f}")
                         # For other models with binary predictions, F1 is an additional metric
                     except Exception as e:
                         log(f"  Could not calculate F1 for {name}: {e}")
                 else:
                     log(f"  Skipping F1 calculation for {name}: Not enough unique classes in predictions or true labels.")
        elif len(np.unique(y_test_binary_anomaly)) <= 1:
             log(f"  Skipping performance metric calculation for {name}: Test set contains only one class of anomaly labels.")


    except Exception as e:
        log(f"Error evaluating {name}: {e}")
        performance_metric = None

    return anomaly_score, is_anomaly, performance_metric

def get_unsupervised_model_performance_data_corrected(X_test_scaled, y_test, unsupervised_models):
    """
    Evaluates all unsupervised models and returns a dictionary containing
    their anomaly scores, binary anomaly labels, and performance metrics.

    Args:
        X_test_scaled (np.ndarray): Scaled test features.
        y_test (pd.Series): True labels for the test set.
        unsupervised_models (dict): Dictionary containing all trained unsupervised models and train_X.

    Returns:
        dict: A dictionary where keys are model names and values are tuples
              (anomaly_score, is_anomaly, performance_metric).
              anomaly_score (np.ndarray): Anomaly scores from the model.
              is_anomaly (np.ndarray): Binary anomaly labels (1 for anomaly, 0 for normal).
              performance_metric (float): A chosen metric for evaluation (e.g., AUC or F1).
    """
    unsupervised_model_names = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "DBSCAN", "KMeans", "Autoencoder", "LSTM(Classifier)"]
    model_mapping = {
        "IsolationForest": unsupervised_models.get('iso'),
        "OneClassSVM": unsupervised_models.get('ocsvm'),
        "LocalOutlierFactor": unsupervised_models.get('lof'),
        "DBSCAN": unsupervised_models.get('dbscan'),
        "KMeans": unsupervised_models.get('kmeans'),
        "Autoencoder": unsupervised_models.get('dense_ae'),
        "LSTM(Classifier)": unsupervised_models.get('lstm_ae') # Assuming this maps to lstm_ae
    }

    performance_data = {}
    # y_test_binary_anomaly is calculated inside evaluate_unsupervised_model_corrected

    for name in unsupervised_model_names:
        model = model_mapping.get(name)
        if model is None:
            log(f"Skipping {name}: Model not found in unsupervised_models dictionary.")
            continue

        log(f"Evaluating {name}...")
        anomaly_score, is_anomaly, performance_metric = evaluate_unsupervised_model_corrected(
            name, model, X_test_scaled, y_test, unsupervised_models
        )

        performance_data[name] = (anomaly_score, is_anomaly, performance_metric)

    return performance_data


# 1. Refactor Data Loading and Splitting
def load_and_split_data(data_path, label_col, test_size, random_state, df = None):
    """
    Loads the dataset and splits it into training and testing sets.
    """
    log("Loading dataset...")
    #validate df
    if df is None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        #df = pd.read_csv(data_path)
        else:
            df = pd.read_csv(data_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")
    df[label_col] = df[label_col].astype(int) # ensure integer labels
    X = df.drop(columns=[label_col])
    y = df[label_col]

    log("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    return X_train, X_test, y_train, y_test, X.columns # Return original column names

# 2. Refactor Data Scaling
def scale_data(X_train, X_test):
    """
    Scales the training and testing data using StandardScaler.
    Returns the fitted scaler and the scaled data.
    """
    log("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

# 3. Refactor Anomaly Feature Extraction
def extract_unsupervised_features(X_train_scaled, X_test_scaled, y_train, unsupervised_params, random_state):
    """
    Train unsupervised detectors and produce anomaly features for train/test.
    Returns features_train (DataFrame), features_test (DataFrame), unsupervised_models (dict).
    unsupervised_models will include 'train_X' stored as numpy array to support inference mapping.
    """
    log("Extracting anomaly features...")
    features_train = pd.DataFrame(index=np.arange(X_train_scaled.shape[0]))
    features_test = pd.DataFrame(index=np.arange(X_test_scaled.shape[0]))

    # Isolation Forest
    log("Fitting IsolationForest...")
    iso = IsolationForest(contamination=unsupervised_params['contamination'], random_state=random_state)
    iso.fit(X_train_scaled)
    features_train['iso_df'] = iso.decision_function(X_train_scaled)
    features_test['iso_df'] = iso.decision_function(X_test_scaled)

    # One-Class SVM
    log("Fitting One-Class SVM...")
    ocsvm = OneClassSVM(nu=unsupervised_params['contamination'], kernel='rbf', gamma='scale')
    ocsvm.fit(X_train_scaled)
    features_train['ocsvm_df'] = ocsvm.decision_function(X_train_scaled)
    features_test['ocsvm_df'] = ocsvm.decision_function(X_test_scaled)

    # Local Outlier Factor (novelty True so we can use decision_function/predict)
    log("Fitting Local Outlier Factor (LOF)...")
    lof = LocalOutlierFactor(n_neighbors=20, contamination=unsupervised_params['contamination'], novelty=True)
    lof.fit(X_train_scaled)
    features_train['lof_df'] = lof.decision_function(X_train_scaled)
    features_test['lof_df'] = lof.decision_function(X_test_scaled)

    # DBSCAN anomaly flag with nearest neighbor assignment for test set
    log("Running DBSCAN clustering...")
    db = DBSCAN(eps=unsupervised_params['dbscan_eps'], min_samples=unsupervised_params['dbscan_min_samples'])
    db_labels_train = db.fit_predict(X_train_scaled)  # labels for training samples
    # nearest neighbor mapping from test samples -> nearest train index
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_train_scaled) # Use unsupervised_models['train_X']
    nn_idx = nbrs.kneighbors(X_test_scaled, return_distance=False)[:, 0]
    assigned_train_labels = db_labels_train[nn_idx]
    features_train['dbscan_anomaly'] = (db_labels_train == -1).astype(float)
    features_test['dbscan_anomaly'] = (assigned_train_labels == -1).astype(float)

    # KMeans distances to cluster centers
    log("Running KMeans clustering...")
    kmeans = KMeans(n_clusters=unsupervised_params['kmeans_clusters'], random_state=random_state)
    kmeans.fit(X_train_scaled)
    train_k_labels = kmeans.predict(X_train_scaled)
    test_k_labels = kmeans.predict(X_test_scaled)
    train_distances = np.linalg.norm(X_train_scaled - kmeans.cluster_centers_[train_k_labels], axis=1)
    test_distances = np.linalg.norm(X_test_scaled - kmeans.cluster_centers_[test_k_labels], axis=1)
    features_train['kmeans_dist'] = train_distances
    features_test['kmeans_dist'] = test_distances

    # Dense Autoencoder reconstruction error
    log("Training Dense Autoencoder...")
    input_dim = X_train_scaled.shape[1]
    dense_ae = build_dense_autoencoder(input_dim)
    # Define "normal" mask (update threshold to suit your label encoding)
    normal_mask = (y_train <= 1).to_numpy() if hasattr(y_train, "to_numpy") else (y_train <= 1)
    X_ae_train = X_train_scaled[normal_mask]
    # Only use validation_split if we have enough samples
    fit_kwargs = {"epochs": unsupervised_params['autoencoder_epochs'], "batch_size": unsupervised_params['autoencoder_batch'], "verbose": 0,
                  "callbacks": [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]}
    if len(X_ae_train) > 50:
        fit_kwargs["validation_split"] = 0.1
    dense_ae.fit(X_ae_train, X_ae_train, **fit_kwargs)
    features_train['ae_mse'] = np.mean((X_train_scaled - dense_ae.predict(X_train_scaled, verbose=0)) ** 2, axis=1)
    features_test['ae_mse'] = np.mean((X_test_scaled - dense_ae.predict(X_test_scaled, verbose=0)) ** 2, axis=1)

    # LSTM Autoencoder reconstruction error (reshape sequences with timesteps=1)
    log("Training LSTM Autoencoder...")
    timesteps = 1
    input_dim = X_train_scaled.shape[1]
    X_train_seq = X_train_scaled.reshape((X_train_scaled.shape[0], timesteps, input_dim))
    X_test_seq = X_test_scaled.reshape((X_test_scaled.shape[0], timesteps, input_dim))
    lstm_ae = build_lstm_autoencoder(timesteps, input_dim)
    fit_kwargs_lstm = {"epochs": unsupervised_params['lstm_epochs'], "batch_size": unsupervised_params['lstm_batch'], "verbose": 0,
                       "callbacks": [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]}
    if len(X_ae_train) > 50:
        fit_kwargs_lstm["validation_split"] = 0.1
    lstm_ae.fit(X_train_seq[normal_mask], X_train_seq[normal_mask], **fit_kwargs_lstm)
    features_train['lstm_mse'] = np.mean((X_train_seq - lstm_ae.predict(X_train_seq, verbose=0)) ** 2, axis=(1, 2))
    features_test['lstm_mse'] = np.mean((X_test_seq - lstm_ae.predict(X_test_seq, verbose=0)) ** 2, axis=(1, 2))

    unsupervised_models = {
        'iso': iso, 'ocsvm': ocsvm, 'lof': lof, 'dbscan': db,
        'kmeans': kmeans, 'dense_ae': dense_ae, 'lstm_ae': lstm_ae,
        'train_X': np.asarray(X_train_scaled)  # save training X scaled for inference mapping
    }

    return features_train, features_test, unsupervised_models

# 4. Refactor Stacked Model Training
def train_stacked_model(X_train_ext, X_test_ext, y_train, random_state):
    """
    Trains the Random Forest base model and the Gradient Boosting meta-learner.
    Returns the trained base and meta models.
    """
    log("Training stacked supervised model...")
    log("  Training RandomForest base model...")
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_ext, y_train)
    rf_train_proba = rf.predict_proba(X_train_ext)
    rf_test_proba = rf.predict_proba(X_test_ext)

    X_train_stack = np.hstack([X_train_ext.values, rf_train_proba])
    X_test_stack = np.hstack([X_test_ext.values, rf_test_proba])

    log("  Training GradientBoosting meta model...")
    gb = GradientBoostingClassifier(n_estimators=200, random_state=random_state)
    gb.fit(X_train_stack, y_train)

    return rf, gb, X_test_stack

# 5. Refactor Stacked Model Evaluation
def evaluate_stacked_model(model_name, y_test, y_pred, model_output_dir):
    """
    Evaluates the stacked model and saves the metrics.
    """
    log(f"Evaluating {model_name}...")
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Use the existing print_model_performance_report for comprehensive output
    print_model_performance_report(model_name, y_test, y_pred)

    # Save metrics to JSON
    metrics_path = os.path.join(model_output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "classification_report": report_dict,
                   "confusion_matrix": cm.tolist()}, f, indent=4)
    log(f"Saved evaluation metrics to {metrics_path}")

def visualize_all_unsupervised_models(X_test_scaled, y_test, unsupervised_models, X_columns):
    """
    Visualizes the performance of each unsupervised model using scatter plots,
    ROC curves, and Precision-Recall curves in separate subplots.
    """
    log("Visualizing performance for all unsupervised models...")

    unsupervised_model_names_eval = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "DBSCAN", "KMeans", "Autoencoder", "LSTM(Classifier)"]
    model_mapping_eval = {
        "IsolationForest": unsupervised_models.get('iso'),
        "OneClassSVM": unsupervised_models.get('ocsvm'),
        "LocalOutlierFactor": unsupervised_models.get('lof'),
        "DBSCAN": unsupervised_models.get('dbscan'),
        "KMeans": unsupervised_models.get('kmeans'),
        "Autoencoder": unsupervised_models.get('dense_ae'),
        "LSTM(Classifier)": unsupervised_models.get('lstm_ae')
    }

    # Prepare the true anomaly labels for the test set
    y_test_binary_anomaly = (y_test.reset_index(drop=True) >= 2).astype(int)

    for name in unsupervised_model_names_eval:
        model = model_mapping_eval.get(name)
        if model is None:
            log(f"Skipping visualization for {name}: Model not found.")
            continue

        log(f"  Generating visualization data for {name}...")

        # Get anomaly scores and predicted anomalies (if available) for the current model
        # Use evaluate_unsupervised_model_corrected to get the necessary data for visualization
        anomaly_score, is_anomaly, _ = evaluate_unsupervised_model_corrected(
             name, model, X_test_scaled, y_test, unsupervised_models # Pass unsupervised_models here
        )

        # Check if anomaly_score is available
        if anomaly_score is None:
            log(f"Skipping visualization for {name}: Anomaly score not available.")
            continue

        # Prepare data for visualization
        viz_data = pd.DataFrame(X_test_scaled, columns=X_columns)
        viz_data['true_anomaly'] = y_test_binary_anomaly
        viz_data['anomaly_score'] = anomaly_score

        if is_anomaly is not None:
            viz_data['predicted_anomaly'] = is_anomaly
        else:
             # Use median as a simple threshold for visualization if binary labels are not directly available
            threshold = np.median(anomaly_score)
            viz_data['predicted_anomaly'] = (anomaly_score > threshold).astype(int)


        # Visualize the model's performance using the existing pipeline
        log(f"  Visualizing performance for {name}...")
        # Pass the correct arguments to visualizing_model_performance_pipeline
        visualizing_model_performance_pipeline(
            viz_data, # Use the prepared viz_data DataFrame
            'Threat Score' if 'Threat Score' in viz_data.columns else viz_data.columns[0], # Use 'Threat Score' if available, otherwise the first column
            'Impact Score' if 'Impact Score' in viz_data.columns else viz_data.columns[1],  # Use 'Impact Score' if available, otherwise the second column
            'anomaly_score', # Pass the anomaly score column name
            'true_anomaly', # Pass the true anomaly label column name
            name, # Pass the model name
            title=f"{name} Performance Visualization" # Set the title
        )

    log("Finished visualizing performance for all unsupervised models.")
    return viz_data


# 6.  Unsupervised Model Evaluation and Visualization
def evaluate_and_visualize_unsupervised_models(X_test_scaled, y_test, unsupervised_models, X_columns):
    """
    Evaluates all unsupervised models, identifies the best, and visualizes its performance.
    Also generates a comparative performance analysis of all unsupervised models.
    """
    print("\n---Unsupervised Model Evaluation and Identification of the Best Model ---\n")

    log("Evaluating unsupervised models and identifying the best...")

    #---------------------------------------------------
    # Call the new function to visualize all unsupervised models
    # Ensure X_test_scaled, y_test, unsupervised_models, and X_columns are available from the previous run

    if 'X_test_scaled' in locals() and 'y_test' in locals() and 'unsupervised_models' in locals() and 'X_columns' in locals():
        visualize_all_unsupervised_models(X_test_scaled, y_test, unsupervised_models, X_columns)
    else:
        print("Required variables (X_test_scaled, y_test, unsupervised_models, X_columns) are not available. Please run the main pipeline first.")
    #-------------------------------------------------
    # Call get_best_unsupervised_model_corrected with only 4 arguments
    best_unsupervised_model_name, best_anomaly_score, best_is_anomaly = get_best_unsupervised_model_corrected(X_test_scaled, y_test, unsupervised_models)
    print(f"\nBest unsupervised model for anomaly detection: {best_unsupervised_model_name}")

    # Prepare data for visualization of the best unsupervised model
    viz_data = pd.DataFrame(X_test_scaled, columns=X_columns)
    viz_data['true_anomaly'] = (y_test.reset_index(drop=True) >= 2).astype(int)
    viz_data['anomaly_score'] = best_anomaly_score

    if best_is_anomaly is not None:
        viz_data['predicted_anomaly'] = best_is_anomaly
    else:
        # Use median as a simple threshold for visualization if binary labels are not directly available
        if best_anomaly_score is None:
            print("[ERROR] No valid unsupervised model was selected. Skipping visualization.")
            # Return None to indicate no visualization data was generated
            viz_data = None
        else:
            threshold = np.median(best_anomaly_score)
            viz_data['predicted_anomaly'] = (best_anomaly_score > threshold).astype(int)

    # Visualize the best unsupervised model's performance
    if viz_data is not None:
        log(f"Visualizing performance for the best unsupervised model ({best_unsupervised_model_name})...")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{best_unsupervised_model_name} Performance Visualization\n")

        # Check if 'Threat Score' and 'Impact Score' columns exist before plotting
        if 'Threat Score' in viz_data.columns and 'Impact Score' in viz_data.columns:
            create_scatter_plot(viz_data, 'Threat Score', 'Impact Score', hue='true_anomaly', ax=ax1)
        else:
            log("Warning: 'Threat Score' or 'Impact Score' not found in visualization data. Skipping scatter plot.")
            # Optionally, create a placeholder or plot different columns
            ax1.set_title("Scatter Plot Skipped (Missing Columns)")
            ax1.text(0.5, 0.5, "Missing 'Threat Score' or 'Impact Score'", horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)


        # Ensure anomaly_score column exists before creating ROC and PR curves
        if 'anomaly_score' in viz_data.columns and viz_data['anomaly_score'] is not None:
             create_roc_curve(viz_data, 'anomaly_score', 'true_anomaly', ax=ax2)
             create_precision_recall_curve(viz_data, 'anomaly_score', 'true_anomaly', ax=ax3)
        else:
             log("Warning: Anomaly score not available for visualization. Skipping ROC and Precision-Recall curves.")
             ax2.set_title("ROC Curve Skipped (No Anomaly Score)")
             ax2.text(0.5, 0.5, "No Anomaly Score Available", horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
             ax3.set_title("Precision-Recall Curve Skipped (No Anomaly Score)")
             ax3.text(0.5, 0.5, "No Anomaly Score Available", horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)


        plt.tight_layout()
        plt.show()
        if 'anomaly_score' in viz_data.columns and viz_data['anomaly_score'] is not None:
            generate_visualization_insight(viz_data, 'anomaly_score', 'true_anomaly', best_unsupervised_model_name)
        else:
             log("Skipping visualization insight generation due to missing anomaly scores.")


    # 7. Comparative Performance Analysis and Visualization of Unsupervised Models
    log("Generating comparative performance analysis...")
    unsupervised_performance = {}
    unsupervised_model_names_eval = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "DBSCAN", "KMeans", "Autoencoder", "LSTM(Classifier)"]
    model_mapping_eval = {
        "IsolationForest": unsupervised_models.get('iso'),
        "OneClassSVM": unsupervised_models.get('ocsvm'),
        "LocalOutlierFactor": unsupervised_models.get('lof'),
        "DBSCAN": unsupervised_models.get('dbscan'),
        "KMeans": unsupervised_models.get('kmeans'),
        "Autoencoder": unsupervised_models.get('dense_ae'),
        "LSTM(Classifier)": unsupervised_models.get('lstm_ae')
    }

    for name in unsupervised_model_names_eval:
        model = model_mapping_eval.get(name)
        if model is None:
            log(f"Skipping evaluation for {name}: Model not found.")
            continue

        log(f"  Evaluating {name}...")
        # Call evaluate_unsupervised_model_corrected with only 4 arguments
        anomaly_score, is_anomaly, performance_metric = evaluate_unsupervised_model_corrected(
            name, model, X_test_scaled, y_test, unsupervised_models # Add unsupervised_models here
        )

        if performance_metric is not None:
            unsupervised_performance[name] = performance_metric
            log(f"    {name} performance ({'AUC' if name not in ['DBSCAN'] else 'F1'}): {performance_metric:.4f}")
        else:
            log(f"    Could not evaluate {name} performance.")

    # Display comparative table
    log("  Displaying unsupervised model performance table...")
    unsupervised_performance_df = pd.DataFrame(list(unsupervised_performance.items()), columns=['Model', 'Performance Metric'])
    display(unsupervised_performance_df)

    # Display comparative chart (horizontal bar chart, sorted)
    log("  Displaying unsupervised model performance chart...")
    plt.figure(figsize=(7, 3))

    # Sort models by performance metric (descending)
    order = unsupervised_performance_df.sort_values(
        'Performance Metric', ascending=False
    )['Model']

    sns.barplot(
        y='Model',
        x='Performance Metric',
        data=unsupervised_performance_df,
        order=order,
        palette='viridis'
    )

    plt.title('Unsupervised Model Performance Comparison')
    plt.xlabel('Performance Metric')
    plt.ylabel('Unsupervised Model')
    plt.tight_layout()
    plt.show()

    # Generate insights from comparative analysis
    print("\n--- Insights from Unsupervised Model Performance Comparison ---")

    # Filter out models where performance metric was None
    evaluable_models_df = unsupervised_performance_df.dropna(subset=['Performance Metric'])

    if not evaluable_models_df.empty:
        # Find the best performing model among those successfully evaluated
        best_model_row = evaluable_models_df.loc[evaluable_models_df['Performance Metric'].idxmax()]
        best_model_name_comp = best_model_row['Model']
        best_performance_comp = best_model_row['Performance Metric']

        # Find the worst performing model among those successfully evaluated
        worst_model_row = evaluable_models_df.loc[evaluable_models_df['Performance Metric'].idxmin()]
        worst_model_name_comp = worst_model_row['Model']
        worst_performance_comp = worst_model_row['Performance Metric']


        print(f"Best Performing Model: {best_model_name_comp} (Metric: {best_performance_comp:.4f})")
        print(f"Worst Performing Model: {worst_model_name_comp} (Metric: {worst_performance_comp:.4f})")

        autoencoder_performance_df = evaluable_models_df[evaluable_models_df['Model'].isin(['Autoencoder', 'LSTM(Classifier)'])]
        autoencoder_performance = autoencoder_performance_df['Performance Metric'].mean() if not autoencoder_performance_df.empty else float('nan')

        clustering_performance_df = evaluable_models_df[evaluable_models_df['Model'].isin(['DBSCAN', 'KMeans'])]
        clustering_performance = clustering_performance_df['Performance Metric'].mean() if not clustering_performance_df.empty else float('nan')

        tree_based_performance_df = evaluable_models_df[evaluable_models_df['Model'].isin(['IsolationForest', 'LocalOutlierFactor'])]
        tree_based_performance = tree_based_performance_df['Performance Metric'].mean() if not tree_based_performance_df.empty else float('nan')

        svm_performance_df = evaluable_models_df[evaluable_models_df['Model'].isin(['OneClassSVM'])]
        svm_performance = svm_performance_df['Performance Metric'].mean() if not svm_performance_df.empty else float('nan')


        print("\nPerformance Comparison by Model Type:")
        print(f"- Autoencoder Models (Mean Metric): {autoencoder_performance:.4f}")
        print(f"- Clustering Models (Mean Metric): {clustering_performance:.4f}")
        print(f"- Tree-based Models (Mean Metric): {tree_based_performance:.4f}")
        print(f"- One-Class SVM (Metric): {svm_performance:.4f}")

        print("\nPotential Reasons for Performance Differences:")
        print("- Autoencoders (especially LSTM) might be better at capturing complex, non-linear relationships and temporal patterns in the data, which is relevant for anomaly detection in time-series-like threat data.")
        print("- Clustering methods like DBSCAN and KMeans might struggle if anomalies don't form distinct clusters or if the concept of 'normal' varies significantly.")
        print("- Tree-based methods (Isolation Forest, LOF) are often effective but might be sensitive to feature scaling and hyperparameters.")
        print("- One-Class SVM is designed for novelty detection but its performance can be sensitive to the choice of kernel and hyperparameters.")
        print("- The specific nature of the anomalies in this dataset (e.g., their density, shape, and relationship to normal data) heavily influences which model type is most suitable.")

        baseline_auc = 0.5 # Random guessing for AUC
        print("\nComparison to Baseline:")
        print(f"- The best model ({best_model_name_comp}) achieved a metric of {best_performance_comp:.4f}.")
        auc_models_df = evaluable_models_df[evaluable_models_df['Model'].isin(['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'KMeans', 'Autoencoder', 'LSTM(Classifier)'])] # Include autoencoders and LSTM for AUC comparison
        if not auc_models_df.empty:
            best_auc = auc_models_df['Performance Metric'].max()
            print(f"- Compared to a random guessing baseline (AUC = {baseline_auc}), the best AUC model performs {'better' if best_auc > baseline_auc else 'worse' if best_auc < baseline_auc else 'similarly'}.")
        else:
            print("- No AUC-based models were evaluated for baseline comparison.")


        print("\nImplications for the Stacked Model:")
        print(f"- The anomaly features generated by the unsupervised models, particularly the best performing ones like {best_model_name_comp}, will be fed into the supervised meta-learner.")
        print("- Higher quality anomaly features (from better performing unsupervised models) are likely to improve the performance of the stacked supervised model, especially in distinguishing between different threat levels.")
        print("- The meta-learner can learn to leverage the strengths of different anomaly detection signals from the various unsupervised models.")
        print("- While the individual unsupervised models might not be perfect anomaly detectors on their own, their outputs serve as valuable additional features for the supervised layer.")
    else:
        print("No unsupervised models were successfully evaluated for comparative analysis.")
    print("\n--- End of Insights ---")
    #return best_model_name_comp, y_test, y_pred, metrics_path, anomaly_score, is_anomaly
    #returbest_unsupervised_model_name, best_anomaly_score, best_is_anomaly
    return viz_data

# 7. Refactor Saving Models and Metrics
def save_all_models_and_metrics(output_dir, scaler, base_model, meta_model, unsupervised_models, metrics_path):
    """
    Saves the scaler, all trained models, and evaluation metrics.
    """
    log("Saving models and metrics...")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(base_model, os.path.join(output_dir, "rf_base.joblib"))
    joblib.dump(meta_model, os.path.join(output_dir, "gb_meta.joblib"))
    # save classical unsupervised models
    for name in ['iso', 'ocsvm', 'lof', 'dbscan', 'kmeans']:
        joblib.dump(unsupervised_models[name], os.path.join(output_dir, f"{name}.joblib"))
    # save train_X_scaled for DBSCAN mapping
    np.save(os.path.join(output_dir, "train_X_scaled.npy"), unsupervised_models['train_X'])
    # save Keras models in native format (.keras)
    dense_path = os.path.join(output_dir, "dense_autoencoder.keras")
    lstm_path = os.path.join(output_dir, "lstm_autoencoder.keras")
    unsupervised_models['dense_ae'].save(dense_path)
    unsupervised_models['lstm_ae'].save(lstm_path)
    log(f"  Scaler and ALL models saved in '{output_dir}'")

    # Metrics are already saved in evaluate_stacked_model, just log the path
    log(f"Evaluation metrics saved in {metrics_path}")

#-----------------------------------------
#display input and the output
# data frame of the stacked cliassifier
#-----------------------------------------

def display_model_input_output(X_test_ext, X_test_stack, y_test, y_pred, X_columns, viz_data):
    """
    Displays the input DataFrame for the stacked model and the final output DataFrames
    including original features, anomaly features, RF proba, true labels, predictions and viz_data.

    Args:
        X_test_ext (pd.DataFrame): Extended test features (original + anomaly features).
        X_test_stack (np.ndarray): Stacked test features (original + anomaly + RF proba).
        y_test (pd.Series): True labels for the test set.
        y_pred (np.ndarray): Predicted labels from the stacked model.
        X_columns (list): List of original column names.
        viz_data (pd.DataFrame) that contains Threat Score, true_anomaly, anomaly_score, predicted_anomaly
    """
    log("Displaying Stacked Model Input and Output DataFrames...")

    # Create DataFrame for the stacked input (including RF proba)
    # Get the number of original features and anomaly features
    num_original_features = len(X_columns)
    num_anomaly_features = X_test_ext.shape[1] - num_original_features
    num_rf_proba_features = X_test_stack.shape[1] - X_test_ext.shape[1]

    # Create column names for the stacked DataFrame
    original_cols = list(X_columns)
    anomaly_cols = [f'anomaly_{i}' for i in range(num_anomaly_features)]
    rf_proba_cols = [f'rf_proba_{i}' for i in range(num_rf_proba_features)]
    stacked_cols = original_cols + anomaly_cols + rf_proba_cols

    X_test_stack_df = pd.DataFrame(X_test_stack, columns=stacked_cols, index=X_test_ext.index)
    X_test_stack_df["true_anomaly"] = viz_data["true_anomaly"]
    X_test_stack_df["anomaly_score"] = viz_data["anomaly_score"]
    X_test_stack_df["predicted_anomaly"] = viz_data["predicted_anomaly"]



    print("\n--- Stacked Model Input DataFrame & (First 5 rows) ---")
    X_test_stack_df.info()
    display(X_test_stack_df.head())

    # Create the final output DataFrame
    output_df = X_test_stack_df.copy() # Start with the stacked input
    output_df['True_Threat_Level'] = y_test.reset_index(drop=True) # Add true labels
    output_df['Predicted_Threat_Level'] = y_pred # Add predicted labels


    output_df.rename(columns={'True_Threat_Level': 'Actual Threat', 'Predicted_Threat_Level': 'Pred Threat'}, inplace=True)
    output_df = output_df.reset_index(drop=True)
    output_df.index.name = 'Sample ID'
    output_df.index += 1
    print("\n--- Stacked Model Output DataFrame & (First 5 rows) ---")
    output_df.info()
    display(output_df.head())

    log("Finished displaying Stacked Model Input and Output DataFrames.")
    return output_df
#--------------------------
#get model label predict
#-------------------------
def get_stacked_model_predictions(gb_model, X_test_stack):
    """
    Generates and returns the predicted labels from the trained Gradient Boosting meta-learner.

    Args:
        gb_model: The trained Gradient Boosting Classifier model.
        X_test_stack (np.ndarray): Stacked test features (original + anomaly + RF proba).

    Returns:
        np.ndarray: Predicted labels from the stacked model.
    """
    log("Generating predictions from the stacked model...")
    y_pred = gb_model.predict(X_test_stack)
    log("Finished generating predictions.")
    return y_pred
#-------------------
#  Sub Functions
#------------------

def step_load_and_split(augmented_data = None):
    """Step 1: Load and split dataset"""
    return load_and_split_data(DATA_PATH, LABEL_COL, TEST_SIZE, RANDOM_STATE, augmented_data)


def step_scale_data(X_train, X_test):
    """Step 2: Scale data"""
    return scale_data(X_train, X_test)


def step_extract_anomaly_features(X_train_scaled, X_test_scaled, y_train, X_columns):
    """Step 3: Extract anomaly features using unsupervised models"""
    unsupervised_params = {
        'contamination': CONTAMINATION,
        'dbscan_eps': DBSCAN_EPS,
        'dbscan_min_samples': DBSCAN_MIN_SAMPLES,
        'kmeans_clusters': KMEANS_CLUSTERS,
        'autoencoder_epochs': AUTOENCODER_EPOCHS,
        'autoencoder_batch': AUTOENCODER_BATCH,
        'lstm_epochs': LSTM_EPOCHS,
        'lstm_batch': LSTM_BATCH
    }

    anomaly_train_df, anomaly_test_df, unsupervised_models = extract_unsupervised_features(
        X_train_scaled, X_test_scaled, y_train, unsupervised_params, RANDOM_STATE
    )

    X_train_ext = pd.concat([pd.DataFrame(X_train_scaled, columns=X_columns), anomaly_train_df], axis=1)
    X_test_ext = pd.concat([pd.DataFrame(X_test_scaled, columns=X_columns), anomaly_test_df], axis=1)

    return X_train_ext, X_test_ext, unsupervised_models


def step_train_stacked_model(X_train_ext, X_test_ext, y_train):
    """Step 4: Train stacked supervised model"""
    return train_stacked_model(X_train_ext, X_test_ext, y_train, RANDOM_STATE)


def step_evaluate_stacked_model(gb, X_test_stack, y_test):
    """Step 5: Evaluate stacked supervised model"""
    y_pred = gb.predict(X_test_stack)
    evaluate_stacked_model(type(gb).__name__, y_test, y_pred, MODEL_OUTPUT_DIR)
    return os.path.join(MODEL_OUTPUT_DIR, "metrics.json")


def step_evaluate_unsupervised(X_test_scaled, y_test, unsupervised_models, X_columns):
    """Step 6: Evaluate and visualize unsupervised models"""
    viz_data = evaluate_and_visualize_unsupervised_models(X_test_scaled, y_test, unsupervised_models, X_columns)
    return viz_data


def step_save_all(scaler, rf, gb, unsupervised_models, metrics_path):
    """Step 7: Save scaler, models, and metrics"""
    save_all_models_and_metrics(MODEL_OUTPUT_DIR, scaler, rf, gb, unsupervised_models, metrics_path)


#------------------
#  Main Pipeline
#------------------

def run_stacked_model_pipeline_integrated(augmented_data = None):
    """
    Orchestrates the entire data science process for the stacked model,
    integrating the steps previously in main_model_dev_stacked.
    """
    log("Starting integrated stacked model pipeline...")

    # 1. Load and split
    X_train, X_test, y_train, y_test, X_columns = step_load_and_split(augmented_data)

    # 2. Scale data
    scaler, X_train_scaled, X_test_scaled = step_scale_data(X_train, X_test)

    # 3. Extract anomaly features
    X_train_ext, X_test_ext, unsupervised_models = step_extract_anomaly_features(
        X_train_scaled, X_test_scaled, y_train, X_columns
    )

    # 4. Train stacked supervised model
    rf, gb_model, X_test_stack = step_train_stacked_model(X_train_ext, X_test_ext, y_train)

    # 5. Evaluate stacked model
    metrics_path = step_evaluate_stacked_model(gb_model, X_test_stack, y_test)

    # 6. Evaluate unsupervised models
    viz_data = step_evaluate_unsupervised(X_test_scaled, y_test, unsupervised_models, X_columns)

    # 7. Save everything
    step_save_all(scaler, rf, gb_model, unsupervised_models, metrics_path)

    # 8. get stacked model predictions
    y_pred = get_stacked_model_predictions(gb_model, X_test_stack)

    #9. Display model input and output
    output_df = display_model_input_output(X_test_ext, X_test_stack, y_test, y_pred, X_columns, viz_data)

    #10 save output_df
    output_df.to_csv(os.path.join(MODEL_OUTPUT_DIR, "stacked_anomaly_prediction_clissifier.csv"), index=False)
    log("Stacked model output data saved to :" + os.path.join(MODEL_OUTPUT_DIR, "stacked_anomaly_prediction_clissifier.csv"))


    log("Stacked model pipeline execution finished.")


# To run the integrated pipeline
if __name__ == "__main__":
    run_stacked_model_pipeline_integrated()

```

</details

GradientBoostingClassifier classification_report:  


<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_cm_class_report.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>  


<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_bell_curve.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>  


Overall Model Accuracy :  0.951188986232791

--- Performance Insight for GradientBoostingClassifier ---  

Metrics per Class:  

  Class: Low  
    Precision: 0.9644 - Of all instances predicted as 'Low', this is the proportion that were actually 'Low'.  
    Recall: 0.9788 - Of all actual 'Low' instances, this is the proportion that the model correctly identified.  
    F1-Score: 0.9715 - A balanced measure of Precision and Recall for this class.  
    Support: 471.0 - The number of actual instances of this class in the test set.  
    PPV (Positive Predictive Value): 0.9644 - The probability that a positive prediction is a true positive.  
    NPV (Negative Predictive Value): 0.9688 - The probability that a negative prediction is a true negative.  
    Sensitivity (True Positive Rate): 0.9788 - Ability to correctly identify positive instances.  
    Specificity (True Negative Rate): 0.6506 - Ability to correctly identify negative instances.  

        

  Class: Medium  
    Precision: 1.0000 - Of all instances predicted as 'Medium', this is the proportion that were actually 'Medium'.  
    Recall: 0.7667 - Of all actual 'Medium' instances, this is the proportion that the model correctly identified.  
    F1-Score: 0.8679 - A balanced measure of Precision and Recall for this class.  
    Support: 30.0 - The number of actual instances of this class in the test set.  
    PPV (Positive Predictive Value): 1.0000 - The probability that a positive prediction is a true positive.  
    NPV (Negative Predictive Value): 0.9910 - The probability that a negative prediction is a true negative.  
    Sensitivity (True Positive Rate): 0.7667 - Ability to correctly identify positive instances.  
    Specificity (True Negative Rate): 33.4348 - Ability to correctly identify negative instances.  

        

  Class: High  
    Precision: 0.9261 - Of all instances predicted as 'High', this is the proportion that were actually 'High'.  
    Recall: 0.9705 - Of all actual 'High' instances, this is the proportion that the model correctly identified.  
    F1-Score: 0.9477 - A balanced measure of Precision and Recall for this class.  
    Support: 271.0 - The number of actual instances of this class in the test set.  
    PPV (Positive Predictive Value): 0.9261 - The probability that a positive prediction is a true positive.  
    NPV (Negative Predictive Value): 0.9845 - The probability that a negative prediction is a true negative.  
    Sensitivity (True Positive Rate): 0.9705 - Ability to correctly identify positive instances.  
    Specificity (True Negative Rate): 1.7852 - Ability to correctly identify negative instances.  

  Class: Critical  
    Precision: 0.9286 - Of all instances predicted as 'Critical', this is the proportion that were actually 'Critical'.  
    Recall: 0.4815 - Of all actual 'Critical' instances, this is the proportion that the model correctly identified.  
    F1-Score: 0.6341 - A balanced measure of Precision and Recall for this class.  
    Support: 27.0 - The number of actual instances of this class in the test set.  
    PPV (Positive Predictive Value): 0.9286 - The probability that a positive prediction is a true positive.  
    NPV (Negative Predictive Value): 0.9822 - The probability that a negative prediction is a true negative.  
    Sensitivity (True Positive Rate): 0.4815 - Ability to correctly identify positive instances.  
    Specificity (True Negative Rate): 55.0714 - Ability to correctly identify negative instances.  
  
Aggregated Metrics:  
  Accuracy: 0.9512 - The overall proportion of correctly classified instances.  
  Precision (Macro): 0.9548 - Average Precision across all classes, unweighted.  
  Recall (Macro): 0.7993 - Average Recall across all classes, unweighted.  
  F1 Score (Macro): 0.8553 - Average F1-Score across all classes, unweighted.  
  Precision (Weighted): 0.9515 - Average Precision across all classes, weighted by support.  
  Recall (Weighted): 0.9512 - Average Recall across all classes, weighted by support.  
  F1 Score (Weighted): 0.9482 - Average F1-Score across all classes, weighted by support.  

Business Insights:  
- The model 'GradientBoostingClassifier' achieved an overall accuracy of 0.9512.  
- Pay close attention to the model's performance on classes: Critical. Lower F1-scores here might indicate challenges in correctly identifying these specific threat levels.  
- For 'Critical' threats (Recall: 0.4815, Precision: 0.9286), the model is able to capture a significant portion of actual critical events (Recall), and when it flags an event as critical, it is often correct (Precision). However, further investigation into missed critical events might be warranted to minimize risk.  
- Improving recall for Critical threats should be a priority to ensure all high-severity events are detected.  
- Consider deploying this model for automated threat detection, but maintain human oversight, especially for high-severity alerts.  
- Continuously monitor model performance in a production environment as threat patterns can evolve.  
- Investigate misclassified instances to understand the limitations and potential areas for model improvement or data enhancement.  
[INFO] Saved evaluation metrics to CyberThreat_Insight/stacked_models_deployment/metrics.json  

---Unsupervised Model Evaluation and Identification of the Best Model ---  

[INFO] Evaluating unsupervised models and identifying the best...  
[INFO] Visualizing performance for all unsupervised models...  
[INFO]   Generating visualization data for IsolationForest...  
[INFO]   IsolationForest performance (Average Precision): 0.2366  
[INFO]   Visualizing performance for IsolationForest...    


<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_if_scater_plot_roc_recall.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>   


--- Visualization Insights for IsolationForest ---

Scatter Plot Insight:  
- The scatter plot visually separates points identified as anomalies (298 points, typically shown in red) from normal points (501 points, typically shown in blue).  
- A clear separation between the red and blue points indicates that the model is effectively distinguishing between normal and anomalous behavior based on the chosen features.  

ROC Curve Insight:  
- The ROC curve shows the trade-off between the True Positive Rate (ability to detect actual anomalies) and the False Positive Rate (incorrectly flagging normal points as anomalies) at various threshold settings.  
- An Area Under the ROC Curve (AUC) of 0.1667 indicates the overall ability of the model to discriminate between positive (anomaly) and negative (normal) classes.  
- An AUC value below 0.5 suggests the model performs worse than random guessing.

  

Precision-Recall Curve Insight:  
- The Precision-Recall curve highlights the trade-off between Precision (accuracy of positive predictions) and Recall (completeness of positive predictions) as the decision threshold is varied.  
- This curve is particularly informative for imbalanced datasets, where the number of anomalies is much smaller than normal instances.  
- A curve that stays high as recall increases indicates that the model can achieve high precision even when identifying a large proportion of actual anomalies.  

Business Insights from Visualization:  
- The visualization for the IsolationForest suggests that the model's ability to discriminate anomalies is limited. The scatter plot may show significant overlap, and the AUC is low.  
- This model might not be suitable for direct deployment without significant improvements or a different approach.  
- The shape of the Precision-Recall curve provides insight into how many anomalies the model can find before its positive predictions become unreliable. A curve that drops sharply indicates that increasing recall comes at a high cost of precision (more false alarms).  



End Visualization Insights for IsolationForest.
[INFO]   Generating visualization data for OneClassSVM...
[INFO]   OneClassSVM performance (Average Precision): 0.2713
[INFO]   Visualizing performance for OneClassSVM...  

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_oneclossSVM_scater_plot_roc_recall.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>   

--- Visualization Insights for OneClassSVM ---   

Scatter Plot Insight:  
- The scatter plot visually separates points identified as anomalies (298 points, typically shown in red) from normal points (501 points, typically shown in blue).  
- A clear separation between the red and blue points indicates that the model is effectively distinguishing between normal and anomalous behavior based on the chosen features.  

ROC Curve Insight:  
- The ROC curve shows the trade-off between the True Positive Rate (ability to detect actual anomalies) and the False Positive Rate (incorrectly flagging normal points as anomalies) at various threshold settings.  
- An Area Under the ROC Curve (AUC) of 0.3158 indicates the overall ability of the model to discriminate between positive (anomaly) and negative (normal) classes.  
- An AUC value below 0.5 suggests the model performs worse than random guessing.  

Precision-Recall Curve Insight:  
- The Precision-Recall curve highlights the trade-off between Precision (accuracy of positive predictions) and Recall (completeness of positive predictions) as the decision threshold is varied.  
- This curve is particularly informative for imbalanced datasets, where the number of anomalies is much smaller than normal instances.  
- A curve that stays high as recall increases indicates that the model can achieve high precision even when identifying a large proportion of actual anomalies.  

Business Insights from Visualization:  
- The visualization for the OneClassSVM suggests that the model's ability to discriminate anomalies is limited. The scatter plot may show significant overlap, and the AUC is low.  
- This model might not be suitable for direct deployment without significant improvements or a different approach.  
- The shape of the Precision-Recall curve provides insight into how many anomalies the model can find before its positive predictions become unreliable. A curve that drops sharply indicates that increasing recall comes at a high cost of precision (more false alarms).  

End Visualization Insights for OneClassSVM.  
[INFO]   Generating visualization data for LocalOutlierFactor...  
[INFO]   LocalOutlierFactor performance (Average Precision): 0.3211  
[INFO]   Visualizing performance for LocalOutlierFactor...    

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_local_outlier_scater_plot_roc_recall.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>  

--- Visualization Insights for LocalOutlierFactor ---  
  
Scatter Plot Insight:  
- The scatter plot visually separates points identified as anomalies (298 points, typically shown in red) from normal points (501 points, typically shown in blue).  
- A clear separation between the red and blue points indicates that the model is effectively distinguishing between normal and anomalous behavior based on the chosen features.  

ROC Curve Insight:  
- The ROC curve shows the trade-off between the True Positive Rate (ability to detect actual anomalies) and the False Positive Rate (incorrectly flagging normal points as anomalies) at various threshold settings.  
- An Area Under the ROC Curve (AUC) of 0.3883 indicates the overall ability of the model to discriminate between positive (anomaly) and negative (normal) classes.  
- An AUC value below 0.5 suggests the model performs worse than random guessing.  

Precision-Recall Curve Insight:  
- The Precision-Recall curve highlights the trade-off between Precision (accuracy of positive predictions) and Recall (completeness of positive predictions) as the decision threshold is varied.  
- This curve is particularly informative for imbalanced datasets, where the number of anomalies is much smaller than normal instances.
- A curve that stays high as recall increases indicates that the model can achieve high precision even when identifying a large proportion of actual anomalies.  

Business Insights from Visualization:  
- The visualization for the LocalOutlierFactor suggests that the model's ability to discriminate anomalies is limited. The scatter plot may show significant overlap, and the AUC is low.  
- This model might not be suitable for direct deployment without significant improvements or a different approach.  
- The shape of the Precision-Recall curve provides insight into how many anomalies the model can find before its positive predictions become unreliable. A curve that drops sharply indicates that increasing recall comes at a high cost of precision (more false alarms).  

End Visualization Insights for LocalOutlierFactor.  
[INFO]   Generating visualization data for DBSCAN...  
[INFO] Skipping visualization for DBSCAN: Anomaly score not available.  
[INFO]   Generating visualization data for KMeans...  
[INFO]   KMeans performance (Average Precision): 0.2511  
[INFO]   Visualizing performance for KMeans...    


<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_Kmeans_roc_recall.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>

--- Visualization Insights for KMeans ---

Scatter Plot Insight:  
- The scatter plot visually separates points identified as anomalies (298 points, typically shown in red) from normal points (501 points, typically shown in blue).  
- A clear separation between the red and blue points indicates that the model is effectively distinguishing between normal and anomalous behavior based on the chosen features.  

ROC Curve Insight:  
- The ROC curve shows the trade-off between the True Positive Rate (ability to detect actual anomalies) and the False Positive Rate (incorrectly flagging normal points as anomalies) at various threshold settings.  
- An Area Under the ROC Curve (AUC) of 0.2286 indicates the overall ability of the model to discriminate between positive (anomaly) and negative (normal) classes.   
- An AUC value below 0.5 suggests the model performs worse than random guessing.  

Precision-Recall Curve Insight:  
- The Precision-Recall curve highlights the trade-off between Precision (accuracy of positive predictions) and Recall (completeness of positive predictions) as the decision threshold is varied.  
- This curve is particularly informative for imbalanced datasets, where the number of anomalies is much smaller than normal instances.  
- A curve that stays high as recall increases indicates that the model can achieve high precision even when identifying a large proportion of actual anomalies.  

Business Insights from Visualization:  
- The visualization for the KMeans suggests that the model's ability to discriminate anomalies is limited. The scatter plot may show significant overlap, and the AUC is low.  
- This model might not be suitable for direct deployment without significant improvements or a different approach.  
- The shape of the Precision-Recall curve provides insight into how many anomalies the model can find before its positive predictions become unreliable. A curve that drops sharply indicates that increasing recall comes at a high cost of precision (more false alarms).  

End Visualization Insights for KMeans.  
[INFO]   Generating visualization data for Autoencoder...  
[INFO]   Autoencoder performance (Average Precision): 0.5623   
[INFO]   Visualizing performance for Autoencoder...    



<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_best_unsup_model_aut.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>  

--- Visualization Insights for Autoencoder ---  

Scatter Plot Insight:  
- The scatter plot visually separates points identified as anomalies (298 points, typically shown in red) from normal points (501 points, typically shown in blue).  
- A clear separation between the red and blue points indicates that the model is effectively distinguishing between normal and anomalous behavior based on the chosen features.  

ROC Curve Insight:  
- The ROC curve shows the trade-off between the True Positive Rate (ability to detect actual anomalies) and the False Positive Rate (incorrectly flagging normal points as anomalies) at various threshold settings.   
- An Area Under the ROC Curve (AUC) of 0.6977 indicates the overall ability of the model to discriminate between positive (anomaly) and negative (normal) classes.  
- An AUC value above 0.5 suggests the model performs better than random guessing.  

Precision-Recall Curve Insight:   
- The Precision-Recall curve highlights the trade-off between Precision (accuracy of positive predictions) and Recall (completeness of positive predictions) as the decision threshold is varied.  
- This curve is particularly informative for imbalanced datasets, where the number of anomalies is much smaller than normal instances.
- A curve that stays high as recall increases indicates that the model can achieve high precision even when identifying a large proportion of actual anomalies.  

Business Insights from Visualization:  
- The visualization for the Autoencoder indicates that the model is somewhat effective at identifying anomalies, performing better than random guessing. However, there might be room for improvement in separating normal and anomalous instances.  
- Further refinement of features or model parameters could enhance its utility in practice.  
- The shape of the Precision-Recall curve provides insight into how many anomalies the model can find before its positive predictions become unreliable. A curve that drops sharply indicates that increasing recall comes at a high cost of precision (more false alarms).   

End Visualization Insights for Autoencoder.  
[INFO]   Generating visualization data for LSTM(Classifier)...  
[INFO]   LSTM(Classifier) performance (Average Precision): 0.3288  
[INFO]   Visualizing performance for LSTM(Classifier)...    



<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_LSTM_scater_plot_ROC_recall.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>   

--- Visualization Insights for LSTM(Classifier) ---   

Scatter Plot Insight:  
- The scatter plot visually separates points identified as anomalies (298 points, typically shown in red) from normal points (501 points, typically shown in blue).  
- A clear separation between the red and blue points indicates that the model is effectively distinguishing between normal and anomalous behavior based on the chosen features.  

ROC Curve Insight:  
- The ROC curve shows the trade-off between the True Positive Rate (ability to detect actual anomalies) and the False Positive Rate (incorrectly flagging normal points as anomalies) at various threshold settings.   
- An Area Under the ROC Curve (AUC) of 0.4633 indicates the overall ability of the model to discriminate between positive (anomaly) and negative (normal) classes.  
- An AUC value below 0.5 suggests the model performs worse than random guessing.  

Precision-Recall Curve Insight:  
- The Precision-Recall curve highlights the trade-off between Precision (accuracy of positive predictions) and Recall (completeness of positive predictions) as the decision threshold is varied.   
- This curve is particularly informative for imbalanced datasets, where the number of anomalies is much smaller than normal instances.  
- A curve that stays high as recall increases indicates that the model can achieve high precision even when identifying a large proportion of actual anomalies.  

Business Insights from Visualization:  
- The visualization for the LSTM(Classifier) suggests that the model's ability to discriminate anomalies is limited. The scatter plot may show significant overlap, and the AUC is low.   
- This model might not be suitable for direct deployment without significant improvements or a different approach.  
- The shape of the Precision-Recall curve provides insight into how many anomalies the model can find before its positive predictions become unreliable. A curve that drops sharply indicates that increasing recall comes at a high cost of precision (more false alarms).  

End Visualization Insights for LSTM(Classifier).   
[INFO] Finished visualizing performance for all unsupervised models.  
[INFO] Evaluating IsolationForest...  
[INFO]   IsolationForest performance (Average Precision): 0.2366  
[INFO] Evaluating OneClassSVM...   
[INFO]   OneClassSVM performance (Average Precision): 0.2713   
[INFO] Evaluating LocalOutlierFactor...  
[INFO]   LocalOutlierFactor performance (Average Precision): 0.3211  
[INFO] Evaluating DBSCAN...   
[INFO] Evaluating KMeans...  
[INFO]   KMeans performance (Average Precision): 0.2511   
[INFO] Evaluating Autoencoder...  
[INFO]   Autoencoder performance (Average Precision): 0.5623  
[INFO] Evaluating LSTM(Classifier)...   
[INFO]   LSTM(Classifier) performance (Average Precision): 0.3288  
[INFO]   
Best unsupervised model: Autoencoder with performance: 0.5623    

--- End of Unsupervised Model Visualizations and Insights ---  

Best unsupervised model for anomaly detection: Autoencoder   
[INFO] Visualizing performance for the best unsupervised model (Autoencoder)...    

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_autoencoder_roc_recall.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>   

--- Visualization Insights for Autoencoder ---  

Scatter Plot Insight:  
- The scatter plot visually separates points identified as anomalies (298 points, typically shown in red) from normal points (501 points, typically shown in blue).  
- A clear separation between the red and blue points indicates that the model is effectively distinguishing between normal and anomalous behavior based on the chosen features.  

ROC Curve Insight:  
- The ROC curve shows the trade-off between the True Positive Rate (ability to detect actual anomalies) and the False Positive Rate (incorrectly flagging normal points as anomalies) at various threshold settings.  
- An Area Under the ROC Curve (AUC) of 0.6977 indicates the overall ability of the model to discriminate between positive (anomaly) and negative (normal) classes.  
- An AUC value above 0.5 suggests the model performs better than random guessing.  

Precision-Recall Curve Insight:   
- The Precision-Recall curve highlights the trade-off between Precision (accuracy of positive predictions) and Recall (completeness of positive predictions) as the decision threshold is varied.  
- This curve is particularly informative for imbalanced datasets, where the number of anomalies is much smaller than normal instances.  
- A curve that stays high as recall increases indicates that the model can achieve high precision even when identifying a large proportion of actual anomalies.  

Business Insights from Visualization:  
- The visualization for the Autoencoder indicates that the model is somewhat effective at identifying anomalies, performing better than random guessing. However, there might be room for improvement in separating normal and anomalous instances.  
- Further refinement of features or model parameters could enhance its utility in practice.  
- The shape of the Precision-Recall curve provides insight into how many anomalies the model can find before its positive predictions become unreliable. A curve that drops sharply indicates that increasing recall comes at a high cost of precision (more false alarms).  

End Visualization Insights for Autoencoder.  
[INFO] Generating comparative performance analysis...  
[INFO]   Evaluating IsolationForest...  
[INFO]   IsolationForest performance (Average Precision): 0.2366  
[INFO]     IsolationForest performance (AUC): 0.2366  
[INFO]   Evaluating OneClassSVM...  
[INFO]   OneClassSVM performance (Average Precision): 0.2713  
[INFO]     OneClassSVM performance (AUC): 0.2713  
[INFO]   Evaluating LocalOutlierFactor...  
[INFO]   LocalOutlierFactor performance (Average Precision): 0.3211  
[INFO]     LocalOutlierFactor performance (AUC): 0.3211  
[INFO]   Evaluating DBSCAN...   
[INFO]     Could not evaluate DBSCAN performance.   
[INFO]   Evaluating KMeans...  
[INFO]   KMeans performance (Average Precision): 0.2511  
[INFO]     KMeans performance (AUC): 0.2511  
[INFO]   Evaluating Autoencoder...  
[INFO]   Autoencoder performance (Average Precision): 0.5623  
[INFO]     Autoencoder performance (AUC): 0.5623   
[INFO]   Evaluating LSTM(Classifier)...  
[INFO]   LSTM(Classifier) performance (Average Precision): 0.3288  
[INFO]     LSTM(Classifier) performance (AUC): 0.3288  
[INFO]   Displaying unsupervised model performance table...    
INFO]   Displaying unsupervised model performance chart...  

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_unsup_model_perf.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>  

   
--- Insights from Unsupervised Model Performance Comparison ---  
Best Performing Model: Autoencoder (Metric: 0.5623)  
Worst Performing Model: IsolationForest (Metric: 0.2366)  

Performance Comparison by Model Type:  
- Autoencoder Models (Mean Metric): 0.4456  
- Clustering Models (Mean Metric): 0.2511  
- Tree-based Models (Mean Metric): 0.2789   
- One-Class SVM (Metric): 0.2713  

Potential Reasons for Performance Differences:  
- Autoencoders (especially LSTM) might be better at capturing complex, non-linear relationships and temporal patterns in the data, which is relevant for anomaly detection in time-series-like threat data.  
- Clustering methods like DBSCAN and KMeans might struggle if anomalies don't form distinct clusters or if the concept of 'normal' varies significantly.  
- Tree-based methods (Isolation Forest, LOF) are often effective but might be sensitive to feature scaling and hyperparameters.  
- One-Class SVM is designed for novelty detection but its performance can be sensitive to the choice of kernel and hyperparameters.  
- The specific nature of the anomalies in this dataset (e.g., their density, shape, and relationship to normal data) heavily influences which model type is most suitable.  

Comparison to Baseline:  
- The best model (Autoencoder) achieved a metric of 0.5623.  
- Compared to a random guessing baseline (AUC = 0.5), the best AUC model performs better.  

Implications for the Stacked Model:  
- The anomaly features generated by the unsupervised models, particularly the best performing ones like Autoencoder, will be fed into the supervised meta-learner.  
- Higher quality anomaly features (from better performing unsupervised models) are likely to improve the performance of the stacked supervised model, especially in distinguishing between different threat levels.   
- The meta-learner can learn to leverage the strengths of different anomaly detection signals from the various unsupervised models.  
- While the individual unsupervised models might not be perfect anomaly detectors on their own, their outputs serve as valuable additional features for the supervised layer.  

--- End of Insights ---  
[INFO] Saving models and metrics...  
[INFO]   Scaler and ALL models saved in 'CyberThreat_Insight/stacked_models_deployment'  
[INFO] Evaluation metrics saved in CyberThreat_Insight/stacked_models_deployment/metrics.json  
[INFO] Generating predictions from the stacked model...  
[INFO] Finished generating predictions.  
[INFO] Displaying Stacked Model Input and Output DataFrames...  

--- Stacked Model Input DataFrame & (First 5 rows) ---  
<class 'pandas.core.frame.DataFrame'>  
RangeIndex: 799 entries, 0 to 798   
Data columns (total 24 columns):  
###   Column                      Non-Null Count  Dtype    
---  ------                      --------------  -----  
 0   Issue Response Time Days    799 non-null    float64  
 1   Impact Score                799 non-null    float64  
 2   Cost                        799 non-null    float64  
 3   Session Duration in Second  799 non-null    float64    
 4   Num Files Accessed          799 non-null    float64  
 5   Login Attempts              799 non-null    float64  
 6   Data Transfer MB            799 non-null    float64  
 7   CPU Usage %                 799 non-null    float64  
 8   Memory Usage MB             799 non-null    float64  
 9   Threat Score                799 non-null    float64  
 10  anomaly_0                   799 non-null    float64  
 11  anomaly_1                   799 non-null    float64  
 12  anomaly_2                   799 non-null    float64  
 13  anomaly_3                   799 non-null    float64  
 14  anomaly_4                   799 non-null    float64  
 15  anomaly_5                   799 non-null    float64  
 16  anomaly_6                   799 non-null    float64  
 17  rf_proba_0                  799 non-null    float64  
 18  rf_proba_1                  799 non-null    float64  
 19  rf_proba_2                  799 non-null    float64  
 20  rf_proba_3                  799 non-null    float64  
 21  true_anomaly                799 non-null    int64    
 22  anomaly_score               799 non-null    float64  
 23  predicted_anomaly           799 non-null    int64    
dtypes: float64(22), int64(2)  

-- Stacked Model Output DataFrame & (First 5 rows) ---    
<class 'pandas.core.frame.DataFrame'>    
RangeIndex: 799 entries, 1 to 799    
Data columns (total 26 columns):    
###   Column                      Non-Null Count  Dtype    
---  ------                      --------------  -----     
 0   Issue Response Time Days    799 non-null    float64  
 1   Impact Score                799 non-null    float64  
 2   Cost                        799 non-null    float64  
 3   Session Duration in Second  799 non-null    float64  
 4   Num Files Accessed          799 non-null    float64  
 5   Login Attempts              799 non-null    float64  
 6   Data Transfer MB            799 non-null    float64  
 7   CPU Usage %                 799 non-null    float64  
 8   Memory Usage MB             799 non-null    float64  
 9   Threat Score                799 non-null    float64    
 10  anomaly_0                   799 non-null    float64  
 11  anomaly_1                   799 non-null    float64  
 12  anomaly_2                   799 non-null    float64  
 13  anomaly_3                   799 non-null    float64  
 14  anomaly_4                   799 non-null    float64  
 15  anomaly_5                   799 non-null    float64  
 16  anomaly_6                   799 non-null    float64  
 17  rf_proba_0                  799 non-null    float64  
 18  rf_proba_1                  799 non-null    float64  
 19  rf_proba_2                  799 non-null    float64  
 20  rf_proba_3                  799 non-null    float64  
 21  true_anomaly                799 non-null    int64    
 22  anomaly_score               799 non-null    float64  
 23  predicted_anomaly           799 non-null    int64    
 24  Actual Threat               799 non-null    int64    
 25  Pred Threat                 799 non-null    int64    
dtypes: float64(22), int64(4)  
