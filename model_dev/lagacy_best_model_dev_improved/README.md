
<h1 align="center">
Hybrid ML Approach for Cyber Threat Classification
</h1> 
<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_improved_headline.png" 
       alt="Centered Image" 
       style="width: 1000px; height: Auto;">
</p>  

<p align="center">
    Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI
</p> 


### Introduction

Cybersecurity threats continue to evolve at a rapid pace, rendering traditional signature-based detection methods increasingly inadequate. As a result, identifying anomalous behavior has become essential for detecting novel and emerging threats. <a href="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/model_dev/lagacy_best_model_dev/README.md" target="_blank">In the previous article</a> , supervised machine learning models were evaluated using standard classification metrics, including accuracy, precision, recall, F1-score, and confusion matrices. Among these models, Random Forest and Gradient Boosting demonstrated strong performance, accurately predicting all four threat classes.

In contrast, unsupervised models were initially evaluated by transforming anomaly scores into binary labels (normal versus anomalous). While effective at detecting deviations from normal behavior, these models were limited to binary predictions typically identifying only the lowest threat class—and failed to capture nuanced severity levels such as High and Critical.

This article presents a Cyber Threat Detection Engine that integrates both supervised and unsupervised machine learning techniques to address these limitations. By combining precise classification of known threats with robust detection of previously unseen anomalies, the proposed system aims to deliver a more comprehensive and resilient approach to cyber threat classification across all severity levels: Low, Medium, High, and Critical.

### Supervised vs Unsupervised Learning in this Project

This project utilizes both supervised and unsupervised learning approaches. Supervised models like Random Forest and Gradient Boosting are trained on labeled data to directly classify known threat levels. Unsupervised models such as Isolation Forest and KMeans are used for anomaly detection and clustering to identify potentially novel threats. The outputs of these unsupervised models are then mapped to the defined threat levels. This combined approach leverages labeled data for known threats while enabling the detection of new anomalies.


### Data Injection and Preprocessing:
<a 
  href="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/datagen/README.md"
  target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Feature Engineering:
<a 
  href="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/feature_engineering/README.md"
  target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Model Development and Evaluation  
## Overview

Purely unsupervised anomaly detection models are inherently limited:
they can identify *novel or rare behavior*, but they **cannot natively classify cyber threats by severity** (Low / Medium / High / Critical).

This project addresses that limitation through a **principled hybrid learning architecture** that:

> **Transforms unsupervised anomaly detectors into severity-aware feature generators by aligning their latent structure with known threat labels, enabling robust supervised multiclass threat classification.**

Rather than forcing unsupervised models to make semantic decisions they were never designed for, this architecture **leverages their strengths**—structure discovery and novelty detection—while delegating severity interpretation to supervised learning.

---

## Core Problem Addressed

Unsupervised anomaly detectors typically produce one of the following outputs:

* **Binary anomaly flags** (normal vs anomalous)
* **Continuous anomaly scores** (distance, isolation, reconstruction error)
* **Unlabeled clusters** (cluster IDs, density groups, noise points)

These outputs:

* Lack semantic meaning
* Do not encode threat hierarchy
* Cannot distinguish *severity*

### Design Principle

> **Unsupervised models extract structure and risk signals; supervised models assign meaning and severity.**

---

##  Architectural Pattern

### **Unsupervised → Feature Generation → Supervised Multiclass Learning**

Unsupervised models are **never used as final classifiers**.
Instead, they act as **intermediate signal generators**.

### Extracted Risk Signals

| Signal Type         | Description                   |
| ------------------- | ----------------------------- |
| Anomaly score       | Continuous risk measure       |
| Binary anomaly flag | Thresholded novelty indicator |
| Cluster membership  | Behavioral regime identifier  |
| Distance / density  | Deviation from learned norms  |

These signals are later **aligned with known threat labels** and fed into supervised models.

---

## Weak Supervision via Label Alignment (Critical Step)

The key innovation lies in **aligning unsupervised outputs with known threat labels using training data only**.

### Concept

For each unsupervised structure (cluster, anomaly flag, density group):

1. Observe the **true labels** within that structure
2. Assign the **majority threat severity**
3. Store the mapping
4. Apply it consistently to unseen data

This introduces **semantic meaning without retraining unsupervised models**.

---

### 🔧 Key Function: Cluster / Signal → Severity Mapping

```python
def map_clusters_to_labels(cluster_ids, true_labels):
    """
    Maps unsupervised cluster or anomaly outputs to threat severity
    using majority voting on training data only.
    """
    mapping = {}
    unique_clusters = set(cluster_ids)

    for cluster in unique_clusters:
        indices = [i for i, c in enumerate(cluster_ids) if c == cluster]
        labels = [true_labels[i] for i in indices]

        if labels:
            majority_label = max(set(labels), key=labels.count)
            mapping[cluster] = majority_label

    return mapping
```

**Why this is safe and principled**

* Mapping is learned **only on training data**
* No label leakage into test data
* This is **weak supervision**, not supervised clustering

---

##  Model-Specific Feature Adaptation

*  Isolation Forest / LOF / One-Class SVM

**Raw outputs**

* `decision_function()` → anomaly score
* `predict()` → anomaly flag

**Derived features**

```python
anomaly_score = model.decision_function(X)
is_anomaly = model.predict(X)
severity_hint = severity_map[is_anomaly]
```

*Severity correlates with degree of isolation*

---

### 🔹 Autoencoder

**Raw output**

* Reconstruction error (MSE)

**Feature extraction**

```python
recon_error = np.mean((X - X_recon) ** 2, axis=1)
is_anomaly = recon_error > threshold
severity_hint = severity_map[is_anomaly]
```

*Severity correlates with deviation from learned normal behavior*

---

###  KMeans

**Raw outputs**

* Cluster ID
* Distance to centroid

```python
clusters = kmeans.predict(X)
distances = np.linalg.norm(X - kmeans.cluster_centers_[clusters], axis=1)

severity_hint = [cluster_map[c] for c in clusters]
```

 *Severity correlates with behavioral regime and deviation from centroid*

---

###  DBSCAN

**Raw outputs**

* Cluster ID
* Noise label (`-1`)

```python
clusters = dbscan.fit_predict(X)
severity_hint = [cluster_map.get(c, "High") for c in clusters]
```

 *Severity correlates with density isolation*

---

##  Severity-Aware Feature Construction

All unsupervised outputs are consolidated into a **feature matrix**:

| Feature Type          | Purpose                  |
| --------------------- | ------------------------ |
| Continuous scores     | Preserve risk gradients  |
| Binary flags          | Encode novelty           |
| Cluster IDs           | Capture behavior regimes |
| Mapped severity hints | Inject semantics         |

These features are **interpretable, explainable, and auditable**.

---

##  Supervised Multiclass Threat Classification

The final stage uses supervised models such as:

* Random Forest
* Gradient Boosting
* XGBoost
* LSTM (for temporal sequences)

They learn:

> How combinations of anomaly signals and behavioral patterns map to explicit threat severity levels.

```python
classifier.fit(X_features, y_severity)
```

 **Unsupervised models become feature extractors—not decision makers.**

---

## Architectural Summary

> **This system converts unsupervised anomaly detectors into severity-aware feature generators by aligning their latent structure with known threat labels, enabling robust supervised multiclass cyber threat classification.**

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/hybrid_cyber_threat_severity_classification_pipeline.png" 
       alt="Centered Image" 
       style="width: 80%; height: Auto;">
</p>

---

##  SOC-Grade Architecture Is   

✔ Detects **novel attacks**  
✔ Preserves **risk gradients**  
✔ Produces **explainable signals**  
✔ Aligns with **governance & model risk management**  
✔ Scales to **real-world SOC environments**  

This design mirrors how **human SOC analysts reason**:  
*first detect abnormal behavior, then contextualize and prioritize severity.*  

### Run the code:
<a 
  href="https://colab.research.google.com/github/atsuvovor/CyberThreat_Insight/blob/main/model_dev/lagacy_best_model_dev_improved/lagacy_model_dev_improved.ipynb"
  target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<details>

<summary>Click to view the code</summary>

```python


#CyberThreat-Insight
#Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI
#  ---------------------Model Development---------------------
#Toronto, Septeber 08 2025
#Autor : Atsu Vovor

#Master of Management in Artificial Intelligence
#Consultant Data Analytics Specialist | Machine Learning |
#Data science | Quantitative Analysis |French & English Bilingual

#-------------------
#  Import Library
#------------------
import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LSTM, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
#from google.colab import drive
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
import warnings

# Ignore warnings related to feature names from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Define RANDOM_STATE for reproducibility
RANDOM_STATE = 42


#------------------
# Color Mapping
#------------------
def get_color_map():
    """Define custom color map."""
    # Define the colors
    #colors = ["darkred", "red", "orangered", "orange", "yelloworange", "lightyellow", "yellow", "greenyellow", "green"]
    colors = ["#8B0000", "#FF0000", "#FF4500", "#FFA500", "#FFB347", "#FFFFE0", "#FFFF00", "#ADFF2F", "#008000"]

    # Create a colormap
    custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", colors)

    return custom_cmap
custom_cmap = get_color_map()

# -------------------------
# Utilities
# -------------------------

def ensure_numpy(x):
    """Ensure input is a numpy array."""
    return np.asarray(x) if not isinstance(x, np.ndarray) else x

def multiclass_metrics(y_true, y_pred):
    """Calculate multiclass classification metrics."""
    y_true = ensure_numpy(y_true)
    y_pred = ensure_numpy(y_pred)
    return {
        "Overall Model Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (Macro)": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall (Macro)": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1 Score (Macro)": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    }

def map_clusters_to_labels(train_clusters, train_labels):
    """
    Given cluster labels or binary outputs on the training set and the true training labels,
    return a dict mapping cluster_value -> majority Threat Level in that cluster.
    """
    mapping = {}
    df = pd.DataFrame({"cluster": train_clusters, "label": train_labels})
    for cluster_val, group in df.groupby("cluster"):
        most_common = group["label"].mode()
        mapping[cluster_val] = int(most_common.iloc[0]) if not most_common.empty else int(group["label"].iloc[0])
    return mapping

def apply_mapping(preds, mapping, default_label=0):
    """Map predicted cluster/binary values to multiclass labels using mapping dict."""
    mapped = [mapping.get(p, default_label) for p in preds]
    return np.array(mapped, dtype=int)

# print performance metrics charts
def print_model_metrics_charts(df):
    """Prints performance metrics charts for each model."""

    metrics_to_plot = ['Overall Model Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)']
    num_metrics = len(metrics_to_plot)

    # Create subplots: 2 rows, 2 columns for 4 metrics
    fig = make_subplots(rows=2, cols=2, subplot_titles=metrics_to_plot)

    # Define a list of colors for the bars in each subplot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Example colors

    for i, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            # Sort data for each metric
            sorted_df = df.sort_values(by=metric, ascending=False)

            # Add bar trace to the corresponding subplot
            row = (i // 2) + 1
            col = (i % 2) + 1

            fig.add_trace(go.Bar(
                x=sorted_df['model'],
                y=sorted_df[metric],
                name=metric, # Name for legend (optional in subplots)
                marker_color=colors[i] # Use a different color for each metric
            ), row=row, col=col)

            # Update layout for the subplot axes if needed
            fig.update_xaxes(title_text='Model', row=row, col=col)
            fig.update_yaxes(title_text=metric, row=row, col=col)

        else:
            print(f"Metric '{metric}' not found in metrics_df.")

    # Update overall layout
    fig.update_layout(height=700, width=900, title_text="Model Performance Metrics", showlegend=False)
    fig.show()

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

def get_label_class_name(model_y_test, y_model_pred):
    """Get class names based on labels."""
    labels = sorted(list(set(model_y_test) | set(y_model_pred)))
    level_mapping = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
    class_names = [level_mapping.get(label) for label in labels]

    return class_names, labels
#------------------------------
def plot_metrics_bell_curve(model_name, y_true, y_pred, class_names=None):
    """
    Plots PPV, Sensitivity, NPV, Specificity for multi-class classification
    on separate bell-curve style subplots for clarity.
    """

    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]

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
        specificity = TN[i] / (TN[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0

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
#-----------------------------

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
    plot_metrics_bell_curve(model_name, model_y_test, y_model_pred, class_names=class_names)
    print("\nOverall Model Accuracy : ", metrics_dic.get("Overall Model Accuracy ", 0))
    # Call the generate_performance_insight function for the best model
    generate_performance_insight(model_name, report, model_y_test, y_model_pred )

    return  metrics_dic
#----------------------------------------

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

#----------------------------------------------
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
# -------------------------
# Unsupervised model wrappers (produce cluster/binary preds)
# -------------------------
def iso_forest_train_and_map(X_train, y_train, X_test):
    """Train Isolation Forest and map predictions."""
    model = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    model.fit(X_train)
    raw_train = model.decision_function(X_train)   # decision function as score
    raw_test = model.decision_function(X_test)
    raw_train_bin = np.where(model.predict(X_train) == -1, 1, 0) # -1 anomaly, 1 normal
    raw_test_bin = np.where(model.predict(X_test) == -1, 1, 0)
    mapping = map_clusters_to_labels(raw_train_bin, y_train)
    mapped_test = apply_mapping(raw_test_bin, mapping, default_label=int(Counter(y_train).most_common(1)[0][0]))

    X_test_viz = X_test.copy()
    X_test_viz['anomaly_score'] = raw_test
    X_test_viz['is_anomaly'] = raw_test_bin

    return mapped_test, model, mapping, X_test_viz


def lof_train_and_map(X_train, y_train, X_test):
    """Train Local Outlier Factor and map predictions."""
    model = LocalOutlierFactor(n_neighbors=20, novelty=True)
    model.fit(X_train)
    raw_train = model.decision_function(X_train)   # decision function as score
    raw_test = model.decision_function(X_test)
    raw_train_bin = np.where(model.predict(X_train) == -1, 1, 0) # -1 anomaly, 1 normal
    raw_test_bin = np.where(model.predict(X_test) == -1, 1, 0)
    mapping = map_clusters_to_labels(raw_train_bin, y_train)
    mapped_test = apply_mapping(raw_test_bin, mapping, default_label=int(Counter(y_train).most_common(1)[0][0]))

    X_test_viz = X_test.copy()
    X_test_viz['anomaly_score'] = raw_test
    X_test_viz['is_anomaly'] = raw_test_bin

    return mapped_test, model, mapping, X_test_viz

def ocsvm_train_and_map(X_train, y_train, X_test):
    """Train One-Class SVM and map predictions."""
    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
    model.fit(X_train)
    raw_train = model.decision_function(X_train) # decision function as score
    raw_test = model.decision_function(X_test)
    raw_train_bin = np.where(model.predict(X_train) == -1, 1, 0) # -1 anomaly, 1 normal
    raw_test_bin = np.where(model.predict(X_test) == -1, 1, 0)
    mapping = map_clusters_to_labels(raw_train_bin, y_train)
    mapped_test = apply_mapping(raw_test_bin, mapping, default_label=int(Counter(y_train).most_common(1)[0][0]))

    X_test_viz = X_test.copy()
    X_test_viz['anomaly_score'] = raw_test
    X_test_viz['is_anomaly'] = raw_test_bin

    return mapped_test, model, mapping, X_test_viz

def dbscan_train_and_map(X_train, y_train, X_test):
    """Train DBSCAN and map predictions."""
    model = DBSCAN(eps=0.5, min_samples=5)
    train_clusters = model.fit_predict(X_train)
    # DBSCAN labels -1 for noise (outliers)
    test_clusters = model.fit_predict(X_test)  # using fit_predict to create model on test (DBSCAN isn't typically used with separate test fit)
    # *Note*: DBSCAN typically doesn't fit on train/test split; this is a pragmatic mapping approach
    mapping = map_clusters_to_labels(train_clusters, y_train)
    mapped_test = apply_mapping(test_clusters, mapping, default_label=int(Counter(y_train).most_common(1)[0][0]))

    X_test_viz = X_test.copy()
    # DBSCAN doesn't have a standard 'score', use -1 or distance to nearest core sample if needed.
    # For visualization, let's use the cluster label itself, or a binary flag for noise (-1).
    X_test_viz['anomaly_score'] = test_clusters # Use cluster label as score placeholder
    X_test_viz['is_anomaly'] = (test_clusters == -1).astype(int) # Binary flag for noise

    return mapped_test, model, mapping, X_test_viz

def kmeans_train_and_map(X_train, y_train, X_test, n_clusters=4):
    """Train KMeans and map predictions."""
    # choose n_clusters default 4 (matching Threat Levels) but user can override
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    model.fit(X_train)
    train_clusters = model.predict(X_train)
    test_clusters = model.predict(X_test)
    mapping = map_clusters_to_labels(train_clusters, y_train)
    mapped_test = apply_mapping(test_clusters, mapping, default_label=int(Counter(y_train).most_common(1)[0][0]))

    X_test_viz = X_test.copy()
    # KMeans score could be distance to the assigned centroid
    test_distances = np.linalg.norm(X_test - model.cluster_centers_[test_clusters], axis=1)
    # Define 'anomaly' based on distance threshold or mapping.
    # For visualization, let's use the distance as score and a simple threshold for binary flag.
    # Threshold could be based on training data distances or a fixed percentile.
    train_distances = np.linalg.norm(X_train - model.cluster_centers_[train_clusters], axis=1)
    distance_threshold = np.percentile(train_distances, 95) # Example threshold
    X_test_viz['anomaly_score'] = test_distances
    X_test_viz['is_anomaly'] = (test_distances > distance_threshold).astype(int)

    return mapped_test, model, mapping, X_test_viz

def autoencoder_train_and_map(X_train_np, y_train_np, X_test_np, X_test_columns, encoding_dim=None, epochs=30, batch_size=32):
    """Train Autoencoder and map predictions."""
    # Ensure inputs are numpy arrays
    X_train_np = ensure_numpy(X_train_np)
    y_train_np = ensure_numpy(y_train_np)
    X_test_np = ensure_numpy(X_test_np)

    n_features = X_train_np.shape[1]
    if encoding_dim is None:
        encoding_dim = max(4, n_features // 2)

    # simple dense autoencoder
    inp = Input(shape=(n_features,))
    x = Dense(64, activation='relu')(inp) # Adding hidden layers to Autoencoder
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    encoded = Dense(encoding_dim, activation="relu")(x) # Renamed to encoded
    x = Dense(16, activation='relu')(encoded) # Adding hidden layers to Decoder
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    decoded = Dense(n_features, activation="sigmoid")(x) # Output layer
    ae = Model(inp, decoded)
    ae.compile(optimizer=Adam(1e-3), loss="mse")
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    # Fit on numpy arrays
    ae.fit(X_train_np, X_train_np, validation_data=(X_test_np, X_test_np), epochs=epochs, batch_size=batch_size, callbacks=[early], verbose=0)

    recon_train = ae.predict(X_train_np, verbose=0)
    mse_train = np.mean(np.square(X_train_np - recon_train), axis=1)
    thresh = np.percentile(mse_train, 95)  # threshold from train distribution
    recon_test = ae.predict(X_test_np, verbose=0)
    mse_test = np.mean(np.square(X_test_np - recon_test), axis=1)
    raw_test_bin = np.where(mse_test > thresh, 1, 0)
    train_bin = np.where(mse_train > thresh, 1, 0)

    # Map using original y_train (assuming it's a pandas Series or can be converted)
    mapping = map_clusters_to_labels(train_bin, y_train_np)
    mapped_test = apply_mapping(raw_test_bin, mapping, default_label=int(Counter(y_train_np).most_common(1)[0][0]))

    # Create X_test_viz as a DataFrame using the provided column names
    X_test_viz = pd.DataFrame(X_test_np, columns=X_test_columns)
    X_test_viz['anomaly_score'] = mse_test
    X_test_viz['is_anomaly'] = raw_test_bin

    return mapped_test, ae, mapping, mse_test, X_test_viz


# -------------------------
# Supervised model wrappers (predict multiclass directly)
# -------------------------
def rf_train_and_predict(X_train, y_train, X_test):
    """Train Random Forest and predict multiclass labels."""
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model, X_test.copy() # Return copy of X_test for consistency

def gb_train_and_predict(X_train, y_train, X_test):
    """Train Gradient Boosting and predict multiclass labels."""
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model, X_test.copy() # Return copy of X_test for consistency

# -------------------------
# LSTM multiclass classifier
# -------------------------
def lstm_classifier_train_and_predict(X_train_np, y_train_np, X_test_np, X_test_columns, timesteps=1, epochs=30, batch_size=32):
    """
    Train LSTM classifier and predict multiclass labels.

    - timesteps: if >1, n_features must be divisible by timesteps and the arrays will be reshaped.
    - y_train must be integer class labels [0..3].
    """
    X_train_np = ensure_numpy(X_train_np)
    y_train_np = ensure_numpy(y_train_np)
    X_test_np = ensure_numpy(X_test_np)
    n_features = X_train_np.shape[1]
    if n_features % timesteps != 0:
        raise ValueError("n_features must be divisible by timesteps when timesteps>1")
    feat_per_step = n_features // timesteps

    X_train_seq = X_train_np.reshape((X_train_np.shape[0], timesteps, feat_per_step))
    X_test_seq = X_test_np.reshape((X_test_np.shape[0], timesteps, feat_per_step))

    n_classes = len(np.unique(y_train_np))
    y_train_cat = tf.keras.utils.to_categorical(y_train_np, num_classes=n_classes)

    inputs = Input(shape=(timesteps, feat_per_step))
    x = LSTM(64, activation='tanh')(inputs)
    x = Dropout(0.2)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_seq, y_train_cat, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[early], verbose=0)
    preds_proba = model.predict(X_test_seq, verbose=0)
    preds = np.argmax(preds_proba, axis=1)

    # Create X_test_viz as a DataFrame using the provided column names
    X_test_viz = pd.DataFrame(X_test_np, columns=X_test_columns)
    # Supervised models don't have inherent 'anomaly_score' or 'is_anomaly' in the same way
    # Add placeholder columns or decide on a different visualization strategy for these models
    X_test_viz['anomaly_score'] = preds_proba[:, 1] if n_classes > 1 else 0 # Example: probability of class 1
    X_test_viz['is_anomaly'] = (preds > 0).astype(int) # Example: predicted class > 0 is anomaly

    return preds, model, X_test_viz


# -------------------------------------------------------------
# orchestrator: trains all models and selects best by accuracy
# -------------------------------------------------------------
def model_development_pipeline(data_path=None, df=None, target_col="Threat Level", test_size=0.2, random_state=42,
                               deploy_folder=".", lstm_timesteps=1):
    """
    Pipeline to load data, train various models, evaluate them, select the best, and save metrics and the best model.

    Provide either data_path (CSV) or df (pandas DataFrame). target_col must exist and be labeled 0..3.
    Returns dict with model results and saves metrics CSV and best model to deploy_folder.
    """
    # Load data
    if df is None:
        if data_path is None:
            raise ValueError("Provide either df or data_path")
        augmented_df = pd.read_csv(data_path)
    else:
        augmented_df = df.copy()

    # Ensure target exists and numeric
    if target_col not in augmented_df.columns:
        raise ValueError(f"{target_col} missing from dataframe")
    # ensure integer labels 0..3
    augmented_df[target_col] = augmented_df[target_col].astype(int)

    X = augmented_df.drop(columns=[target_col])
    y = augmented_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Keep track of original column names for creating visualization DataFrames
    original_columns = X.columns

    results = {}
    metrics_rows = []

    # 1) Supervised classical
    preds_rf, rf_model, X_test_rf_viz = rf_train_and_predict(X_train, y_train, X_test)
    metrics_rf = multiclass_metrics(y_test, preds_rf)
    results["RandomForest"] = {"model": rf_model, "preds": preds_rf, "metrics": metrics_rf, "X_test_viz": X_test_rf_viz}
    metrics_rows.append({"model": "RandomForest", **metrics_rf})

    preds_gb, gb_model, X_test_gb_viz = gb_train_and_predict(X_train, y_train, X_test)
    metrics_gb = multiclass_metrics(y_test, preds_gb)
    results["GradientBoosting"] = {"model": gb_model, "preds": preds_gb, "metrics": metrics_gb, "X_test_viz": X_test_gb_viz}
    metrics_rows.append({"model": "GradientBoosting", **metrics_gb})

    # 2) Unsupervised mapping -> multiclass
    preds_iso, iso_model, iso_map, X_test_iso_viz = iso_forest_train_and_map(X_train, y_train, X_test)
    metrics_iso = multiclass_metrics(y_test, preds_iso)
    results["IsolationForest"] = {"model": iso_model, "preds": preds_iso, "metrics": metrics_iso, "mapping": iso_map, "X_test_viz": X_test_iso_viz}
    metrics_rows.append({"model": "IsolationForest", **metrics_iso})

    preds_ocsvm, ocsvm_model, ocsvm_map, X_test_ocsvm_viz = ocsvm_train_and_map(X_train, y_train, X_test)
    metrics_ocsvm = multiclass_metrics(y_test, preds_ocsvm)
    results["OneClassSVM"] = {"model": ocsvm_model, "preds": preds_ocsvm, "metrics": metrics_ocsvm, "mapping": ocsvm_map, "X_test_viz": X_test_ocsvm_viz}
    metrics_rows.append({"model": "OneClassSVM", **metrics_ocsvm})

    preds_lof, lof_model, lof_map, X_test_lof_viz = lof_train_and_map(X_train, y_train, X_test)
    metrics_lof = multiclass_metrics(y_test, preds_lof)
    results["LocalOutlierFactor"] = {"model": lof_model, "preds": preds_lof, "metrics": metrics_lof, "mapping": lof_map, "X_test_viz": X_test_lof_viz}
    metrics_rows.append({"model": "LocalOutlierFactor", **metrics_lof})

    # DBSCAN (note: DBSCAN doesn't naturally support separate test set; we attempt an approach for mapping)
    try:
        preds_dbscan, dbscan_model, dbscan_map, X_test_dbscan_viz = dbscan_train_and_map(X_train, y_train, X_test)
        metrics_dbscan = multiclass_metrics(y_test, preds_dbscan)
    except Exception as e:
        preds_dbscan = np.full(len(y_test), int(Counter(y_train).most_common(1)[0][0]))
        dbscan_model, dbscan_map, X_test_dbscan_viz = None, {}, X_test.copy()
        metrics_dbscan = multiclass_metrics(y_test, preds_dbscan)
    results["DBSCAN"] = {"model": dbscan_model, "preds": preds_dbscan, "metrics": metrics_dbscan, "mapping": dbscan_map, "X_test_viz": X_test_dbscan_viz}
    metrics_rows.append({"model": "DBSCAN", **metrics_dbscan})

    # KMeans
    try:
        preds_kmeans, kmeans_model, kmeans_map, X_test_kmeans_viz = kmeans_train_and_map(X_train, y_train, X_test, n_clusters=4)
        metrics_kmeans = multiclass_metrics(y_test, preds_kmeans)
    except Exception as e:
        preds_kmeans = np.full(len(y_test), int(Counter(y_train).most_common(1)[0][0]))
        kmeans_model, kmeans_map, X_test_kmeans_viz = None, {}, X_test.copy()
        metrics_kmeans = multiclass_metrics(y_test, preds_kmeans)
    results["KMeans"] = {"model": kmeans_model, "preds": preds_kmeans, "metrics": metrics_kmeans, "mapping": kmeans_map, "X_test_viz": X_test_kmeans_viz}
    metrics_rows.append({"model": "KMeans", **metrics_kmeans})

    # Autoencoder (dense)
    # Pass X_test.values and X.columns to autoencoder_train_and_map
    preds_ae, ae_model, ae_map, ae_scores, X_test_ae_viz = autoencoder_train_and_map(X_train.values, y_train.values, X_test.values, original_columns, epochs=30)
    metrics_ae = multiclass_metrics(y_test, preds_ae)
    results["Autoencoder"] = {"model": ae_model, "preds": preds_ae, "metrics": metrics_ae, "mapping": ae_map, "scores": ae_scores, "X_test_viz": X_test_ae_viz}
    metrics_rows.append({"model": "Autoencoder", **metrics_ae})

    # LSTM classifier (multiclass supervised)
    # Pass X_test.values and X.columns to lstm_classifier_train_and_predict
    try:
        preds_lstm, lstm_model, X_test_lstm_viz = lstm_classifier_train_and_predict(X_train.values, y_train.values, X_test.values, original_columns, timesteps=lstm_timesteps, epochs=30)
        metrics_lstm = multiclass_metrics(y_test, preds_lstm)
    except Exception as e:
        preds_lstm = np.full(len(y_test), int(Counter(y_train).most_common(1)[0][0]))
        lstm_model, X_test_lstm_viz = None, X_test.copy()
        metrics_lstm = multiclass_metrics(y_test, preds_lstm)
    results["LSTM(Classifier)"] = {"model": lstm_model, "preds": preds_lstm, "metrics": metrics_lstm, "X_test_viz": X_test_lstm_viz}
    metrics_rows.append({"model": "LSTM(Classifier)", **metrics_lstm})


    # -------------------------
    # Save metrics summary CSV
    # -------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = os.path.join(deploy_folder, "model_metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)


    # -------------------------
    # Select best model by Overall Model Accuracy
    # -------------------------
    best_row = metrics_df.sort_values(by="Overall Model Accuracy", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_metrics = results[best_model_name]["metrics"]
    best_model_obj = results[best_model_name]["model"]
    print(f"Overall Best model selected by Overall Model Accuracy: {best_model_name} -> Accuracy: {best_metrics['Overall Model Accuracy']:.4f}")


    # -------------------------
    # Deploy best model
    # -------------------------
    os.makedirs(deploy_folder, exist_ok=True)
    model_path = None
    try:
        if best_model_obj is None:
            print("Best model object is None; nothing to save.")
            model_path = None
        else:
            if hasattr(best_model_obj, "predict") and not isinstance(best_model_obj, tf.keras.Model):
                # sklearn-like model -> joblib
                model_path = os.path.join(deploy_folder, f"{best_model_name}_best_model.joblib")
                joblib.dump(best_model_obj, model_path)
            else:
                # assume TF model
                model_path = os.path.join(deploy_folder, f"{best_model_name}_best_model_tf")
                best_model_obj.save(model_path, overwrite=True, include_optimizer=False)
        print(f"Saved best model to: {model_path}")
    except Exception as e:
        print("Failed to save best model:", e)
        model_path = None


    pipeline_result = {
        "results": results,
        "metrics_df": metrics_df,
        "best_model_name": best_model_name,
        "best_model_obj": best_model_obj,
        "best_model_path": model_path,
        "y_test": y_test,
        "X_test": X_test, # Include original X_test
        "X_train": X_train, # Include original X_train
        "y_train": y_train # Include original y_train
    }

    return pipeline_result


def main_model_delopement_function():
    """Main function to run the model development pipeline."""

    #try:
    #    drive.mount('/content/drive')
    #    print("Google Drive mounted successfully!")
    #except ValueError as e:
    #    print(f"Error mounting Google Drive: {e}")
    #    print("Please check your connection and try again.")
    #    return None # Exit the function if mounting fails


   
    pipeline = model_development_pipeline(
        data_path="CyberThreat_Insight/cybersecurity_data/x_y_augmented_data_google_drive.csv",
        target_col="Threat Level",
        deploy_folder="CyberThreat_Insight/model_deployment",
        lstm_timesteps=1)

    # Make metrics_df globally available
    global metrics_df
    metrics_df = pipeline["metrics_df"]

    best_model_name = pipeline["best_model_name"]
    best_model = pipeline["results"][best_model_name]["model"] # Get model object from results
    y_pred = pipeline["results"][best_model_name]["preds"]
    y_test = pipeline["y_test"]
    X_test = pipeline["X_test"] # Get original X_test

    #print main fearures
    print(f"\nMain Features: {X_test.columns} \n")
    display(X_test.head())

    metrics = pipeline["results"][best_model_name]["metrics"]
    best_model_metric = metrics["Overall Model Accuracy"]
    print("\nmetrics_df")
    display(metrics_df)
    print_model_metrics_charts(metrics_df)
    print(f"Overall Best model selected by Overall Model Accuracy: {best_model_name} -> Accuracy: {best_model_metric:.4f}")
    print(pipeline["best_model_path"])


    # Print aggregated performance metrics for the overall best model
    print(f"\n{best_model_name} Agreggated Peformance Metrics:")
    print_model_performance_report(best_model_name, y_test, y_pred)

    print(f"\nOverall Model Accuracy :  {best_model_metric}")

    # Get the classification report dictionary for the best model
    best_model_report = classification_report(y_test, y_pred, output_dict=True)

    # Call the generate_performance_insight function for the best model
    #generate_performance_insight(best_model_name, best_model_report, y_test, y_pred)


    print("\n Model Performance Visualisation:\n")

    # Find the best unsupervised model
    unsupervised_models = ["IsolationForest", "OneClassSVM", "LocalOutlierFactor", "DBSCAN", "KMeans", "Autoencoder"]
    unsupervised_metrics = metrics_df[metrics_df['model'].isin(unsupervised_models)]

    if not unsupervised_metrics.empty:
        best_unsupervised_row = unsupervised_metrics.sort_values(by="Overall Model Accuracy", ascending=False).iloc[0]
        best_unsupervised_model_name = best_unsupervised_row["model"]
        best_unsupervised_metrics = pipeline["results"][best_unsupervised_model_name]["metrics"]

        print(f"\nBest Unsupervised model selected by Overall Model Accuracy: {best_unsupervised_model_name} -> Accuracy: {best_unsupervised_metrics['Overall Model Accuracy']:.4f}")

        # Retrieve the augmented X_test DataFrame for the best unsupervised model
        X_test_for_viz = pipeline["results"][best_unsupervised_model_name].get("X_test_viz")

        if X_test_for_viz is not None:
            print(f"\nVisualizing performance for the best unsupervised model: {best_unsupervised_model_name}")
            #display(X_test_for_viz.head()) # Display head of the visualization data
            # Visualize using the augmented DataFrame with generic anomaly columns
            visualizing_model_performance_pipeline(
                data=X_test_for_viz,
                x="Session Duration in Second",
                y="Data Transfer MB",
                anomaly_score="anomaly_score",  # Use generic column name
                is_anomaly="is_anomaly",      # Use generic column name
                best_unsupervised_model_name=best_unsupervised_model_name,
                title=f"{best_unsupervised_model_name} Performance Visualization\n"
            )
        else:
             print(f"Visualization data (X_test_viz) not available for the best unsupervised model: {best_unsupervised_model_name}.")
             print("Skipping detailed anomaly visualization for this model.")
    else:
        print("No unsupervised models found in the metrics dataframe.")


    print("\nModel development pipeline completed.")


# -------------------------
# If run as script - example
# -------------------------
if __name__ == "__main__":
    main_model_delopement_function()


```

</details



We developed and evaluated several models, including Random Forest, Gradient Boosting, Isolation Forest, One-Class SVM, Local Outlier Factor, DBSCAN, KMeans, and an LSTM Classifier. The models were trained and tested on augmented cybersecurity data.

The performance of each model was assessed using standard multiclass classification metrics, including Accuracy, Precision, Recall, and F1-Score. The results are summarized in the following table:

| model              | Overall Model Accuracy | Precision (Macro) | Recall (Macro) | F1 Score (Macro) |
| :----------------- | :--------------------- | :---------------- | :------------- | :--------------- |
| RandomForest       | 0.981203               | 0.970459          | 0.893963       | 0.926572         |
| GradientBoosting   | 0.978697               | 0.973815          | 0.884219       | 0.919050         |
| IsolationForest    | 0.588972               | 0.147243          | 0.250000       | 0.185331         |
| OneClassSVM        | 0.588972               | 0.147243          | 0.250000       | 0.185331         |
| LocalOutlierFactor | 0.588972               | 0.147243          | 0.250000       | 0.185331         |
| DBSCAN             | 0.588972               | 0.147243          | 0.250000       | 0.185331         |
| KMeans             | 0.724311               | 0.353093          | 0.364184       | 0.354422         |
| Autoencoder        | 0.588972               | 0.147243          | 0.250000       | 0.185331         |
| LSTM(Classifier)   | 0.775689               | 0.377089          | 0.407664       | 0.391757         |


<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_improved_metrics_bar.png" 
       alt="Centered Image" 
       style="width: 1oo%; height: Auto;">
</p>   

Based on the overall model accuracy, the **RandomForest** model was identified as the best performing model with an accuracy of 0.9812.

### Performance Analysis of the Best Model (RandomForest)

The RandomForest model demonstrated strong performance across all threat levels, particularly for 'Low' and 'High' severity. The classification report and confusion matrix provide detailed insights into its performance:

*   **Classification Report:** Shows high precision and recall for 'Low' and 'High' classes. While the precision for 'Critical' threats is good (0.9444), the recall is lower (0.6800), indicating that some critical events are being missed.
*   **Confusion Matrix:** Visually confirms the model's ability to correctly classify most instances. However, it also highlights the misclassifications, particularly the instances of 'Critical' threats being misclassified as 'Low'.
*   **Aggregated Metrics:** The macro and weighted averages for Precision, Recall, and F1 Score further support the model's overall strong performance.
*   **Per-class Metrics (Bell Curves):** The bell curves for PPV, Sensitivity, NPV, and Specificity provide a visual representation of the model's performance on each class. The 'Critical' class shows a wider spread in the Sensitivity curve, reflecting the lower recall for this class.

The performance insight for the RandomForest model emphasizes its strong overall accuracy but also points out the need to improve recall for 'Critical' threats to minimize risk.  

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_improved_confusion Matrix2.png" 
       alt="Centered Image" 
       style="width: 1oo%; height: Auto;">
</p> 
 
<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_improved_metrics_curves.png" 
       alt="Centered Image" 
       style="width: 1oo%; height: Auto;">
</p> 

### Performance Insight for RandomForest  

**Metrics per Class:**  


  **Class: Low**  
    Precision: 0.9812 - Of all instances predicted as 'Low', this is the proportion that were actually 'Low'.
    Recall: 0.9958 - Of all actual 'Low' instances, this is the proportion that the model correctly identified.
    F1-Score: 0.9884 - A balanced measure of Precision and Recall for this class.
    Support: 471.0 - The number of actual instances of this class in the test set.
    PPV (Positive Predictive Value): 0.9812 - The probability that a positive prediction is a true positive.
    NPV (Negative Predictive Value): 0.9938 - The probability that a negative prediction is a true negative.
    Sensitivity (True Positive Rate): 0.9958 - Ability to correctly identify positive instances.
    Specificity (True Negative Rate): 0.9726 - Ability to correctly identify negative instances.

  **Class: Medium**    
    Precision: 1.0000 - Of all instances predicted as 'Medium', this is the proportion that were actually 'Medium'.
    Recall: 0.8667 - Of all actual 'Medium' instances, this is the proportion that the model correctly identified.
    F1-Score: 0.9286 - A balanced measure of Precision and Recall for this class.
    Support: 30.0 - The number of actual instances of this class in the test set.
    PPV (Positive Predictive Value): 1.0000 - The probability that a positive prediction is a true positive.
    NPV (Negative Predictive Value): 0.9948 - The probability that a negative prediction is a true negative.
    Sensitivity (True Positive Rate): 0.8667 - Ability to correctly identify positive instances.
    Specificity (True Negative Rate): 1.0000 - Ability to correctly identify negative instances.

  **Class: High**   
    Precision: 0.9819 - Of all instances predicted as 'High', this is the proportion that were actually 'High'.
    Recall: 1.0000 - Of all actual 'High' instances, this is the proportion that the model correctly identified.
    F1-Score: 0.9909 - A balanced measure of Precision and Recall for this class.
    Support: 271.0 - The number of actual instances of this class in the test set.
    PPV (Positive Predictive Value): 0.9819 - The probability that a positive prediction is a true positive.
    NPV (Negative Predictive Value): 1.0000 - The probability that a negative prediction is a true negative.
    Sensitivity (True Positive Rate): 1.0000 - Ability to correctly identify positive instances.
    Specificity (True Negative Rate): 0.9905 - Ability to correctly identify negative instances.

  **Class: Critical**    
    Precision: 0.9474 - Of all instances predicted as 'Critical', this is the proportion that were actually 'Critical'.
    Recall: 0.6667 - Of all actual 'Critical' instances, this is the proportion that the model correctly identified.
    F1-Score: 0.7826 - A balanced measure of Precision and Recall for this class.
    Support: 27.0 - The number of actual instances of this class in the test set.
    PPV (Positive Predictive Value): 0.9474 - The probability that a positive prediction is a true positive.
    NPV (Negative Predictive Value): 0.9885 - The probability that a negative prediction is a true negative.
    Sensitivity (True Positive Rate): 0.6667 - Ability to correctly identify positive instances.
    Specificity (True Negative Rate): 0.9987 - Ability to correctly identify negative instances.

**Aggregated Metrics:**    
  Accuracy: 0.9812 - The overall proportion of correctly classified instances.
  Precision (Macro): 0.9776 - Average Precision across all classes, unweighted.
  Recall (Macro): 0.8823 - Average Recall across all classes, unweighted.
  F1 Score (Macro): 0.9226 - Average F1-Score across all classes, unweighted.
  Precision (Weighted): 0.9810 - Average Precision across all classes, weighted by support.
  Recall (Weighted): 0.9812 - Average Recall across all classes, weighted by support.
  F1 Score (Weighted): 0.9800 - Average F1-Score across all classes, weighted by support.

**Business Insights:**    
- The model 'RandomForest' achieved an overall accuracy of 0.9812.
- The model demonstrates strong overall performance across all major metrics and classes.
- For 'Critical' threats (Recall: 0.6667, Precision: 0.9474), the model is able to capture a significant portion of actual critical events (Recall), and when it flags an event as critical, it is often correct (Precision). However, further investigation into missed critical events might be warranted to minimize risk.
- Improving recall for Critical threats should be a priority to ensure all high-severity events are detected.
- Consider deploying this model for automated threat detection, but maintain human oversight, especially for high-severity alerts.
- Continuously monitor model performance in a production environment as threat patterns can evolve.
- Investigate misclassified instances to understand the limitations and potential areas for model improvement or data enhancement.
 

### Visualization of Unsupervised Model Performance (KMeans)

To understand the anomaly detection capabilities of the unsupervised models, we visualized the performance of the best unsupervised model, KMeans (Accuracy: 0.7243). The visualization included:

*   **Scatter Plot:** Showed a visual separation between points identified as anomalies (red) and normal points (blue) based on 'Session Duration in Second' and 'Data Transfer MB'.
*   **ROC Curve:** An AUC of 1.00 indicated excellent discriminative power between the identified anomalies and normal points in this specific visualization based on the KMeans clustering.
*   **Precision-Recall Curve:** Highlighted the trade-off between precision and recall for the identified anomalies.


 Model Performance Visualisation:  

<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_improved_ROC_RECALL_curves.png" 
       alt="Centered Image" 
       style="width: 1oo%; height: Auto;">
</p>  

Best Unsupervised model selected by Overall Model Accuracy: KMeans -> Accuracy: 0.8048

### Visualization Insights for KMeans

Scatter Plot Insight:  
- The scatter plot visually separates points identified as anomalies (38 points, typically shown in red) from normal points (761 points, typically shown in blue).
- A clear separation between the red and blue points indicates that the model is effectively distinguishing between normal and anomalous behavior based on the chosen features.

ROC Curve Insight:  
- The ROC curve shows the trade-off between the True Positive Rate (ability to detect actual anomalies) and the False Positive Rate (incorrectly flagging normal points as anomalies) at various threshold settings.
- An Area Under the ROC Curve (AUC) of 1.0000 indicates the overall ability of the model to discriminate between positive (anomaly) and negative (normal) classes.
- An AUC value above 0.8 suggests good discriminative power.

Precision-Recall Curve Insight:  
- The Precision-Recall curve highlights the trade-off between Precision (accuracy of positive predictions) and Recall (completeness of positive predictions) as the decision threshold is varied.
- This curve is particularly informative for imbalanced datasets, where the number of anomalies is much smaller than normal instances.
- A curve that stays high as recall increases indicates that the model can achieve high precision even when identifying a large proportion of actual anomalies.

Business Insights from Visualization:  
- The visualization for the KMeans suggests that the model has good potential for identifying anomalous behavior. The clear separation in the scatter plot and the high AUC value are positive indicators.
- This model could be used to prioritize events for further investigation by security analysts.
- The shape of the Precision-Recall curve provides insight into how many anomalies the model can find before its positive predictions become unreliable. A curve that drops sharply indicates that increasing recall comes at a high cost of precision (more false alarms).
The visualization insights suggest that KMeans, in this context, is effective at identifying data points that are distant from the cluster centers, which are treated as anomalies.

### Conclusion

This project successfully developed and evaluated a range of machine learning models for cybersecurity threat detection. The combined approach of supervised and unsupervised learning provides a comprehensive framework for identifying both known and potentially novel threats. The RandomForest model emerged as the best performer based on overall accuracy. While the model shows strong overall performance, future work should focus on improving the detection of 'Critical' threats, potentially through further feature engineering, exploring different model architectures, or implementing techniques specifically designed to handle imbalanced datasets and rare events.

### Future Work and Improvements

*   **Advanced Feature Engineering:** Explore creating more complex features that capture temporal dependencies or interaction patterns indicative of anomalous behavior.
*   **Hyperparameter Tuning:** Conduct more extensive hyperparameter tuning for all models, especially for the best performing ones and those with potential for improvement (e.g., LSTM, unsupervised models).
*   **Ensemble Methods:** Investigate combining multiple models (e.g., creating an ensemble of the best supervised and unsupervised models) to potentially improve overall performance and robustness.
*   **Handling Imbalanced Data:** Implement techniques specifically designed for imbalanced datasets, such as oversampling minority classes (e.g., Critical threats), undersampling majority classes, or using algorithms that are less sensitive to imbalance.
*   **Real-time Detection:** Adapt the models and pipeline for real-time or near-real-time threat detection in a streaming data environment.
*   **Model Explainability:** Incorporate techniques to explain model predictions, particularly for critical alerts, to provide security analysts with actionable insights.
*   **Integration with Security Systems:** Explore integrating the developed engine with existing security information and event management (SIEM) systems or other security platforms.
*   **Exploring Other Models:** Evaluate additional models, including deep learning architectures specifically designed for anomaly detection or time series analysis.  

As next step, we will work to improve the this model with a stacked supervised model using unsupervised anomaly features .
 Click here to check the next step:
<a 
  href="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/model_dev/stacked_model/README.md"
  target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

## 🤝 Connect with me
I am always open to collaboration and discussion about new projects or technical roles.

Atsu Vovor  
Consultant, Data & Analytics    
Ph: 416-795-8246 | ✉️ atsu.vovor@bell.net    
🔗 <a href="https://www.linkedin.com/in/atsu-vovor-mmai-9188326/" target="_blank">LinkedIn</a> | <a href="https://atsuvovor.github.io/projects_portfolio.github.io/" target="_blank">GitHub</a> | <a href="https://public.tableau.com/app/profile/atsu.vovor8645/vizzes" target="_blank">Tableau Portfolio</a>    
📍 Mississauga ON      

### Thank you for visiting!🙏

