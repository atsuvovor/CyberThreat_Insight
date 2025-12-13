
<h1 align="center">
Model Development - Cyber Threat Detection Engine improved
</h1> 
<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_improved_headline.png" 
       alt="Centered Image" 
       style="width: 1000px; height: Auto;">
</p>  

<p align="center">
    Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI**
</p> 


### Introduction

Cybersecurity threats are constantly evolving, making traditional signature-based detection insufficient. Identifying anomalous behavior is crucial for detecting novel threats. This article describes a Cyber Threat Detection Engine that uses both supervised and unsupervised machine learning to classify known threats and identify new anomalies. We evaluate various models to build a robust system for classifying cyber threats by severity level (Low, Medium, High, Critical).

### Supervised vs Unsupervised Learning in this Project

This project utilizes both supervised and unsupervised learning approaches. Supervised models like Random Forest and Gradient Boosting are trained on labeled data to directly classify known threat levels. Unsupervised models such as Isolation Forest and KMeans are used for anomaly detection and clustering to identify potentially novel threats. The outputs of these unsupervised models are then mapped to the defined threat levels. This combined approach leverages labeled data for known threats while enabling the detection of new anomalies.

### Model Development and Evaluation

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

Based on the overall model accuracy, the **RandomForest** model was identified as the best performing model with an accuracy of 0.9812.
<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/lagacy_model_improved_metrics_bar.png" 
       alt="Centered Image" 
       style="width: 1oo%; height: Auto;">
</p> 

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

### Visualization of Unsupervised Model Performance (KMeans)

To understand the anomaly detection capabilities of the unsupervised models, we visualized the performance of the best unsupervised model, KMeans (Accuracy: 0.7243). The visualization included:

*   **Scatter Plot:** Showed a visual separation between points identified as anomalies (red) and normal points (blue) based on 'Session Duration in Second' and 'Data Transfer MB'.
*   **ROC Curve:** An AUC of 1.00 indicated excellent discriminative power between the identified anomalies and normal points in this specific visualization based on the KMeans clustering.
*   **Precision-Recall Curve:** Highlighted the trade-off between precision and recall for the identified anomalies.

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
---
put performace metrics bar charts here
---
Overall Best model selected by Overall Model Accuracy: RandomForest -> Accuracy: 0.9812
CyberThreat_Insight/model_deployment/RandomForest_best_model.joblib

RandomForest Agreggated Peformance Metrics:

RandomForest classification_report:
put confurion matrice and performance report here
---
put raandomforesr-multiclass meyrics bell curves here
---
Overall Model Accuracy :  0.981226533166458

--- Performance Insight for RandomForest ---

Metrics per Class:

  Class: Low
    Precision: 0.9812 - Of all instances predicted as 'Low', this is the proportion that were actually 'Low'.
    Recall: 0.9958 - Of all actual 'Low' instances, this is the proportion that the model correctly identified.
    F1-Score: 0.9884 - A balanced measure of Precision and Recall for this class.
    Support: 471.0 - The number of actual instances of this class in the test set.
    PPV (Positive Predictive Value): 0.9812 - The probability that a positive prediction is a true positive.
    NPV (Negative Predictive Value): 0.9938 - The probability that a negative prediction is a true negative.
    Sensitivity (True Positive Rate): 0.9958 - Ability to correctly identify positive instances.
    Specificity (True Negative Rate): 0.9726 - Ability to correctly identify negative instances.

  Class: Medium
    Precision: 1.0000 - Of all instances predicted as 'Medium', this is the proportion that were actually 'Medium'.
    Recall: 0.8667 - Of all actual 'Medium' instances, this is the proportion that the model correctly identified.
    F1-Score: 0.9286 - A balanced measure of Precision and Recall for this class.
    Support: 30.0 - The number of actual instances of this class in the test set.
    PPV (Positive Predictive Value): 1.0000 - The probability that a positive prediction is a true positive.
    NPV (Negative Predictive Value): 0.9948 - The probability that a negative prediction is a true negative.
    Sensitivity (True Positive Rate): 0.8667 - Ability to correctly identify positive instances.
    Specificity (True Negative Rate): 1.0000 - Ability to correctly identify negative instances.

  Class: High
    Precision: 0.9819 - Of all instances predicted as 'High', this is the proportion that were actually 'High'.
    Recall: 1.0000 - Of all actual 'High' instances, this is the proportion that the model correctly identified.
    F1-Score: 0.9909 - A balanced measure of Precision and Recall for this class.
    Support: 271.0 - The number of actual instances of this class in the test set.
    PPV (Positive Predictive Value): 0.9819 - The probability that a positive prediction is a true positive.
    NPV (Negative Predictive Value): 1.0000 - The probability that a negative prediction is a true negative.
    Sensitivity (True Positive Rate): 1.0000 - Ability to correctly identify positive instances.
    Specificity (True Negative Rate): 0.9905 - Ability to correctly identify negative instances.

  Class: Critical
    Precision: 0.9474 - Of all instances predicted as 'Critical', this is the proportion that were actually 'Critical'.
    Recall: 0.6667 - Of all actual 'Critical' instances, this is the proportion that the model correctly identified.
    F1-Score: 0.7826 - A balanced measure of Precision and Recall for this class.
    Support: 27.0 - The number of actual instances of this class in the test set.
    PPV (Positive Predictive Value): 0.9474 - The probability that a positive prediction is a true positive.
    NPV (Negative Predictive Value): 0.9885 - The probability that a negative prediction is a true negative.
    Sensitivity (True Positive Rate): 0.6667 - Ability to correctly identify positive instances.
    Specificity (True Negative Rate): 0.9987 - Ability to correctly identify negative instances.

Aggregated Metrics:
  Accuracy: 0.9812 - The overall proportion of correctly classified instances.
  Precision (Macro): 0.9776 - Average Precision across all classes, unweighted.
  Recall (Macro): 0.8823 - Average Recall across all classes, unweighted.
  F1 Score (Macro): 0.9226 - Average F1-Score across all classes, unweighted.
  Precision (Weighted): 0.9810 - Average Precision across all classes, weighted by support.
  Recall (Weighted): 0.9812 - Average Recall across all classes, weighted by support.
  F1 Score (Weighted): 0.9800 - Average F1-Score across all classes, weighted by support.

Business Insights:
- The model 'RandomForest' achieved an overall accuracy of 0.9812.
- The model demonstrates strong overall performance across all major metrics and classes.
- For 'Critical' threats (Recall: 0.6667, Precision: 0.9474), the model is able to capture a significant portion of actual critical events (Recall), and when it flags an event as critical, it is often correct (Precision). However, further investigation into missed critical events might be warranted to minimize risk.
- Improving recall for Critical threats should be a priority to ensure all high-severity events are detected.
- Consider deploying this model for automated threat detection, but maintain human oversight, especially for high-severity alerts.
- Continuously monitor model performance in a production environment as threat patterns can evolve.
- Investigate misclassified instances to understand the limitations and potential areas for model improvement or data enhancement.

Overall Model Accuracy :  0.981226533166458

 Model Performance Visualisation:


Best Unsupervised model selected by Overall Model Accuracy: KMeans -> Accuracy: 0.8048

---
put recall ROC scatter plot here
---

--- Visualization Insights for KMeans ---

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

End Visualization Insights for KMeans.

Model development pipeline completed.

Visualizing performance for the best unsupervised model: KMeans
