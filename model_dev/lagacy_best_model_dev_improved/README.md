## **CyberThreat-Insight**  
**Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI**


# Model Development - Cyber Threat Detection Engine(improved version)
## Anomalous Behavior Detection in Cybersecurity Analytics

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

### Performance Analysis of the Best Model (RandomForest)

The RandomForest model demonstrated strong performance across all threat levels, particularly for 'Low' and 'High' severity. The classification report and confusion matrix provide detailed insights into its performance:

*   **Classification Report:** Shows high precision and recall for 'Low' and 'High' classes. While the precision for 'Critical' threats is good (0.9444), the recall is lower (0.6800), indicating that some critical events are being missed.
*   **Confusion Matrix:** Visually confirms the model's ability to correctly classify most instances. However, it also highlights the misclassifications, particularly the instances of 'Critical' threats being misclassified as 'Low'.
*   **Aggregated Metrics:** The macro and weighted averages for Precision, Recall, and F1 Score further support the model's overall strong performance.
*   **Per-class Metrics (Bell Curves):** The bell curves for PPV, Sensitivity, NPV, and Specificity provide a visual representation of the model's performance on each class. The 'Critical' class shows a wider spread in the Sensitivity curve, reflecting the lower recall for this class.

The performance insight for the RandomForest model emphasizes its strong overall accuracy but also points out the need to improve recall for 'Critical' threats to minimize risk.

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
