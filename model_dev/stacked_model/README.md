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
