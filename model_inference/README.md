<h3 align="center">Stacked Anomaly Detection Classifier Inference</h3>
<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_model_classifier2.png" 
       alt="Centered Image" 
       style="width: 800px; height: 40%;">
</p>  


**Toronto, Sept 17 2025**  

**Model Name:** Stacked Anomaly Detection Classifier

**Model Type:** Stacked Ensemble Model


🔄 👉Run Model Inference
<a 
  href="https://colab.research.google.com/github/atsuvovor/Cybersecurity-Data-Generator/blob/main/CyberInsightDataGenerator.ipynb" 
  target="_parent">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Project Overview

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

<h3 align="center">Stacked Anomaly Detection Classifier Model flowchart</h3>
<p align="center">
  <img src="https://github.com/atsuvovor/CyberThreat_Insight/blob/main/images/stacked_anomaly_classifier_flowchart.png" 
       alt="Stacked Model Flowchart"" 
       style="width: 800px; height: 40%;">
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

**Considerations for Production:**

*   **Scalability:** The deployment environment should be able to handle the volume and velocity of real-world security data.
*   **Latency:** For real-time threat detection, the model inference process needs to be fast.
*   **Monitoring:** Continuous monitoring of the model's performance in production is essential to detect concept drift or changes in threat patterns.
*   **Maintenance:** The model may need to be retrained periodically with new data to maintain its effectiveness.
*   **Integration:** The model needs to be integrated with existing security information and event management (SIEM) systems or other security tools.  


Conclusion  


This project successfully developed and evaluated a stacked anomaly detection classifier leveraging unsupervised anomaly features for enhanced cybersecurity threat detection. The integrated pipeline demonstrated the process from data loading and preprocessing through unsupervised feature engineering, stacked model training, and comprehensive evaluation. The results indicate the potential of this hybrid approach to improve the accuracy and interpretability of anomaly detection compared to using individual unsupervised methods alone. While the model shows promising performance, particularly in distinguishing various threat levels, the analysis also highlighted areas for potential improvement, such as enhancing the recall for critical threats and further optimizing individual unsupervised components. The saved model artifacts and detailed documentation provide a solid foundation for potential real-world deployment and further research into advanced anomaly detection techniques in cybersecurity.



---
## 🤝 Connect with me
I am always open to collaboration and discussion about new projects or technical roles.

Atsu Vovor  
Consultant, Data & Analytics    
Ph: 416-795-8246 | ✉️ atsu.vovor@bell.net    
🔗 <a href="https://www.linkedin.com/in/atsu-vovor-mmai-9188326/" target="_blank">LinkedIn</a> | <a href="https://atsuvovor.github.io/projects_portfolio.github.io/" target="_blank">GitHub</a> | <a href="https://public.tableau.com/app/profile/atsu.vovor8645/vizzes" target="_blank">Tableau Portfolio</a>    
📍 Mississauga ON      

### Thank you for visiting!🙏


