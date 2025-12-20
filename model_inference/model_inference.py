# --------------------------
#   Necessary Imports
# --------------------------
import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
import sys


#--------------------------------
# Add the directory containing your modules to the system path
sys.path.append('/content/CyberThreat_Insight/model_inference')
#CyberThreat_Insight.feature_engineering/
from CyberThreat_Insight.feature_engineering.f_engineering import (load_objects_from_drive,
                                                                   features_engineering_pipeline,
                                                                   data_augmentation_pipeline)
#CyberThreat_Insight/model_dev/stacked_model/
from CyberThreat_Insight.model_dev.stacked_model.stacked_anomaly_detection_classifier import (print_model_performance_report,
                                                                                            visualizing_model_performance_pipeline,
                                                                                            get_best_unsupervised_model_corrected)

#get_best_unsupervised_model_corrected(X_test_scaled, y_test, unsupervised_models)
#best_model_name, best_anomaly_score, best_is_anomaly = get_best_unsupervised_model_corrected(X_test_scaled, y_test, unsupervised_models)
# --------------------------
#   Configurations
# --------------------------
#Stacked Supervised Model using Unsupervised Anomaly Features
MODEL_TYPE = "Stacked Supervised Model using Unsupervised Anomaly Features"
MODEL_NAME = "Stacked_AD_classifier"
THREASHHOLD_PERC = 95
LABEL_COL = "Threat Level"  # Ground truth label column name
MODELS_DIR = "CyberThreat_Insight/stacked_models_deployment"
SIMULATED_REAL_TIME_DATA_FILE = \
                         "/content/drive/My Drive/Cybersecurity Data/normal_and_anomalous_cybersecurity_dataset_for_google_drive_kb.csv"

# --------------------------
#   Ensure model directory exists
# --------------------------
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# --------------------------
#   Logging Function
# --------------------------
def log(msg):
    """Logs a message to both console and a file in MODEL_OUTPUT_DIR."""
    print(f"[INFO] {msg}")
    with open(os.path.join(MODELS_DIR, "log.txt"), "a") as f:
        f.write(f"{msg}\n")

# --------------------------
#   Check Required Model Files (Table View)
# --------------------------
def check_required_files(output_dir):
    """Checks if all model/scaler files exist before loading, shows table."""
    required_files = [
        "scaler.joblib", "rf_base.joblib", "gb_meta.joblib",
        "iso.joblib", "ocsvm.joblib", "lof.joblib", "dbscan.joblib", "kmeans.joblib",
        "train_X_scaled.npy", "dense_autoencoder.keras", "lstm_autoencoder.keras"
    ]

    print("\n📂 Checking Required Model Files:\n" + "-" * 50)
    missing_files = []

    for f in required_files:
        file_path = os.path.join(output_dir, f)
        if os.path.exists(file_path):
            print(f"✅ {f} — FOUND")
        else:
            print(f"❌ {f} — MISSING")
            missing_files.append(f)

    print("-" * 50)

    if missing_files:
        raise FileNotFoundError(f"\nMissing required model files:\n - " + "\n - ".join(missing_files))

#-----------------------------
#  Anomaly Features Inference
#----------------------------
def extract_anomaly_features_inference_(X_scaled, unsupervised_models):
    """Extracts anomaly features from X_scaled using unsupervised models."""
    log("Extracting anomaly features...")
    anomaly_features = {}
    for name, model in unsupervised_models.items():
        if name in ['train_X', 'dense_ae', 'lstm_ae']:
            continue
        anomaly_features[name] = model.predict(X_scaled)
        return pd.DataFrame(anomaly_features)

def extract_anomaly_features_inference(X_scaled, unsupervised_models):
    """
    Use trained unsupervised models (and saved train_X) to compute anomaly features for new X_scaled.
    Does NOT retrain any models.
    """
    features = pd.DataFrame(index=np.arange(X_scaled.shape[0]))

    iso = unsupervised_models['iso']
    ocsvm = unsupervised_models['ocsvm']
    lof = unsupervised_models['lof']
    db = unsupervised_models['dbscan']
    kmeans = unsupervised_models['kmeans']
    dense_ae = unsupervised_models['dense_ae']
    lstm_ae = unsupervised_models['lstm_ae']
    train_X = unsupervised_models.get('train_X', None)
    if train_X is None:
        raise ValueError("Missing 'train_X' in unsupervised_models; needed for DBSCAN assignment.")

    # IsolationForest
    features['iso_df'] = iso.decision_function(X_scaled)

    # One-Class SVM
    features['ocsvm_df'] = ocsvm.decision_function(X_scaled)

    # LOF (novelty=True required for decision_function on new data)
    features['lof_df'] = lof.decision_function(X_scaled)

    # DBSCAN assignment using nearest neighbor to training samples
    nbrs = NearestNeighbors(n_neighbors=1).fit(train_X)
    nn_idx = nbrs.kneighbors(X_scaled, return_distance=False)[:, 0]
    db_labels_train = db.labels_
    assigned_train_labels = db_labels_train[nn_idx]
    features['dbscan_anomaly'] = (assigned_train_labels == -1).astype(float)

    # KMeans distance to cluster centers
    k_labels = kmeans.predict(X_scaled)
    k_dist = np.linalg.norm(X_scaled - kmeans.cluster_centers_[k_labels], axis=1)
    features['kmeans_dist'] = k_dist

    # Dense AE MSE
    features['ae_mse'] = np.mean((X_scaled - dense_ae.predict(X_scaled, verbose=0)) ** 2, axis=1)

    # LSTM AE MSE (reshape)
    timesteps = 1
    input_dim = X_scaled.shape[1]
    X_seq = X_scaled.reshape((X_scaled.shape[0], timesteps, input_dim))
    features['lstm_mse'] = np.mean((X_seq - lstm_ae.predict(X_seq, verbose=0)) ** 2, axis=(1, 2))

    return features
# --------------------------
#   Load Trained Features
# --------------------------
def load_treaned_features(scaler, input_data):
    """Ensures new data matches the trained feature set."""
    log("Loading trained features...")

    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"Input CSV file not found: {input_data}")
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data must be a filepath or a pandas DataFrame.")

    if LABEL_COL in df.columns:
        df = df.drop(columns=[LABEL_COL])

    trained_feature_names = list(scaler.feature_names_in_)
    X_new = df[[c for c in df.columns if c in trained_feature_names]].copy()

    for col in trained_feature_names:
        if col not in X_new.columns:
            X_new[col] = 0

    X_new = X_new[trained_feature_names]
    return X_new

# --------------------------
#   Load Scaler and Models
# --------------------------
def load_scaler_and_models(output_dir):
    """Loads scaler, supervised models, and unsupervised models from output_dir."""
    check_required_files(output_dir)  # Ensure all files exist before loading

    scaler = joblib.load(os.path.join(output_dir, "scaler.joblib"))
    base_model = joblib.load(os.path.join(output_dir, "rf_base.joblib"))
    meta_model = joblib.load(os.path.join(output_dir, "gb_meta.joblib"))

    unsupervised_models = {}
    for name in ['iso', 'ocsvm', 'lof', 'dbscan', 'kmeans']:
        unsupervised_models[name] = joblib.load(os.path.join(output_dir, f"{name}.joblib"))

    unsupervised_models['train_X'] = np.load(os.path.join(output_dir, "train_X_scaled.npy"))
    unsupervised_models['dense_ae'] = load_model(os.path.join(output_dir, "dense_autoencoder.keras"))
    unsupervised_models['lstm_ae'] = load_model(os.path.join(output_dir, "lstm_autoencoder.keras"))

    return scaler, base_model, meta_model, unsupervised_models

#----------------------------------
# Encode Simulated Real Time Data
#----------------------------------

def encode_simulated_real_time_data(df_p, LABEL_COL):

    df = df_p.copy()
    fe_processed_df, loaded_label_encoders, num_fe_scaler = load_objects_from_drive()
    features_engineering_columns = fe_processed_df.columns.tolist()
    input_feature_column = [col for col in features_engineering_columns if col != LABEL_COL]
    features_engineering_columns.remove(LABEL_COL)

    #encode features using loaded_label_encoders and num_fe_scaler
    for col, encoder in loaded_label_encoders.items():
        df[col] = encoder.transform(df[col])

    return df


# --------------------------
#   Prediction Function
# --------------------------

def model_2SM2UAF_predict_anomaly_features_inference(encoded_df,
                                                     y_pred,
                                                     y_test,
                                                     LABEL_COL,
                                                     threashhold_perc = 95):

    #encoded_df = encode_simulated_real_time_data(df, LABEL_COL)
    #features_engineering_columns = df.columns.tolist()
    #features_engineering_columns.remove(LABEL_COL)

    #df[features_engineering_columns] = num_fe_scaler.transform(df[features_engineering_columns])
    #threshold = np.percentile(y_probas, threashhold_perc)
    #encoded_df["Pred Threat"] = y_pred
    #--------------------------------
    mse = np.mean(np.power(y_test - y_pred, 2))
    threshold = np.percentile(mse, threashhold_perc)
    encoded_df["anomaly_score"] = mse
    encoded_df["is_anomaly"] = encoded_df["anomaly_score"] > threshold
    #-----------------------------
    #encoded_df["Pred Threat Probability"] = y_probas
    #encoded_df["anomaly_score"] = y_probas
    #ncoded_df["is_anomaly"] = encoded_df["anomaly_score"] > threshold

    #------------------------------
    #y_preds = model.predict(df[input_feature_column])
    #    df["Pred Threat"] = y_preds
    #    mse = np.mean(np.power(df[input_feature_column] - y_preds, 2), axis=1)
    #    threshold = np.percentile(mse, 95)
    #    df["anomaly_score"] = mse
    #-----------------------------

    return encoded_df


def predict_new_data(input_data, LABEL_COL, model_dir=MODELS_DIR):


    """Predicts on new data and evaluates if LABEL_COL exists."""
    log("Loading scaler and models for inference...")
    scaler, base_model, meta_model, unsupervised_models = load_scaler_and_models(model_dir)

    #if isinstance(input_data, str):
    #    df_raw = pd.read_csv(input_data)
    #elif isinstance(input_data, pd.DataFrame):
    #    df_raw = input_data.copy()
    #else:
    #    raise TypeError("input_data must be a filepath or a pandas DataFrame.")

    #encoded_df_raw = encode_simulated_real_time_data(df_raw, LABEL_COL)
    #----------------------------------------------------
    #features_engineering_columns = encoded_df_raw.columns.tolist()
    #features_engineering_columns.remove(LABEL_COL)

    if isinstance(input_data, str):
        augmented_df, d_loss_real_list, d_loss_fake_list, g_loss_list = data_augmentation_pipeline(
                                    file_path=input_data,
                                    lead_save_true_false = False)
        encoded_df_raw = augmented_df.copy()
    else:
        log("Initial data frame is empty...")
        raise TypeError("input_data must be a filepath ")


    #-------------------------------------------------------------------

    y_test = encoded_df_raw[LABEL_COL] if LABEL_COL in encoded_df_raw.columns else None
    X_new = load_treaned_features(scaler, encoded_df_raw)

    log("Scaling input features...")
    X_scaled = scaler.transform(X_new)

    log("Extracting anomaly features...")
    anomaly_features = extract_anomaly_features_inference(X_scaled, unsupervised_models)

    X_ext = pd.concat(
        [pd.DataFrame(X_scaled, columns=X_new.columns).reset_index(drop=True),
         anomaly_features.reset_index(drop=True)],
        axis=1
    )

    base_proba = base_model.predict_proba(X_ext)
    X_stack = np.hstack([X_ext.values, base_proba])
    y_pred = meta_model.predict(X_stack)
    y_proba = meta_model.predict_proba(X_stack)

    log("Prediction complete.")

    if y_test is not None:
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        #cm = confusion_matrix(y_test, y_pred)

        #print(f"\nAccuracy: {acc:.4f}")
        print("\nClassification Report:\n", report)
        #print("\nConfusion Matrix:\n", cm)

        df_raw_anomaly_pred = model_2SM2UAF_predict_anomaly_features_inference(encoded_df_raw,
                                                                               y_pred,
                                                                               y_test,
                                                                               LABEL_COL,
                                                                               THREASHHOLD_PERC)

        best_model_name, best_anomaly_score, best_is_anomaly = get_best_unsupervised_model_corrected(X_new,
                                                                                                     y_test, unsupervised_models)

        model_metrics_dic = print_model_performance_report(MODEL_NAME, y_test, y_pred)
        best_anomaly_score, best_is_anomaly
        visualizing_model_performance_pipeline(
            data=df_raw_anomaly_pred,
            x="Impact Score",#"Session Duration in Second",
            y="Data Transfer MB",# #'Threat Score'
            anomaly_score="anomaly_score",  # Use model_type to construct column name
            is_anomaly="is_anomaly",  # Use model_type to construct column name
            best_unsupervised_model_name = best_model_name,
            title="Model Performance Visualization\n"
            )
            #'Threat Score', 'Impact Score'
    return y_pred, y_proba
# This cell defines a function to format and display model inference output with dynamic insights.

def display_model_inference_output(preds, probs, class_names):
    """
    Formats and displays model inference output (predicted classes and probabilities)
    with dynamic explanations and business insights.

    Args:
        preds (np.ndarray): Array of predicted class labels.
        probs (np.ndarray): Array of prediction probabilities.
        class_names (dict): Mapping from numerical class labels to names.
    """
    # --- Explanation and Business Insight ---
    print("--- Model Prediction Output Analysis ---")
    print("\nBased on the model's inference results:")

    # Display the shape of the predicted classes array.
    # This shows the total number of instances for which a class prediction was made.
    num_instances = preds.shape[0]
    print(f"\nShape of Predicted Classes: {preds.shape}")
    print(f"Business Insight: The model processed and made predictions for a total of {num_instances} instances.")

    # Display the shape of the prediction probabilities array.
    # This shows the total number of instances and the number of classes (columns) with probability scores.
    num_classes = probs.shape[1]
    print(f"\nShape of Prediction Probabilities: {probs.shape}")
    print(f"Business Insight: For each instance, the model provided a probability score for each of the {num_classes} possible threat levels.")


    print("\n--- First 10 Predictions and Probabilities ---")

    # Display the first 10 predicted class labels, including their names.
    print("\nFirst 10 Predicted Classes (Numerical and Name):")
    for i, pred in enumerate(preds[:10]):
        print(f"Instance {i+1}: {pred} ({class_names.get(pred, 'Unknown Class')})")


    # Display the first 10 rows of prediction probabilities, rounded for clarity.
    # Each row shows the probability of the instance belonging to each of the 4 classes.
    print("\nFirst 10 Prediction Probabilities:")
    # Create a temporary DataFrame to display with column names
    probs_df = pd.DataFrame(probs[:10], columns=[class_names.get(i, f'Class {i}') for i in range(probs.shape[1])])
    display(np.round(probs_df, 4))
    print("Business Insight: Examining the probabilities for individual instances shows the model's confidence in its predictions for specific events.")


    print("\n--- Prediction Probability Summary Statistics ---")

    # Display the average probability across all predictions and all classes.
    avg_prob = np.mean(probs)
    print(f"\nAverage Prediction Probability (across all classes and instances): {avg_prob:.4f}")
    insight_avg_prob = f"An average probability around {1/num_classes:.2f} (for {num_classes} classes) might suggest a relatively balanced distribution of predictions or model uncertainty across classes. Further analysis of the probability distribution is recommended." if num_classes > 0 else "Cannot calculate average probability with 0 classes."
    print(f"Business Insight: {insight_avg_prob}")


    # Display the maximum probability assigned to any class for any instance.
    max_prob = np.max(probs)
    print(f"\nMaximum Prediction Probability (assigned to any class for any instance): {max_prob:.4f}")
    insight_max_prob = "A maximum probability of 1.0 indicates the model is highly confident in some of its individual predictions." if max_prob == 1.0 else "The maximum probability is less than 1.0, suggesting the model has some level of uncertainty even in its most confident predictions."
    print(f"Business Insight: {insight_max_prob}")


    # Display the minimum probability assigned to any class for any instance.
    min_prob = np.min(probs)
    print(f"\nMinimum Prediction Probability (assigned to any class for any instance): {min_prob:.4f}")
    insight_min_prob = "A minimum probability of 0.0 means the model is completely certain some instances do not belong to certain classes." if min_prob == 0.0 else "The minimum probability is greater than 0.0, suggesting the model assigns some non-zero probability to all classes for all instances."
    print(f"Business Insight: {insight_min_prob}")


    print("\n--- Overall Business Insight from Prediction Output ---")
    print("""
This output provides a snapshot of the model's inference phase.
- The **shapes** confirm the total number of instances processed and the number of classes evaluated.
- The **predicted classes** indicate the primary threat level identified for each instance, enabling prioritized operational responses.
- The **prediction probabilities** offer a measure of the model's confidence. While high confidence in individual cases is good, the overall average probability suggests further investigation into the probability distribution and model uncertainty is valuable.
- To gain more specific business insights, analyze the distribution of predicted threat levels across all instances and investigate instances with lower confidence scores.
""")

# --------------------------
#   Main Execution
# --------------------------
if __name__ == "__main__":
    preds, probs = predict_new_data(SIMULATED_REAL_TIME_DATA_FILE, LABEL_COL)

    # Log shapes
    #print(f"\nPredicted classes shape: {preds.shape}")
    #print(f"Prediction probabilities shape: {probs.shape}")

    # Show first few predictions
    #print("\nPredicted classes (first 10):", preds[:10])
    #print("Prediction probabilities (first 10):\n", np.round(probs[:10], 4))

    # Aggregate probability stats
    #print(f"\nAverage probability: {np.mean(probs):.4f}")
    #print(f"Max probability: {np.max(probs):.4f}")
    #print(f"Min probability: {np.min(probs):.4f}")

    class_names = {
        0: 'Low',
        1: 'Medium',
        2: 'High',
        3: 'Critical'
    }
    display_model_inference_output(preds, probs, class_names)
