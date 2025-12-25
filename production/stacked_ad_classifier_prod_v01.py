'''
#  CyberThreat-Insight – Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI  
##End User Documentation

**Toronto, Sept 17 202**  
**Autor : Atsu Vovor**
>Master of Management in Artificial Intelligence      
>Consultant Data Analytics Specialist | Machine Learning | Data   science | Quantitative Analysis |French & English Bilingual   



**Project:** Stacked Anomaly Detection Classifier  
**Model Type:** Stacked Supervised Classifier with Unsupervised Anomaly Features 
'''


import sys
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.neighbors import NearestNeighbors
#from google.colab import drive
# Mount Google Drive
#drive.mount('/content/drive')
# Add the directory containing your modules to the system path
sys.path.append('/content/CyberThreat_Insight/production')
from CyberThreat_Insight.utils.gdrive_utils import load_csv_from_gdrive_url

MODEL_DIR = "CyberThreat_Insight/stacked_models_deployment"
#DATA_FILE = "CyberThreat_Insight/cybersecurity_data/normal_and_anomalous_cybersecurity_dataset_for_google_drive_kb.csv"
#DATA_PATH =  "CyberThreat_Insight/cybersecurity_data"
#DATA_PATH = DATA_PATH + "/x_y_augmented_data_google_drive.csv"
DATA_PATH =  "CyberThreat_Insight/cybersecurity_data"
AUGMENTED_DATA_PATH = DATA_PATH + "/x_y_augmented_data_google_drive.csv"
NEW_DATA_URL = "https://drive.google.com/file/d/1Nr9PymyvLfDh3qTfaeKNVbvLwt7lNX6l/view?usp=sharing"
AUGMENTED_DATA_URL = "https://drive.google.com/file/d/10UYplPdqse328vu1S1tdUAlYMN_TJ8II/view?usp=sharing"
#------------------------------------------------------------------------------------------
def log(msg):
    print(f"[INFO] {msg} to {DATA_PATH}")
    

#load augmented data to mutch it columns with the operational data features for prediction
def load_aumented_dataset(AUGMENTED_DATA_URL = None, LABEL_COL = None, AUGMENTED_DATA_PATH = None):
    
    if AUGMENTED_DATA_PATH is not None:
        augmented_df = pd.read_csv(AUGMENTED_DATA_PATH)

    elif AUGMENTED_DATA_URL is not None:
        data_path = load_csv_from_gdrive_url(gdrive_url = AUGMENTED_DATA_URL,
                                            output_dir = "CyberThreat_Insight/cybersecurity_data",
                                            filename = "x_y_augmented_data_google_drive.csv")
      
        augmented_df = pd.read_csv(data_path)
    else:
        raise ValueError("No data source provided. Supply either a Google Drive URL or a DataFrame.")
    if LABEL_COL not in augmented_df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in dataset.")
    # ensure integer labels
    augmented_df[LABEL_COL] = augmented_df[LABEL_COL].astype(int)

    return augmented_df
    
def load_new_data(URL, LABEL_COL, df = None):
    """
    Loads the dataset and splits it into training and testing sets.
    """

    #validate df
    if df is not None:
       new_data = df.copy()
        
    if URL is not None:
        log("Loading operational dataset from Google Drive ...")
        data_path = load_csv_from_gdrive_url(gdrive_url = URL,
                                             output_dir = "CyberThreat_Insight/cybersecurity_data",
                                             filename = "normal_and_anomalous_cybersecurity_dataset_for_google_drive_kb.csv")
              
        new_data = pd.read_csv(data_path)
        
    return new_data


def predict_new_data(NEW_DATA_URL = None, AUGMENTED_DATA_PATH = None, model_dir = None, ops_df = None, label_col="Threat Level"):
    """
    Run inference on new data using pretrained stacked anomaly detection classifier.

    Args:
        csv_path (str): Path to new CSV file.
        model_dir (str): Directory containing saved artifacts.
        label_col (str): Name of the target column (if exists, it will be dropped).

    Returns:
        pd.DataFrame: Original data with predictions and anomaly metadata appended.
    """
    # --- Load models ---
    scaler = joblib.load(f"{model_dir}/scaler.joblib")
    rf_base = joblib.load(f"{model_dir}/rf_base.joblib")
    gb_meta = joblib.load(f"{model_dir}/gb_meta.joblib")

    iso = joblib.load(f"{model_dir}/iso.joblib")
    ocsvm = joblib.load(f"{model_dir}/ocsvm.joblib")
    lof = joblib.load(f"{model_dir}/lof.joblib")
    dbscan = joblib.load(f"{model_dir}/dbscan.joblib")
    kmeans = joblib.load(f"{model_dir}/kmeans.joblib")
    kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float32)

    dense_autoencoder = load_model(f"{model_dir}/dense_autoencoder.keras")
    lstm_autoencoder = load_model(f"{model_dir}/lstm_autoencoder.keras")

    train_X_scaled = np.load(f"{model_dir}/train_X_scaled.npy")

    # --- Load new dataset ---
    
    df_new = load_new_data(NEW_DATA_URL, label_col, ops_df)

    augmented_df = load_aumented_dataset(AUGMENTED_DATA_URL = None, 
                                         LABEL_COL = label_col, 
                                         AUGMENTED_DATA_PATH = AUGMENTED_DATA_PATH)

    #X_new and df_augmented have the same column names
    X_new = df_new[augmented_df.columns].drop(columns=[label_col], errors="ignore")

    np.seterr(all="raise")  # catches silent numeric issues

    print("[INFO] Inference input dtypes:")
    print(X_new.dtypes.value_counts())
    # ---------------- FORCE NUMERIC + FLOAT32 ----------------
    X_new = (
        X_new
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    
    # ---------------- SCALE (RETURNS float64!) ----------------
    X_new_scaled = scaler.transform(X_new)

    # ---------------- FORCE BACK TO float32 ----------------
    X_new_scaled = np.asarray(X_new_scaled, dtype=np.float32)

    # ---------------- HARD SAFETY CHECK ----------------
    assert X_new_scaled.dtype == np.float32, X_new_scaled.dtype

    # ---------------- ANOMALY FEATURES ----------------
    #features_new = pd.DataFrame(index=np.arange(X_new_scaled.shape[0]))
    
    #X_new_scaled = (
    #pd.DataFrame(X_new_scaled)
    #.apply(pd.to_numeric, errors="coerce")
    #.fillna(0.0)
    #.astype(np.float32)
    #.to_numpy()
    #)
    #X_new_scaled = np.asarray(X_new, dtype=np.float32)
    
    # --- Generate anomaly features ---
    features_new = pd.DataFrame(index=np.arange(X_new_scaled.shape[0]))

    features_new['iso_df'] = iso.decision_function(X_new_scaled)
    features_new['ocsvm_df'] = ocsvm.decision_function(X_new_scaled)
    features_new['lof_df'] = lof.decision_function(X_new_scaled)

    # DBSCAN mapping
    nbrs = NearestNeighbors(n_neighbors=1).fit(train_X_scaled)
    nn_idx = nbrs.kneighbors(X_new_scaled, return_distance=False)[:, 0]
    assigned_labels = dbscan.labels_[nn_idx]
    features_new['dbscan_anomaly'] = (assigned_labels == -1).astype(float)

    # KMeans distances
    print("X dtype:", X_new_scaled.dtype)
    print("Centers dtype:", kmeans.cluster_centers_.dtype)

    k_labels = kmeans.predict(X_new_scaled)
    features_new['kmeans_dist'] = np.linalg.norm(
        X_new_scaled - kmeans.cluster_centers_[k_labels], axis=1
    )

    # Dense Autoencoder
    recon_ae = dense_autoencoder.predict(X_new_scaled, verbose=0)
    features_new['ae_mse'] = np.mean((X_new_scaled - recon_ae) ** 2, axis=1)

    # LSTM Autoencoder
    timesteps, input_dim = 1, X_new_scaled.shape[1]
    X_seq = X_new_scaled.reshape((X_new_scaled.shape[0], timesteps, input_dim))
    recon_lstm = lstm_autoencoder.predict(X_seq, verbose=0)
    features_new['lstm_mse'] = np.mean((X_seq - recon_lstm) ** 2, axis=(1,2))

    # --- Build extended features STACK FEATURES---
    X_new_ext = pd.DataFrame(
        np.hstack([X_new_scaled, features_new]),
        columns = list(X_new.columns) + list(features_new.columns)
    )

    # RF proba
    rf_proba = rf_base.predict_proba(X_new_ext)

    # Final stacked input
    X_new_stack = np.hstack([X_new_ext.values, rf_proba])

    # --- Predict threat levels ---
    y_pred = gb_meta.predict(X_new_stack)
    y_pred_proba = gb_meta.predict_proba(X_new_stack)

    # --- Add outputs back to df_new ---
    df_new['Predicted Threat Level'] = y_pred

    # Add anomaly info
    # map df_new[label_col]
    df_new[label_col] = df_new[label_col].map({'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3})
    df_new['true_anomaly'] = (df_new[label_col] >= 2).astype(int) if label_col in df_new else np.nan
    df_new['anomaly_score'] = features_new.mean(axis=1)  # aggregate score across detectors
    df_new['predicted_anomaly'] = (df_new['Predicted Threat Level'] >= 2).astype(int)

    return df_new


#main
if __name__ == "__main__":
    results_df = predict_new_data(NEW_DATA_URL, AUGMENTED_DATA_PATH, MODEL_DIR)
    display(results_df.head())

