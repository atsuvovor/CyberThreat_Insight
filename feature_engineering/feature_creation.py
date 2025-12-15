#!/usr/bin/env python3
# --------------------------------------------------------------
# CyberThreat Insight – Cholesky-Based Perturbation
# SHAP, PCA, Data Augmentation (SMOTE + GAN)
# Author: Atsu Vovor
# --------------------------------------------------------------

# ==============================
# Feature Creation
# ==============================


# --- Imports ---
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

import shap


# --- Column Dictionary ---
def get_column_dic():
    columns_dic = {
        "numerical_columns": [
            "Timestamps", "Issue Response Time Days", "Impact Score", "Cost",
            "Session Duration in Second", "Num Files Accessed", "Login Attempts",
            "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score"
        ],
        "explanatory_data_analysis_columns": [
            "Date Reported", "Issue Response Time Days", "Impact Score", "Cost",
            "Session Duration in Second", "Num Files Accessed", "Login Attempts",
            "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score"
        ],
        "user_activity_features": [
            "Risk Level", "Issue Response Time Days", "Impact Score", "Cost",
            "Session Duration in Second", "Num Files Accessed", "Login Attempts",
            "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score"
        ],
        "initial_dates_columns": ["Date Reported", "Date Resolved", "Timestamps"],
        "categorical_columns": [
            "Issue ID", "Issue Key", "Issue Name", "Category", "Severity", "Status", "Reporters",
            "Assignees", "Risk Level", "Department Affected", "Remediation Steps", "KPI/KRI",
            "User ID", "Activity Type", "User Location", "IP Location", "Threat Level",
            "Defense Action", "Color"
        ],
        "features_engineering_columns": [
            "Issue Response Time Days", "Impact Score", "Cost",
            "Session Duration in Second", "Num Files Accessed", "Login Attempts",
            "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score", "Threat Level"
        ],
        "numerical_behavioral_features": [
            "Login Attempts", "Data Transfer MB", "CPU Usage %", "Memory Usage MB",
            "Session Duration in Second", "Num Files Accessed", "Threat Score"
        ]
    }
    return columns_dic


#from json import load
# ------------------------------Save df_fe, label_encoders anf numerical columns scaler to to your Google Drive------------------------
def save_objects_to_drive(df_fe,
                          cat_cols_label_encoders,
                          num_fe_scaler,
                          filepath_df="CyberThreat_Insight/cybersecurity_data/df_fe.pkl",
                          filepath_cat_cols_label_encoders="CyberThreat_Insight/model_deployment/cat_cols_label_encoders.pkl",
                          filepath_num_fe_scaler="CyberThreat_Insight/model_deployment/num_fe_scaler.pkl"):
    try:
        # Ensure the directory exists for df_fe
        df_directory = os.path.dirname(filepath_df)
        if not os.path.exists(df_directory):
            os.makedirs(df_directory)
            print(f"Created directory: {df_directory}")

        # Ensure the directory exists for label_encoders and scaler
        model_directory = os.path.dirname(filepath_cat_cols_label_encoders)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
            print(f"Created directory: {model_directory}")


        with open(filepath_df, 'wb') as f:
            pickle.dump(df_fe, f)
        print(f"DataFrame saved successfully to: {filepath_df}")

        with open(filepath_cat_cols_label_encoders, 'wb') as f:
            pickle.dump(cat_cols_label_encoders, f)
        print(f"Label encoders saved successfully to: {filepath_cat_cols_label_encoders}")

        with open(filepath_num_fe_scaler, 'wb') as f:
            pickle.dump(num_fe_scaler, f)
        print(f"Label encoders saved successfully to: {filepath_num_fe_scaler}")

    except Exception as e:
        print(f"An error occurred while saving: {e}")


# ----------------Load df_fe and label_encoders from your Google Drive----------------------------------
def load_objects_from_drive(filepath_df="CyberThreat_Insight/cybersecurity_data/df_fe.pkl",
                            filepath_cat_cols_label_encoders="CyberThreat_Insight/model_deployment/cat_cols_label_encoders.pkl",
                            filepath_num_fe_scaler="CyberThreat_Insight/model_deployment/num_fe_scaler.pkl"):
    try:
        with open(filepath_df, 'rb') as f:
            df_fe = pickle.load(f)
        print(f"DataFrame loaded successfully from: {filepath_df}")

        with open(filepath_cat_cols_label_encoders, 'rb') as f:
            cat_cols_label_encoders = pickle.load(f)
        print(f"Label encoders loaded successfully from: {filepath_cat_cols_label_encoders}")

        with open(filepath_num_fe_scaler, 'rb') as f:
            num_fe_scaler = pickle.load(f)
        print(f"Label encoders loaded successfully from: {filepath_num_fe_scaler}")

        return df_fe, cat_cols_label_encoders, num_fe_scaler

    except Exception as e:
        print(f"An error occurred while loading: {e}")
        return None, None, None # Return None for the third value as well


#-------------Generate Synthetic Anomalies Using Cholesky-Based Perturbation-------------------

def get_files_path(
        normal_operations_file_path = "CyberThreat_Insight/cybersecurity_data/cybersecurity_dataset_combined.csv",
        combined_normal_and_anomaly_file_path = "/content/cybersecurity_dataset_combined_output_file.csv"):

    return {
        "normal_operations_file_path": normal_operations_file_path,
        "combined_normal_and_anomaly_file_path":  combined_normal_and_anomaly_file_path
    }

def load_Synthetic_dataset(filepath):
    return pd.read_csv(filepath)

def scale_data(df, features):
    scaler = StandardScaler()
    features_to_scale = [f for f in features if f != 'Timestamps']
    scaled = scaler.fit_transform(df[features_to_scale].dropna())
    return scaled, scaler

def cholesky_decomposition(scaled_data):
    cov_matrix = np.cov(scaled_data, rowvar=False)
    L = np.linalg.cholesky(cov_matrix)
    return L

def generate_cholesky_anomalies(real_data, L, num_samples=1000):
    np.random.seed(42)
    normal_samples = np.random.randn(num_samples, real_data.shape[1])
    synthetic_anomalies = normal_samples @ L.T
    return synthetic_anomalies

def inverse_transform(synthetic_data, scaler):
    return scaler.inverse_transform(synthetic_data)

def create_anomaly_df(original_data, synthetic_original, features):
    df_synthetic = pd.DataFrame(synthetic_original, columns=features)

    # Create full column DataFrame for synthetic data with same structure as original
    df_synthetic_full = pd.DataFrame(columns=original_data.columns)

    # Fill known numerical features
    for col in features:
        df_synthetic_full[col] = df_synthetic[col]

    # Fill in the rest of the columns using random sampling or generation
    for col in original_data.columns:
        if col not in features:
            if original_data[col].dtype == 'object':
                df_synthetic_full[col] = np.random.choice(original_data[col].dropna().unique(), size=len(df_synthetic_full))
            elif np.issubdtype(original_data[col].dtype, np.datetime64):
                # If timestamps exist, shift a base date with random offsets
                base = pd.to_datetime("2024-01-01")
                df_synthetic_full[col] = base + pd.to_timedelta(np.random.randint(0, 90, size=len(df_synthetic_full)), unit='D')
            else:
                df_synthetic_full[col] = np.random.choice(original_data[col].dropna(), size=len(df_synthetic_full))

    #df_synthetic_full["Threat Level"] = "Anomalous"
    df_synthetic_full["Source"] = "Synthetic"

    df_real = original_data.copy()
    df_real["Source"] = "Real"

    df_combined = pd.concat([df_real, df_synthetic_full], ignore_index=True)
    return df_combined

def save_dataset(df, path):
    df.to_csv(path, index=False)
    print(f"Saved combined dataset with synthetic anomalies to: {path}")

def data_injection_cholesky_based_perturbation(file_paths = "", save_data_true_false = True):

    print("Anomaly Injection – Cholesky-Based Perturbation...")
    if save_data_true_false == True:
        file_paths = get_files_path()
        df_real = load_Synthetic_dataset(file_paths["normal_operations_file_path"])
    else:
        df_real = load_Synthetic_dataset(file_paths)


    #df_real.info()
    #display(df_real.head())

    # Call get_column_dic here to access numerical_columns
    columns_dic = get_column_dic()
    numerical_columns = columns_dic["numerical_columns"]

    numerical_columns_for_scaling = [col for col in numerical_columns if col != "Timestamps"]

    scaled_data, scaler = scale_data(df_real, numerical_columns_for_scaling)

    L = cholesky_decomposition(scaled_data)
    synthetic_scaled = generate_cholesky_anomalies(df_real[numerical_columns_for_scaling], L, num_samples=100)
    synthetic_original = inverse_transform(synthetic_scaled, scaler)

    normal_and_combined_cholesky_based_perturbation_df = create_anomaly_df(df_real, synthetic_original, numerical_columns_for_scaling)

    #normal_and_combined_cholesky_based_perturbation_df.info()
    #display(normal_and_combined_cholesky_based_perturbation_df.head())

    if save_data_true_false == True:
        save_dataset(normal_and_combined_cholesky_based_perturbation_df, file_paths["combined_normal_and_anomaly_file_path"])

    return normal_and_combined_cholesky_based_perturbation_df


# -------------------------------Normalize numerical feature--------------------------------------
def normalize_numerical_features(df, p_numerical_columns):

    # normalized_df, scaler =  scale_data(df, p_numerical_columns)
    # return normalized_df, scaler
    scaler = MinMaxScaler()
    df[p_numerical_columns] = scaler.fit_transform(df[p_numerical_columns])
    return df, scaler # Return the DataFrame and scaler



def encode_dates(df, date_columns):
    """
    Extracts date components from specified columns in a DataFrame.

    Parameters:
      df (DataFrame): The DataFrame containing date columns.
      date_columns (list): List of date columns to extract components from.

    Returns:
      DataFrame: DataFrame with additional date component columns.
    """
    processed_df = df.copy()

    for date_col in date_columns:
        # Convert the column to datetime if it's not already
        processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')

        # Check if the column is a datetime column before applying .dt accessor
        if pd.api.types.is_datetime64_any_dtype(processed_df[date_col]):
            processed_df[f"year_{date_col}"] = processed_df[date_col].dt.year
            processed_df[f"month_{date_col}"] = processed_df[date_col].dt.month
            processed_df[f"day_{date_col}"] = processed_df[date_col].dt.day
            processed_df[f"day_of_week_{date_col}"] = processed_df[date_col].dt.dayofweek  # Monday=0, Sunday=6
            processed_df[f"day_of_year_{date_col}"] = processed_df[date_col].dt.dayofyear
        else:
            print(f"Warning: Column '{date_col}' is not a datetime column and will be skipped.")

    # Example of converting timestamps to seconds (if a timestamp column exists)
    if "Timestamps" in date_columns:
        processed_df["timestamp_seconds"] = processed_df["Timestamps"].astype(int) / 10**9

    return processed_df.drop(columns=date_columns)


def encode_categorical_columns(df, categorical_columns):
    """
    Applies label encoding to specified categorical columns in a DataFrame.

    Parameters:
      df (DataFrame): The DataFrame containing categorical columns.
      categorical_columns (list): List of columns to apply label encoding to.

    Returns:
      DataFrame, dict: DataFrame with encoded categorical columns and a dictionary of label encoders.
    """
    processed_df = df.copy()
    label_encoders = {}


    for column in categorical_columns:
        le = LabelEncoder()
        processed_df[column] = le.fit_transform(processed_df[column].astype(str))
        label_encoders[column] = le

    return processed_df, label_encoders

def decode_categorical_columns( df_to_decode, label_encoders):
    """
    Decodes label-encoded categorical columns in a DataFrame.

    Parameters:
      df_to_decode (DataFrame): The DataFrame containing label-encoded categorical columns.
      label_encoders (dict): Dictionary of LabelEncoders used for encoding, with column names as keys.
    """

    # initialize decoded data frame(decoded_df)
    decoded_df = [[]]
    processed_df_to_decode = df_to_decode.copy()

    for column, le in label_encoders.items():
        if column in processed_df_to_decode.columns:
            decoded_df[column] = le.inverse_transform(processed_df_to_decode[column])




def preprocess_dataframe(df, numerical_columns, date_columns, categorical_columns):
    """
    Main function to preprocess a DataFrame by encoding dates and categorical columns.

    Parameters:
      df (DataFrame): Original DataFrame to be copied and processed.
      numerical_columns (list): List of numerical columns (currently unused in this function).
      date_columns (list): List of date columns to extract components from.
      categorical_columns (list): List of categorical columns to encode.

    Returns:
      DataFrame, dict: Processed DataFrame and dictionary of label encoders.
    """

    #Normalize numerical feature
    #processed_df = normalize_numerical_features(df, numerical_columns)
    df, normalize_numerical_features_scaler = normalize_numerical_features(df, [i for i in numerical_columns if i not in ['Timestamps']])

    # Apply date encoding using the df
    processed_df = encode_dates(df, date_columns)  # Use the output of normalize_numerical_features

    # Apply categorical encoding using the processed_df, but exclude date_columns
    processed_df, categorical_columns_label_encoders = encode_categorical_columns(processed_df, [col for col in categorical_columns if col not in date_columns])

    return processed_df, categorical_columns_label_encoders, normalize_numerical_features_scaler # Return processed_df instead of df
#------------------------------------------------------------------------------------------------

# 1. Correlation Heatmap
def plot_correlation_heatmap(ax, df, method='pearson'):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method=method)
    sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f', square=True, ax=ax)
    ax.set_title(f'{method.capitalize()} Correlation Heatmap')

# 2. Feature Importance
def plot_feature_importance(ax, X, y, top_n=None):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_

    if top_n is None or top_n > len(importances):
        top_n = len(importances)

    indices = np.argsort(importances)[-top_n:]
    ax.barh(range(top_n), importances[indices], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([X.columns[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Random Forest Feature Importances")

    return rf

# 3. SHAP Summary Plot (Standalone, not in subplot)
# Function to set font properties for plot axes
def set_font_properties(ax, x_fontsize=8, y_fontsize=8, labelcolor='black', mean_shap_fontsize=8, font_name = 'sans-serif'):
    """
    Set the font properties for axes ticks.

    Args:
    - ax: The axes object for the plot
    - x_fontsize: Font size for x-axis labels
    - y_fontsize: Font size for y-axis labels
    - labelcolor: Color for the labels (default is 'blue')
    """
    ax.tick_params(axis='x', labelsize=x_fontsize, labelcolor=labelcolor)
    ax.tick_params(axis='y', labelsize=y_fontsize, labelcolor=labelcolor)

    # Adjust the font for the x-axis labels
    for label in ax.get_xticklabels():
        label.set_fontsize(x_fontsize)  # Set font size
        label.set_fontname(font_name)  # Default sans-serif font
        label.set_color(labelcolor)  # Set label color


    # Adjust the font for the y-axis labels
    for label in ax.get_yticklabels():

        label.set_fontsize(y_fontsize)  # Set font size
        label.set_fontname(font_name)  # Default sans-serif font
        label.set_color(labelcolor)  # Set label color

    # Adjust mean(|SHAP value|) font size (located in the text below the plot)
    for text in ax.texts:
        if 'mean(|SHAP value|)' in text.get_text():
            text.set_fontsize(mean_shap_fontsize)  # Reduce the font size for the mean(|SHAP value|) text
            text.set_fontname(font_name)  # Default sans-serif font
            text.set_color(labelcolor)  # Set label color



# Function to update plot title font
def update_title(title, fontsize=8, family='sans-serif', fontweight='normal'):
    """
    Update the title of the plot with custom font properties.

    Args:
    - title: Title of the plot
    - fontsize: Font size for the title
    - family: Font family for the title
    - fontweight: Font weight for the title
    """
    plt.title(title, fontsize=fontsize, family=family, fontweight=fontweight)

def smaller_shap_summary_plot(shap_values, X, y, plot_type="bar", plot_size=(30, 15), title="SHAP Summary Plot", class_names=None, xlabel_fontsize=8):
    """
    Generates a smaller SHAP summary plot.

    Args:
        shap_values: SHAP values (output from SHAP model explainer)
        X: The feature matrix(Sample data used for generating the SHAP plot)
        plot_type: The type of plot ("dot", "bar", etc.).
        plot_size: A tuple (width, height) specifying the plot's size in inches.
        title: Custom title for the plot (default is 'SHAP Summary Plot')
        class_names: List of class names for the legend.
        xlabel_fontsize: Font size for the x-axis label.
    """

    # If class_names are not provided, generate default ones
    if class_names is None:
        labels = sorted(list(set(y) | set(y)))
        level_mapping = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
        class_names = [level_mapping.get(label) for label in labels]

    # Adjust the figure size for the SHAP summary plot
    plt.figure(figsize=(plot_size[0]/10, plot_size[1]/10))

    shap.summary_plot(shap_values, X, plot_type=plot_type, class_names=class_names, show=False) #prevent auto showing the plot, so we can modify it.
    #shap.summary_plot(shap_values, X, plot_type=plot_type, feature_names=list(X), class_names=class_names, show=False)
    plt.tight_layout() #reduce white space around plot.

    # Access the current axes (for summary plot)
    ax = plt.gca()

    # Change font properties for feature names and axis labels
    set_font_properties(ax, mean_shap_fontsize=8) # Set the font size for the mean(|SHAP value|) text

    # Update the title with custom font
    update_title(title)

    # Adjust legend font size
    ax.legend(fontsize=8)

    # Adjust x-axis label font size
    ax.set_xlabel(ax.get_xlabel(), fontsize=xlabel_fontsize)


    plt.show() #manually show it.


def plot_shap_summary(model, X_sample, y):
    target_column="Threat Level"
    level_mapping = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    class_names = list(level_mapping.keys())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list) and len(shap_values) > 1:
        #shap.summary_plot(shap_values[1], X_sample, plot_size=(2, 2))  # Binary case
        smaller_shap_summary_plot(shap_values[1], X_sample, y, class_names=class_names)
    else:
        #shap.summary_plot(shap_values, X_sample, plot_size=(2, 2))
        # Generate summary plot with custom class names in the legend
        smaller_shap_summary_plot(shap_values, X_sample, y, class_names=class_names)

# 4. PCA Scree Plot
def plot_pca_variance(ax, X, threshold=0.95):
    pca = PCA().fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(cum_var, marker='o', linestyle='--', color='b')
    ax.axhline(y=threshold, color='r', linestyle='-')
    ax.set_title("PCA Scree Plot")
    ax.set_xlabel("Num Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.grid(True)

# 5. Main Driver Function
def run_feature_analysis(df_fe, target_column="Threat Level", corr_method="pearson"):
    print("Running Feature Analysis Pipeline...")

    df_local = df_fe.copy()

    # Encode target if needed
    if df_local[target_column].dtype == 'object':
        le = LabelEncoder()
        df_local[target_column] = le.fit_transform(df_local[target_column])

    X = df_local.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
    y = df_local[target_column]

    # Create subplots (3 panels: correlation, importance, PCA)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Plot 1: Correlation Heatmap
    plot_correlation_heatmap(axes[0], df_local, method=corr_method)

    # Plot 2: Feature Importance
    model = plot_feature_importance(axes[1], X, y, top_n=15)

    # Plot 3: PCA Scree
    plot_pca_variance(axes[2], X)

    plt.tight_layout()
    plt.show()

    # Plot 4: SHAP Summary (standalone)
    print("\nSHAP Summary Plot:")
    X_sample = shap.utils.sample(X, 200, random_state=42) if len(X) > 200 else X
    plot_shap_summary(model, X_sample, y)

    print("Feature analysis complete.")

#-------------------------features_engineering_pipeline ---------------------------
def features_engineering_pipeline(file_path = None , analysis_true_false = True):

    print("Feature engineering pipeline started.")
    #get features dic
    columns_dic = get_column_dic()
    numerical_columns = columns_dic["numerical_columns"]
    features_engineering_columns = columns_dic["features_engineering_columns"]
    initial_dates_columns = columns_dic["initial_dates_columns"]
    categorical_columns = columns_dic["categorical_columns"]

    #data injection: Anomaly Injection – Cholesky-Based Perturbation
    if analysis_true_false == True:
        naccbp_df = data_injection_cholesky_based_perturbation()
    else:
        naccbp_df = data_injection_cholesky_based_perturbation(file_path, save_data_true_false = False)

    #data collectionn, Generation and Preprocessing
    df = naccbp_df.copy()

    # Convert date columns to datetime objects
    for col in initial_dates_columns:
        df[col] = pd.to_datetime(df[col])  # Convert to datetime

    # We filter the Timestamps from the columns to apply the MinMaxScaler
    df, cat_cols_label_encoders, num_fe_scaler = preprocess_dataframe(df, numerical_columns, initial_dates_columns, categorical_columns)

    #feature analysis
    df_fe = df[features_engineering_columns].copy()
    #display(df_fe.head())

    if analysis_true_false:
        # Run feature analysis
        run_feature_analysis(df_fe, target_column="Threat Level", corr_method="pearson")

        # deploy fe_processd_df and label_encoder to google drive
        save_objects_to_drive(df_fe, cat_cols_label_encoders, num_fe_scaler)

    print("Feature engineering pipeline completed.")

    return df_fe, cat_cols_label_encoders, num_fe_scaler

if __name__ == "__main__":

    fe_processed_df, cat_cols_label_encoders, num_fe_scaler = features_engineering_pipeline()
