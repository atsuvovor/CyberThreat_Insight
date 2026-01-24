#eda.py
#CyberThreat-Insight
#Anomalous Behavior Detection in Cybersecurity Analytics using Generative AI
#  ---------------------Model Development---------------------
#Toronto, Septeber 08 2025
#Autor : Atsu Vovor

#Master of Management in Artificial Intelligence
#Consultant Data Analytics Specialist | Machine Learning |
#Data science | Quantitative Analysis | AI REporting | French & English Bilingual
#--------------------------------------------

#----------------------------------------------
#Exploratory Data Analysis (EDA) Code
# --------------------------
#   Necessary Imports
# --------------------------
import os
import sys
import subprocess

# ---- Helper: Safe import with auto-install ----
def safe_import(pkg, import_name=None, pip_name=None):
    """
    Try to import a package, install it if missing, and then re-import.
    :param pkg: module name for import
    :param import_name: optional alias if import name differs from pip package
    :param pip_name: package name for pip install (defaults to pkg)
    """
    import importlib
    if import_name is None:
        import_name = pkg
    if pip_name is None:
        pip_name = pkg

    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"⚠️ {import_name} not found. Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return importlib.import_module(import_name)

                                       
np = safe_import("numpy")
pd = safe_import("pandas")
faiss = safe_import("faiss", pip_name="faiss-cpu")
fuzz = safe_import("rapidfuzz")
warnings = safe_import("warnings")
json = safe_import("json")
re = safe_import("re")

from datetime import datetime, timedelta
from IPython.display import display
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path




import io
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display
#import json
#import faiss
from typing import List, Dict, Optional, Tuple, Any
from google.colab import userdata
from google.colab.userdata import SecretNotFoundError
#-----
import re
from rapidfuzz import fuzz
import warnings

# ---- Safe FAISS import ----
try:
    import faiss
except ImportError:
    print("⚠️ faiss not found. Installing faiss-cpu...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    import faiss
    
# Ignore specific warnings from libraries
warnings.filterwarnings('ignore')


def coerce_safe_dtypes(df):
    safe_casts = {
        "Login Attempts": "float64",
        "Memory Usage MB": "float64",
        "Impact Score": "int64"
    }

    for col, target_type in safe_casts.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(target_type)
            except Exception:
                pass  # Leave as-is if unsafe

    return df


# Function to normalize numerical features using Min-Max Scaling
def normalize_numerical_features(p_df):
    scaler = MinMaxScaler()
    p_df_daily = p_df.copy()
    # Apply Min-Max scaling to the DataFrame
    df_normalized = pd.DataFrame(scaler.fit_transform(p_df_daily), columns=p_df_daily.columns.to_list(), index=p_df_daily.index)
    return df_normalized


#------------------------------------------------------------------
# Function to plot daily values of numerical features
def plot_numerical_features_daily_values(df, date_column, feature_columns, rows, cols):

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

    # Iterate through features and plot each one
    for i, column in enumerate(feature_columns):
        ax = axes[i]
        ax.plot(df.index, df[column], marker='o', label=column, color='b')
        ax.set_title(column, fontsize=10)
        ax.set_xlabel("Date Reported", fontsize=8)
        ax.set_ylabel(column, fontsize=8)
        ax.grid(True)
        ax.legend(fontsize=8)

        # Format x-axis to prevent overlapping dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=100))  # Show every 100 days
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    # Hide any unused subplots
    for j in range(len(feature_columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------

# Pipeline to plot daily distribution of numerical features (both non-normalized and normalized)
def daily_distribution_of_activity_features_pipeline(df):
    """
    Pipeline to plot daily distribution of numerical features.
    """
    features = df.columns.tolist()
    n_features = len(features)
    rows = int(n_features/4)
    cols =  int(n_features/2)

    print("Non normalized daily distribution")
    plot_numerical_features_daily_values(df, "Date Reported", features, rows, cols)
    #plot_numerical_features_daily_values(df)
    print("Normalized daily distribution")
    df_normalized = normalize_numerical_features(df)
    #plot_numerical_features_daily_values(df_normalized)
    plot_numerical_features_daily_values(df_normalized, "Date Reported", features, rows, cols)
#-------------------------------------------------------------------------

# Function to plot histograms for features
def plot_histograms(df):
    """
    Plots histograms for all features in the list with risk level and displays basic statistics.
    """
    # Define the risk palette for categorical features
    risk_palette = {
                    'Low': 'green',
                    'Medium': 'yellow',
                    'High': 'orange',
                    'Critical': 'red'
                   }

    features = df.columns.tolist()
    n_features = len(features)
    n_cols = int(n_features/2)
    n_rows = int((n_features + n_cols - 1) // n_cols)  # Calculate rows needed for the grid

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, n_rows * 6))  # Dynamically adjust figure size
    axes = np.array(axes)  # Ensure `axes` is always an array
    axes = axes.flatten()  # Flatten to handle indexing consistently

    # Iterate through features and plot histograms
    for i, feature in enumerate(features):
        # Check if the feature is categorical with risk levels or numerical
        if df[feature].dtype == 'object' and set(df[feature].unique()).issubset(risk_palette.keys()):
            sns.histplot(df[feature], palette=risk_palette, ax=axes[i])
        else:
            sns.histplot(df[feature], bins=30, kde=True, ax=axes[i])

        axes[i].set_title(f'Histogram of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')

        # Calculate and display statistics for numeric features
        if np.issubdtype(df[feature].dtype, np.number):
            mean_return = df[feature].mean()
            std_dev = df[feature].std()
            skewness = df[feature].skew()
            kurtosis = df[feature].kurtosis()

            # Display statistics on the plot
            statistics = (f"Mean: {mean_return:.4f}\n"
                      f"Std Dev: {std_dev:.4f}\n"
                      f"Skewness: {skewness:.4f}\n"
                      f"Kurtosis: {kurtosis:.4f}")
            axes[i].text(0.35, -0.18, statistics, transform=axes[i].transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))


    # Hide any unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Add padding to the bottom to make space for statistics
    plt.show()

# Function to plot boxplots for features
def plot_boxplots(df):
    """
    Plots boxplots for all features in the list and displays basic statistics.
    """
    # Define the risk palette for categorical features
    risk_palette = {
                    'Low': 'green',
                    'Medium': 'yellow',
                    'High': 'orange',
                    'Critical': 'red'
                   }

    features  = df.columns.tolist()
    n_features = len(features)
    n_cols = int(n_features/2)
    n_rows = int((n_features + n_cols - 1) // n_cols)  # Calculate rows needed for the grid

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, n_rows * 6))  # Dynamically adjust figure size
    axes = np.array(axes)  # Ensure `axes` is always an array
    axes = axes.flatten()  # Flatten to handle indexing consistently

    # Iterate through features and plot boxplots
    for i, feature in enumerate(features):
        # Check if the feature is categorical with risk levels or numerical
        if df[feature].dtype == 'object' and set(df[feature].unique()).issubset(risk_palette.keys()):
            sns.boxplot(y=df[feature], palette=risk_palette, ax=axes[i])
        else:
            sns.boxplot(y=df[feature], ax=axes[i])

        axes[i].set_title(f'Boxplot of {feature}')
        axes[i].set_ylabel(feature)

        # Calculate and display statistics for numeric features
        if np.issubdtype(df[feature].dtype, np.number):
            mean_return = df[feature].mean()
            std_dev = df[feature].std()
            skewness = df[feature].skew()
            kurtosis = df[feature].kurtosis()

            # Add statistics below the plot
            statistics = (f"Mean: {mean_return:.4f}\n"
                      f"Std Dev: {std_dev:.4f}\n"
                      f"Skewness: {skewness:.4f}\n"
                      f"Kurtosis: {kurtosis:.4f}")
            axes[i].text(0.35, -0.18, statistics, transform=axes[i].transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))

    # Hide any unused subplots
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    #plt.tight_layout()
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Add padding to the bottom to make space for statistics
    plt.show()
#-----------------------------------------------------------------------------------------------------

# Master function to visualize the distribution of activity features (histograms and boxplots)
def visualize_form_of_activity_features_distribution(df):
    """
    Master function to plot histograms and boxplots for all features, with statistics.
    """
    sns.set(style="whitegrid") # Set the style for the plots
    print("Plotting histograms...")
    plot_histograms(df)

    print("Plotting boxplots...")
    plot_boxplots(df)

# Function to create a scatter plot on a specified axis
def plot_scatter(axes, x, y, hue, df, palette, title, xlabel, ylabel, legend_title, ax_index):
    """
    Creates a scatter plot on the specified axis.
    """
    sns.scatterplot(x=x, y=y, hue=hue, data=df, palette=palette, ax=axes[ax_index])
    axes[ax_index].set_title(title)
    axes[ax_index].set_xlabel(xlabel)
    axes[ax_index].set_ylabel(ylabel)
    axes[ax_index].legend(title=legend_title)

# Function to create a correlation heatmap for selected features
def plot_correlation_heatmap(axes, df, features, ax_index):
    """
    Creates a heatmap showing the correlation between selected features.
    """
    # Select only numerical features for correlation calculation
    numeric_features = df[features].select_dtypes(include=['number'])

     # Calculate the correlation matrix
    corr_matrix = numeric_features.corr()

    # Plot the heatmap with annotations
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[ax_index])
    axes[ax_index].set_title("Correlation Heatmap of Numerical Features")


# Function to combine scatter plots and a heatmap into a single figure
def combines_user_activities_scatter_plots_and_heatmap(scatter_df, df):
    """
    Combines scatter plots and heatmap into a single figure using subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # Create subplots (1 row, 3 columns)

    # Plot 1: Session Duration vs Data Transfer
    plot_scatter(
        axes=axes,
        x="Session Duration in Second",
        y="Data Transfer MB",
        hue="User Location",
        df=scatter_df,
        palette="Set1",
        title="Session Duration vs Data Transfer (MB) by Location",
        xlabel="Session Duration (seconds)",
        ylabel="Data Transfer (MB)",
        legend_title="User Location",
        ax_index=0
    )

    # Plot 2: Login Attempts vs Data Transfer
    plot_scatter(
        axes=axes,
        x="Login Attempts",
        y="Data Transfer MB",
        hue="User Location",
        df=scatter_df,
        palette="Set2",
        title="Login Attempts vs Data Transfer (MB) by Location",
        xlabel="Login Attempts",
        ylabel="Data Transfer (MB)",
        legend_title="User Location",
        ax_index=1
    )

    # Plot 3: Correlation Heatmap
    plot_correlation_heatmap(
        axes=axes,
        df=df,
        features=df.columns,
        ax_index=2
    )

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

#-------------------
# Main EDA pipeline
#-------------------
# Main function to execute the exploratory data analysis pipeline
def explaratory_data_analysis_pipeline(df):

    # Define the list of features for EDA
    eda_features =  [
    "Date Reported", "Issue Response Time Days", "Impact Score", "Cost",
    "Session Duration in Second", "Num Files Accessed", "Login Attempts",
    "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score"
    ]

    # Define the list of activity features for visualization
    activity_features = [
    "Risk Level", "Threat Level", "Issue Response Time Days", "Impact Score", "Cost",
    "Session Duration in Second", "Num Files Accessed", "Login Attempts",
    "Data Transfer MB", "CPU Usage %", "Memory Usage MB", "Threat Score"
    ]
    # Load the dataset from the specified file path
    #df = pd.read_csv(file_path_to_normal_and_anomalous_google_drive)

    # Define the reporting frequency for temporal analysis
    reporting_frequency = 'Quarter'
    frequency = reporting_frequency[0].upper()
    if reporting_frequency.capitalize() == 'Month' or reporting_frequency.capitalize() == 'Quarter':
        frequency_date_column = reporting_frequency.capitalize() + '_Year'

    # Create a copy of the DataFrame with selected EDA features and set 'Date Reported' as index
    eda_features_df = df[eda_features].copy()
    eda_features_df = eda_features_df.set_index("Date Reported")

    # Create a copy for frequency-based analysis
    freq_eda_features_df = eda_features_df.copy()

    # Convert index to datetime and then to the specified frequency
    freq_eda_features_df.index = pd.to_datetime(freq_eda_features_df.index)
    freq_eda_features_df[frequency_date_column] =  freq_eda_features_df.index.to_period(frequency)

    # Group by frequency and calculate the mean
    freq_eda_features_df = freq_eda_features_df.groupby(frequency_date_column).mean()
    # Convert the index back to timestamp for plotting
    freq_eda_features_df.index = freq_eda_features_df.index.to_timestamp()
    display(freq_eda_features_df)


    # Create a copy of the DataFrame with selected activity features
    activity_features_df = df[activity_features].copy()

    # Create a DataFrame with features for scatter plots
    scatter_plot_features_df = df[["Session Duration in Second", "Login Attempts",
                                  "Data Transfer MB", "User Location"]].copy()

    # Run the daily distribution pipeline for frequency-based data
    daily_distribution_of_activity_features_pipeline(freq_eda_features_df )
    # Visualize the distribution of activity features (histograms and boxplots)
    visualize_form_of_activity_features_distribution(activity_features_df)
    # Combine and display scatter plots and the correlation heatmap
    combines_user_activities_scatter_plots_and_heatmap(scatter_plot_features_df, activity_features_df)
    return freq_eda_features_df

# ---------------------------------------------
# AI Validator + RAG Q&A Chatbot
# ---------------------------------------------

# Optional Colab/GDrive imports (safe if not available)
try:
    from google.colab import drive, files, userdata  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# SentenceTransformer for embeddings
from sentence_transformers import SentenceTransformer

# Try to detect OpenAI SDK style
NEW_OPENAI_API = False
try:
    from openai import OpenAI  # >= 1.0.0
    NEW_OPENAI_API = True
except Exception:
    pass

try:
    import openai  # == 0.28 style
except Exception:
    openai = None

from transformers import pipeline
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

# --------------------------
#   Expected Schema
# --------------------------
EXPECTED_SCHEMA: Dict[str, str] = {
    "Issue Volume": "int64",
    "Impact Score": "int64",
    "Threat Level": "object",
    "Cost": "float64",
    "Department Affected": "object",
    "Num Files Accessed": "int64",
    "IP Location": "object",
    "CPU Usage %": "float64",
    "Status": "object",
    "Issue Name": "object",
    "Issue ID": "object",
    "User Location": "object",
    "Activity Type": "object",
    "Timestamps": "datetime64[ns]",
    "Category": "object",
    "Date Resolved": "datetime64[ns]",
    "Memory Usage MB": "float64",
    "Date Reported": "datetime64[ns]",
    "Reporters": "object",
    "Threat Score": "float64",
    "Issue Response Time Days": "int64",
    "Defense Action": "object",
    "Remediation Steps": "object",
    #"Is Anomaly": "int64",
    "Assignees": "object",
    "Severity": "object",
    "Session Duration in Second": "int64",
    "KPI/KRI": "object",
    "Risk Level": "object",
    "Data Transfer MB": "float64",
    "User ID": "object",
    "Issue Key": "object",
    "Color": "object",
    "Login Attempts": "float64"
}


# -----------------------------------
# Knowledge Base + Sample Questions
# -----------------------------------

DEFAULT_KB: List[str] = [
    "Common data quality issues in cybersecurity datasets include missing values in critical fields like IP addresses, timestamps, or user IDs.",
    "Unexpected data types for security event codes or categorical features can indicate data ingestion problems.",
    "Duplicate entries for unique identifiers such as issue IDs or user IDs can skew analysis and model training.",
    "Anomalies in numerical features like login attempts, session duration, data transfer volume, or CPU/memory usage can signal malicious activity.",
    "Schema validation is crucial to ensure that the dataset structure (columns and data types) matches the expected format.",
    "Missing values in numerical columns might require imputation or removal depending on the analysis context.",
    "Outliers in numerical features can significantly impact statistical measures and model performance.",
    "Inconsistent data formats for dates and timestamps need to be handled to enable time-series analysis.",
    "Categorical features with a high cardinality or inconsistent values require careful cleaning and encoding.",
    "Ensuring data integrity involves checking for inconsistencies and relationships between different columns.",
    "The dataset must contain exactly 33 columns.",
    "Each column should have the correct data type as defined in the schema.",
    "There should be no duplicate rows in the dataset.",
    "Missing values must be identified and reported.",
    "Categorical columns should be validated for consistency.",
    "Numeric columns should be checked for outliers.",
    "Data validation includes checking column names, types, nulls, duplicates, and ranges."
]

#======================================
# EDA Insights Request
#======================================
eda_format_insights_request_dic = {

        "charts_meta": [
            {"title":"Threat Score by Department",
            "summary":"Top 3 depts by avg threat score are Finance, IT, and R&D with notable spikes in April.",
            "metric":"Threat Score"},
            {"title":"Cost over Time",
            "summary":"Cumulative cost accelerates in Q2 with two extreme outliers.",
            "metric":"Cost"}
            ]
}

# =============================================
# Knowledge Base Doc Generators (HISTORICAL ONLY)
# =============================================

def create_flat_rag_docs(df: pd.DataFrame, chunk_size: int = 20) -> List[str]:
    cost_cols = ["Cost"]
    date_cols = ["Date Reported", "Date Resolved", "Timestamps"]
    numeric_cols = [
        "Session Duration in Second", "Num Files Accessed", "Login Attempts",
        "Data Transfer MB", "Memory Usage MB", "Threat Score"
    ]
    percent_cols = ["CPU Usage %"]
    defense_col = "Defense Action"
    dept_col = "Department Affected"

    all_columns = set(EXPECTED_SCHEMA.keys())
    rag_docs = []

    # Normalize columns present
    use_cols = [c for c in all_columns if c in df.columns]

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        lines = []
        for _, row in chunk.iterrows():
            row_vals = []
            for col in use_cols:
                val = row.get(col, None)
                if col in cost_cols and pd.notna(val):
                    row_vals.append(f"{col}: ${float(val):,.2f}")
                elif col in numeric_cols and pd.notna(val):
                    row_vals.append(f"{col}: {val}")
                elif col in percent_cols and pd.notna(val):
                    row_vals.append(f"{col}: {val}%")
                elif col in date_cols and pd.notna(val):
                    row_vals.append(f"{col}: {pd.to_datetime(val, errors='coerce')}")
                elif col == defense_col:
                    row_vals.append(f"{col}: {val if pd.notna(val) else 'General'}")
                elif col == dept_col:
                    row_vals.append(f"{col}: {val if pd.notna(val) else 'None'}")
                else:
                    if pd.notna(val):
                        row_vals.append(f"{col}: {val}")
            lines.append(", ".join(row_vals))
        if lines:
            rag_docs.append("\n".join(lines))
    return rag_docs

def create_grouped_rag_docs(df: pd.DataFrame, group_by: str = "Department Affected", top_k: int = 3) -> List[str]:
    numerical_columns = [
        "Session Duration in Second", "Num Files Accessed", "Login Attempts",
        "Data Transfer MB", "Memory Usage MB", "Threat Score"
    ]
    percentage_columns = ["CPU Usage %"]
    cost_column = "Cost"
    date_columns = ["Date Reported", "Date Resolved", "Timestamps"]
    categorical_columns = [
        "Severity", "Category", "Status", "KPI/KRI", "Activity Type",
        "Threat Level", "Attack Type", "Phase"
    ]

    if group_by not in df.columns:
        group_by = "Department Affected"

    rag_docs = []
    grouped = df.groupby(group_by, dropna=False)

    for group_name, group_df in grouped:
        doc = [f"{group_by}: {group_name}"]
        doc.append(f"- Total Issues: {len(group_df)}")

        if "Threat Level" in group_df.columns:
            critical_issues = group_df[group_df["Threat Level"].astype(str).str.lower() == "critical"]
            doc.append(f"- Critical Issues: {len(critical_issues)}")

        if cost_column in group_df.columns:
            total_cost = pd.to_numeric(group_df[cost_column], errors="coerce").fillna(0).sum()
            avg_cost = pd.to_numeric(group_df[cost_column], errors="coerce").fillna(0).mean()
            min_cost = pd.to_numeric(group_df[cost_column], errors="coerce").fillna(0).min()
            max_cost = pd.to_numeric(group_df[cost_column], errors="coerce").fillna(0).max()
            doc.append(f"- Total Cost: ${total_cost:,.2f}")
            doc.append(f"- Cost (Avg): ${avg_cost:,.2f}, Min: ${min_cost:,.2f}, Max: ${max_cost:,.2f}")

        if "Defense Action" in group_df.columns:
            defense_counts = group_df["Defense Action"].dropna().value_counts().to_dict()
            doc.append(f"- Defense Actions: {defense_counts}")

        for col in numerical_columns:
            if col in group_df.columns:
                s = pd.to_numeric(group_df[col], errors="coerce").dropna()
                if len(s):
                    doc.append(f"- {col}: Total={s.sum():.2f}, Avg={s.mean():.2f}, Min={s.min():.2f}, Max={s.max():.2f}")

        for col in percentage_columns:
            if col in group_df.columns:
                s = pd.to_numeric(group_df[col], errors="coerce").dropna()
                if len(s):
                    doc.append(f"- {col}: Avg={s.mean():.2f}%, Max={s.max():.2f}%")

        for col in date_columns:
            if col in group_df.columns:
                dt = pd.to_datetime(group_df[col], errors="coerce").dropna()
                if len(dt):
                    doc.append(f"- {col}: Earliest={dt.min()}, Latest={dt.max()}")

        for col in categorical_columns:
            if col in group_df.columns:
                counts = group_df[col].astype(str).value_counts().nlargest(top_k).to_dict()
                doc.append(f"- {col} Top-{top_k}: {counts}")

        rag_docs.append("\n".join(doc))

    return rag_docs

def prepare_rag_documents(simulated_df: pd.DataFrame = None,
                          historical_df: pd.DataFrame = None,
                          group_by: str = None,
                          chunk_size: int = 20,
                          use_grouped: bool = True,
                          DEFAULT_KB = DEFAULT_KB) -> List[str]: # Add use_grouped as a parameter here
    """
    HISTORICAL-ONLY per requirement:
    - Do NOT combine with simulated data.
    - 'combined_df' explicitly set to None (unused).
    """

    combined_df = None  # <-- per instruction
    if historical_df is None or len(historical_df) == 0:
        # return at least base kb so downstream code doesn't fail
        return DEFAULT_KB

    group_by_rag_docs = create_grouped_rag_docs(historical_df, group_by=group_by)
    flat_rag_docs = create_flat_rag_docs(historical_df, chunk_size=chunk_size)

    # Merge into one knowledge base
    return  DEFAULT_KB + group_by_rag_docs + flat_rag_docs


# Merge into one knowledge base
#extended_knowledge = DEFAULT_KB + sample_questions
# =============================================
# RAG Vector Store
# =============================================
class RAGStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.docs: List[str] = []

    def fit(self, docs: List[str]):
        self.docs = docs or []
        if not self.docs:
            self.index = None
            return self
        embs = self.embedder.encode(self.docs, convert_to_numpy=True)
        # ensure float32 for faiss
        embs = np.array(embs).astype('float32')
        dim = embs.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embs)
        return self

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float, str]]:
        if not self.docs or self.index is None:
            return []
        q = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        D, I = self.index.search(q, min(k, len(self.docs)))
        return [(int(i), float(d), self.docs[int(i)]) for i, d in zip(I[0], D[0])]


# =============================================
# Anti-repetition helper
# =============================================
def split_sentences(text: str):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def remove_repetitive_sentences(text: str, threshold: int = 85) -> str:
    sents = split_sentences(text)
    out = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if any(fuzz.ratio(s.lower(), kept.lower()) >= threshold for kept in out):
            continue
        out.append(s)
    return " ".join(out)

# -------------------------------------------------
# Utility: build a flexible LLM client
# -------------------------------------------------

def build_llm_client():
    # 1. OpenAI (priority if available)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("data_analytics_Key")
    if api_key:
        if NEW_OPENAI_API:
            print("Attempting to use OpenAI API...")
            try:
                client = OpenAI(api_key=api_key)
                # Quick check if the client is valid
                client.models.list()
                print("✅ OpenAI API client initialized.")
                return client, "openai-new"
            except Exception as e:
                print(f"⚠️ OpenAI API client init failed: {e}")
                print("Falling back to other options...")

        elif openai is not None:
            print("Attempting to use legacy OpenAI API...")
            try:
                openai.api_key = api_key
                # Quick check if the client is valid
                openai.Model.list()
                print("✅ Legacy OpenAI API client initialized.")
                return openai, "openai-old"
            except Exception as e:
                print(f"⚠️ Legacy OpenAI API client init failed: {e}")
                print("Falling back to other options...")


    # 2. Hugging Face Inference API
    hf_key = None
    if IN_COLAB:
        try:
            hf_key = userdata.get("data_analytics_hg_key")
            print("Attempting to retrieve Hugging Face API key from Colab secrets...")
        except SecretNotFoundError:
            hf_key = None

    if not hf_key:
        hf_key = os.environ.get("data_analytics_hg_key")
        if hf_key:
            print("Attempting to retrieve Hugging Face API key from environment variable...")

    if hf_key and InferenceClient:
        print("Attempting to use Hugging Face Inference API...")
        try:
            client = InferenceClient(token=hf_key)
            # Quick check if the client is valid (e.g., list models or a simple inference)
            # This might be model-dependent and tricky without a specific model endpoint
            # A simpler check might be to just return the client and handle errors on first call
            print("✅ Hugging Face Inference API client initialized.")
            return client, "huggingface-api"
        except Exception as e:
            print(f"⚠️ Hugging Face Inference API client init failed: {e}")
            print("Falling back to local pipeline...")

    # 3. Local offline Hugging Face pipeline
    print("Using local offline Hugging Face pipeline (distilgpt2)...")
    print("⚠️ Note: The local 'distilgpt2' model has a small context window and may struggle with long inputs.")
    try:
        local_pipe = pipeline("text-generation", model="distilgpt2")
        print("✅ Local Hugging Face pipeline initialized.")
        return local_pipe, "huggingface-local"
    except Exception as e:
        print(f"❌ Failed to initialize local Hugging Face pipeline: {e}")
        print("LLM client could not be built.")
        return None, None


# --------------------------
#     AI Agent
# --------------------------
class AIValidatorAgent:
    def __init__(
        self,
        expected_schema: Dict[str, str],
        knowledge_base: Optional[List[str]] = None,
        model_name: str = "gpt-3.5-turbo",
        top_k: int = 3
    ):
        # schema + kb
        self.expected_schema = expected_schema
        self.knowledge_base = knowledge_base
        # embeddings + FAISS (for quick KB retrieval)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        # Build a RAG store attached to the agent (so other methods can call self.rag.search)
        self.rag = RAGStore(model_name="all-MiniLM-L6-v2")
        self.rag.fit(self.knowledge_base)

        # openai client
        self.client, self.sdk_mode = build_llm_client()
        self.model_name = model_name

        # outputs
        self.validation_report: Dict[str, any] = {}
        self.initial_summary: str = ""
        self.refined_summary: Optional[str] = None

    # --------------------------
    # Data loading helpers
    # -------------------------
    def load_data(self, choice: str = None, file_path: str = None, url: str = None) -> pd.DataFrame:
        """
        Entry point to load dataset.
        Shows menu (if no choice passed) and delegates to source-specific loaders.
        """
        if choice is None:
            # fallback to CLI input (non-UI use)
            print("📂 Choose data source:")
            print("1. Google Drive (default)")
            print("2. Upload local file")
            print("3. Load from URL")
            choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            return self.load_from_drive()
        elif choice == "2":
            return self.load_from_upload(file_path=file_path)
        elif choice == "3":
            return self.load_from_url(url=url)
        else:
            raise ValueError("Invalid choice. Use 1/2/3.")

    # --------------------------
    # Source-specific loaders
    # --------------------------

    from pathlib import Path

    def resolve_repo_data_path(self, folder: str, file_name: str) -> Path:
        """
        Resolve a data file path robustly:
          - If 'folder' is absolute, use it.
          - If 'folder' is relative, interpret it relative to the repo root
            (assumes this file lives in repo/.../explanatory_data_analysis/eda.py).
          - If 'folder' starts with the repo name (e.g. "CyberThreat_Insight/..."),
            strip the leading repo name to avoid duplication.
        Returns an absolute Path.
        """
        repo_root = Path(__file__).resolve().parents[1]  # parent of explanatory_data_analysis
        folder_path = Path(folder)

        if folder_path.is_absolute():
            final_folder = folder_path
        else:
            parts = list(folder_path.parts)
            # strip leading repo folder if present (avoid duplication)
            if parts and parts[0] == repo_root.name:
                parts = parts[1:]
            final_folder = repo_root.joinpath(*parts) if parts else repo_root

        return (final_folder / file_name).resolve()

    
    def load_from_drive(self) -> pd.DataFrame:
        if not IN_COLAB:
            raise RuntimeError("Google Drive option requires Google Colab.")
        #from google.colab import drive
        #drive.mount('/content/drive')
        #folder = "/content/drive/My Drive/Cybersecurity Data"
        folder = "CyberThreat_Insight/cybersecurity_data"
        #file_name = "normal_and_anomalous_cybersecurity_dataset_for_google_drive_kb2.csv"
        file_name = "cybersecurity_dataset_combined.csv"
        path = self.resolve_repo_data_path(folder, file_name)
        if not path.exists():
            raise FileNotFoundError(f"❌ File not found in: {path}")
        df = pd.read_csv(path)
        #path = os.path.join(folder, file_name)
        #if not os.path.exists(path):
        #    raise FileNotFoundError(f"❌ File not found in: {path}")
        #df = pd.read_csv(path)
        return self._postprocess(df)

    def load_from_upload(self, file_path: str = None) -> pd.DataFrame:
        if IN_COLAB and file_path is None:
            from google.colab import files
            up = files.upload()
            fn = next(iter(up.keys()))  # take first uploaded file
            df = pd.read_csv(io.BytesIO(up[fn]))
            print("✅ Local file uploaded successfully.")
        else:
            if file_path is None:
                file_path = input("Enter local CSV path: ").strip()
            df = pd.read_csv(file_path)
            print(f"✅ Loaded: {file_path}")
        return self._postprocess(df)

    def load_from_url(self, url: str = None) -> pd.DataFrame:
        import requests
        if url is None:
            url = input("Enter file URL: ").strip()
        resp = requests.get(url)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        print(f"✅ File loaded from URL: {url}")
        return self._postprocess(df)

    # ----------------------
    # Shared post-processing
    # ----------------------
    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert datetime columns if present
        for col in ["Timestamps", "Date Resolved", "Date Reported"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        print("\n📊 Data Loaded. Shape:", df.shape)
        return df


    # --------------------------
    # Validation
    # --------------------------

    def split_sentences(self, text: str):
        """
        Simple regex-based sentence splitter.
        Splits on ., ?, ! followed by space or end of string.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]

    def remove_repetitive_sentences(self, text: str, similarity_threshold: int = 85) -> str:
        """
        Removes repetitive or similar sentences from a paragraph.
        """
        sentences = self.split_sentences(text)
        unique_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check similarity against existing sentences
            is_duplicate = False
            for kept in unique_sentences:
                if fuzz.ratio(sentence.lower(), kept.lower()) >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_sentences.append(sentence)

        return " \n".join(unique_sentences)

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        report: Dict[str, any] = {}

        expected_cols = set(self.expected_schema.keys())
        actual_cols = set(df.columns)

        missing_cols = sorted(list(expected_cols - actual_cols))
        extra_cols = sorted(list(actual_cols - expected_cols))
        report["shape"] = df.shape
        report["missing_columns"] = missing_cols
        report["extra_columns"] = extra_cols

        # dtype mismatches
        type_mismatches = {}
        for col in expected_cols & actual_cols:
            actual_dtype = self._dtype_string(df[col])
            expected_dtype = str(self.expected_schema[col])
            if actual_dtype != expected_dtype:
                type_mismatches[col] = (actual_dtype, expected_dtype)
        report["type_mismatches"] = type_mismatches

        # missing values
        mv = df.isna().sum()
        report["missing_values"] = mv[mv > 0].to_dict()

        # duplicates
        report["duplicates"] = int(df.duplicated().sum())

        # high-level schema check block (for your original printout)
        report["schema_check"] = {
            "missing_columns": missing_cols,
            "extra_columns": extra_cols
        }

        self.validation_report = report
        self.initial_summary = self._compose_initial_summary(report)

        # refine with LLM (if available), using retrieved knowledge(this is generatig allucination)
        #self.refined_summary = self.refine_summary_with_llm(
        #    initial_summary=self.initial_summary,
        #    validation_report=report
        #)

        if self.sdk_mode == "huggingface-local":
            self.refined_summary = self.initial_summary
        else:
            self.refined_summary = self.refine_summary_with_llm(
            initial_summary=self.initial_summary,
            validation_report=report
            )

        return report

    #def validation_errors(self, report):
    #    # align names with validate_dataset output
    #    type_mismatches = report.get("type_mismatches")
    #    missing_columns = report.get("missing_columns")
    #    extra_cols = report.get("extra_cols")
    #    null_values = report.get("missing_values")
    #    duplicate_rows = report.get("duplicates")

    #    validation_errors = []
    #    if type_mismatches:
    #        validation_errors.append("Type mismatches: " + str(type_mismatches))
    #    if missing_columns:
    #        validation_errors.append("Missing columns: " + str(missing_columns))
    #    if extra_cols:
    #        validation_errors.append("Extra columns: " + str(extra_cols))
    #    if null_values:
    #        validation_errors.append("Null values: " + str(null_values))
    #    if duplicate_rows:
    #        validation_errors.append("Duplicate rows: " + str(duplicate_rows))
    #    if validation_errors:
    #        return "\n".join(validation_errors)
    #    # return None if no errors
    #    return None

    def validation_errors(self, report):
        fatal_errors = []
        # Only fail on structural problems
    
        if report.get("missing_columns"):
            fatal_errors.append(f"Missing columns: {report['missing_columns']}")

        if report.get("extra_columns"):
            fatal_errors.append(f"Extra columns: {report['extra_columns']}")

        if report.get("missing_values"):
            fatal_errors.append(f"Null values: {report['missing_values']}")

        if report.get("duplicates", 0) > 0:
            fatal_errors.append(f"Duplicate rows: {report['duplicates']}")

        # Type mismatches are NOT fatal if coercible
        return "\n".join(fatal_errors) if fatal_errors else None

    
    def _dtype_string(self, s: pd.Series) -> str:
        """Normalize dtype str for comparison (pandas often returns e.g., 'int64')."""
        return str(s.dtype)

    def _compose_initial_summary(self, report: Dict[str, any]) -> str:
        parts = []
        shape = report.get("shape")
        if shape:
            parts.append(f"The dataset has {shape[0]} rows and {shape[1]} columns.\n")

        # missing values
        if report.get("missing_values"):
            cols = list(report["missing_values"].keys())
            parts.append(f"Some columns have missing values: {cols}.\n")
        else:
            parts.append("No missing values detected.\n")

        # duplicates
        if report.get("duplicates", 0) > 0:
            parts.append(f"There are {report['duplicates']} duplicate rows.\n")
        else:
            parts.append("No duplicate rows found.\n")

        # type mismatches
        if report.get("type_mismatches"):
            mismatches = [
                f"{col} (found {dtype_pair[0]}, expected {dtype_pair[1]})"
                for col, dtype_pair in report["type_mismatches"].items()
            ]
            parts.append("Type mismatches:\n      " + ";\n      ".join(mismatches) + "\n")

        # schema check
        sc = report.get("schema_check", {})
        if sc:
            if sc.get("missing_columns"):
                parts.append(f"Missing expected columns: {sc['missing_columns']}\n")
            if sc.get("extra_columns"):
                parts.append(f"Extra unexpected columns: {sc['extra_columns']}.\n")
            if not sc.get("missing_columns") and not sc.get("extra_columns"):
                parts.append("Dataset schema matches the expected structure.\n")

        # positive “all good” statement if fully clean
        if (
            not report.get("missing_values") and
            report.get("duplicates", 0) == 0 and
            not report.get("type_mismatches") and
            not report.get("missing_columns") and
            not report.get("extra_columns")
        ):
            parts.append(
                "All validation checks passed: schema matches, dtypes align, "
                "no missing values, and no duplicates."
            )

        return "".join(parts)

    # --------------------------
    # LLM helpers
    # --------------------------
    def _chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if self.client is None:
            print("LLM client is not initialized.")
            return None

        # Determine max length based on SDK mode if possible, default to a conservative value
        max_length = 1024 # Conservative default for local models
        if self.sdk_mode and self.sdk_mode.startswith("openai"):
             max_length = 4096 # Example for some OpenAI models, adjust based on actual model
        # Add checks for other SDK modes and specific models if needed

        truncated_messages = []
        for message in messages:
            content = message["content"]
            if len(content) > max_length:
                # More sophisticated truncation might involve summarizing or selecting key parts
                # For now, simple truncation: keep the beginning and end
                truncated_content = content[:max_length//2] + "\n... [content truncated] ...\n" + content[-max_length//2:]
                truncated_messages.append({"role": message["role"], "content": truncated_content})
            else:
                truncated_messages.append(message)


        try:
            if isinstance(self.sdk_mode, str) and self.sdk_mode.startswith("openai"):
                # OpenAI style
                if self.sdk_mode == "openai-new":
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=truncated_messages,
                        max_tokens=800,
                        temperature=0.4,
                        top_p=0.95
                    )
                    return resp.choices[0].message.content.strip()
                else:
                    resp = self.client.ChatCompletion.create(
                        model=self.model_name,
                        messages=truncated_messages,
                        max_tokens=800,
                        temperature=0.4,
                        top_p=0.95
                    )
                    return resp["choices"][0]["message"]["content"].strip()

            elif self.sdk_mode == "huggingface-api":
                # Hugging Face Inference API (simple prompt style)
                prompt = "\n".join([m["content"] for m in truncated_messages if m["role"] == "user"])
                # This assumes a text-generation endpoint; adjust if using a chat endpoint
                resp = self.client.text_generation("gpt2", prompt, max_new_tokens=200) # Model name might need to be dynamic
                return resp[0]["generated_text"]

            elif self.sdk_mode == "huggingface-local":
                # Local offline pipeline
                prompt = "\n".join([m["content"] for m in truncated_messages if m["role"] == "user"])
                outputs = self.client(prompt, max_new_tokens=200, num_return_sequences=1)
                return outputs[0]["generated_text"]

        except Exception as e:
            print(f"Error interacting with LLM ({self.sdk_mode}, model: {self.model_name}): {e}")
            return None



    def refine_summary_with_llm(self, initial_summary: str, validation_report: Dict[str, any]) -> Optional[str]:
        """
        Use the LLM to rewrite the dataset validation summary into one clear,
        non-repetitive paragraph highlighting key issues and next steps.
        """
        if self.client is None:
            return initial_summary

        # Retrieve relevant knowledge
        kb =self.rag.search(json.dumps(validation_report), k=3)

        # Extract just the string content from the tuples returned by rag.search
        kb_strings = [item[2] for item in kb]

        # Stronger instructions
        hidden_context = (
            "Rewrite the following dataset validation summary into ONE short, clear, and professional paragraph. "
            "Avoid repeating information. "
            "Do not include JSON, do not invent details, and do not list schema objects. "
            "Mention the dataset shape, key validation issues (missing values, duplicates, mismatches), "
            "and overall schema health. "
            "Be factual, concise, and non-repetitive.\n\n"
            "Please rewrite this into one concise, non-repetitive paragraph.\n\n"
            f"Initial Summary:\n{initial_summary}\n\n"
            f"Validation Report: {validation_report}\n\n"
            "Relevant knowledge points: " + "; ".join(kb_strings) # Use kb_strings here
        )

        # Only expose a short version to the "user" so the model doesn’t echo the full context
        visible_prompt = (
            f"Initial Summary:\n{initial_summary}\n"
        )

        messages = [
            {"role": "system", "content": "You are a senior data quality analyst. Use hidden context only for reasoning."},
            {"role": "user", "content": visible_prompt},
            {"role": "system", "content": hidden_context}  # hidden, not echoed
        ]

        refined = self._chat(messages)
        if refined:
            # final cleanup to remove any repetition that slipped through
            return self.remove_repetitive_sentences(refined.strip())
        return initial_summary

    def get_summary(self) -> str:
        return "".join(self.remove_repetitive_sentences(self.refined_summary or self.initial_summary))

    # --------------------------
    # Q&A chatbot
    # --------------------------
    def answer(self, question: str, detail_level: str = "auto") -> str:
        """
        Answers follow-up questions using validation report, refined summary,
        and retrieved knowledge base items.

        detail_level: "auto" | "executive" | "technical"
        - auto: tries to infer from question
        - executive: high-level summary
        - technical: include full validation details
        """

        kb = self.rag.search(question, k=5)
        summary = self.get_summary()
        report = self.validation_report

        # Infer level if auto
        if detail_level == "auto":
            low_words = ["summary", "overview", "high-level", "executive"]
            high_words = ["details", "mismatch", "missing", "duplicates", "dtype", "columns"]
            q_lower = question.lower()
            if any(w in q_lower for w in low_words):
                detail_level = "executive"
            elif any(w in q_lower for w in high_words):
                detail_level = "technical"
            else:
                detail_level = "executive"

        # Prepare context in natural language
        # Truncate context_text to fit the model's maximum sequence length (1024 for distilgpt2)
        max_context_length = 1024
        if detail_level == "executive":
            context_text = f"The dataset validation summary is: {summary}. Relevant knowledge: {', '.join([item[2] for item in kb])}." # Extract strings here too
        else:  # technical
            context_text = f"Validation summary: {summary}. Full validation report: {report}. Relevant knowledge: {', '.join([item[2] for item in kb])}." # And here

        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length//2] + "..." + context_text[-max_context_length//2:]


        # If no LLM, fallback
        if self.client is None:
            fallback = [
                "LLM not configured. Here is a heuristic answer based on available context:",
                f"- Summary: {summary}",
                f"- Validation report: {report}",
                f"- Top knowledge:\n  - " + "\n  - ".join([item[2] for item in kb]) # And here
            ]
            return "\n".join(fallback)

        messages = [
            {"role": "system", "content": "You are a helpful data quality analyst. Always use the provided validation summary and report in your answer."},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Context: {context_text}\n\n"
                "Answer in one concise, professional paragraph. "
                "Avoid repeating details. Include next-step recommendations if appropriate."
            )}
        ]

        raw_ans = self._chat(messages)
        if raw_ans:
            return self.remove_repetitive_sentences(raw_ans.strip())
        return "I'm unable to answer right now."


     # ---- Report overview (insightful, non-repetitive) ----
    def report_overview(self, query_hint: str = "overall risk and cost drivers") -> str:
        stats = self._quick_profile()
        retrieved = self.rag.search(query_hint, k=5)
        context_snips = "\n---\n".join([r[2] for r in retrieved])

        system = (
            "You are a senior cybersecurity data analyst. "
            "Use the provided stats and context to craft ONE concise paragraph of insights. "
            "Strict rules: do NOT repeat yourself; do NOT echo the prompt; do NOT list the inputs; "
            "no disclaimers. Provide crisp takeaways and (if useful) a next action."
        )
        user = (
            f"Dataset quick stats: {json.dumps(stats)}\n\n"
            f"Context from RAG (historical only):\n{context_snips}\n\n"
            "Task: Write a single paragraph highlighting key patterns, risk hotspots, and recommended actions."
        )

        msg = [{"role":"system","content":system},{"role":"user","content":user}]
        ans = self._chat(msg)
        if not ans:
            # Heuristic fallback
            base = [f"Analyzed {stats.get('rows','?')} records across {stats.get('cols','?')} columns."]
            if "cost_total" in stats:
                base.append(f"Total cost {stats['cost_total']} (avg {stats.get('cost_avg','N/A')}).")
            if "top_Department Affected" in stats:
                base.append(f"Most affected departments {stats['top_Department Affected']}.")
            if "top_Threat Level" in stats:
                base.append(f"Threat levels distribution {stats['top_Threat Level']}.")
            ans = " ".join(base)
        return remove_repetitive_sentences(ans)

    # ---- Comment on EDA charts ----
    def comment_on_eda(self, charts: List[Dict[str, str]]) -> List[str]:
        """
        charts: list of dicts with keys like:
          - title: str (e.g., "Threat Score by Department")
          - summary: str (brief description of what the chart shows, e.g., top 3 departments, visible spikes)
          - metric: str (optional metric name)
          - notes: str (optional extra observations)
        Returns: list of one-paragraph comments per chart.
        """
        comments = []
        base_stats = self._quick_profile()

        for ch in charts:
            retrieved = self.rag.search(ch.get("title","chart insight"), k=3)
            ctx = "\n---\n".join([r[2] for r in retrieved])

            system = (
                "You are a senior cybersecurity data analyst. "
                "Provide ONE short paragraph of chart commentary: trend, anomalies/outliers, and a practical action. "
                "Strict rules: no repetition, no generic filler, no listing of inputs, no disclaimers."
            )
            user = (
                f"Dataset quick stats: {json.dumps(base_stats)}\n\n"
                f"Chart title: {ch.get('title','N/A')}\n"
                f"Chart summary: {ch.get('summary','N/A')}\n"
                f"Metric: {ch.get('metric','N/A')}\n"
                f"Notes: {ch.get('notes','')}\n\n"
                f"Relevant historical context:\n{ctx}"
            )

            msg = [{"role":"system","content":system},{"role": "user", "content": user}]
            ans = self._chat(msg)
            if not ans:
                ans = f"{ch.get('title', 'Chart')}: No LLM available; based on summary, highlight the top movers and check anomalies."
            comments.append(remove_repetitive_sentences(ans))
        return comments

    # helper used by report_overview and others
    def _quick_profile(self) -> Dict[str, Any]:
        # Try to summarise current validation_report or return minimal
        r = {"rows": "?", "cols": "?"}
        rep = self.validation_report or {}
        shape = rep.get("shape")
        if shape:
            r["rows"], r["cols"] = int(shape[0]), int(shape[1])
        # try to surface some stats if available in validation_report (e.g., sample top cols)
        return r

    # =====================================================================
    #  Format report insight, build RAG store, create agent, run insights
    # =====================================================================

    def formatting_eda_insights_request(self, df: pd.DataFrame, print_output: bool = True):
        """
        Build RAG docs from historical_df, fit vector store, instantiate CyberInsightsAgent,
        and optionally run example overview / chart commentary

        Returns dict with keys: { 'rag_docs', 'store', 'agent', 'overview', 'comments', 'answers' }
        """
        charts_meta = eda_format_insights_request_dic["charts_meta"]
        #adhoc_questions = eda_format_insights_request_dic["adhoc_questions"]
        group_by = "Department Affected"

        # Basic validation
        if df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
            raise ValueError("historical_df must be a non-empty pandas DataFrame (already validated).")

        rag_docs = prepare_rag_documents(historical_df=df, group_by=group_by)
        insight_agent = AIValidatorAgent(expected_schema=EXPECTED_SCHEMA, knowledge_base=rag_docs)
        store = insight_agent.rag  # RAGStore object

        if print_output:
            print(f"Agent created. LLM mode: {insight_agent.sdk_mode}")

        outputs = {"rag_docs": rag_docs, "store": store, "agent": insight_agent,
                   "overview": "LLM not available for report overview.", "comments": [], "answers": []} # Initialize with fallback


        # 4) Get a one-paragraph report overview (RAG-grounded)
        try:
            overview = insight_agent.report_overview()
            if overview and "No LLM available" not in overview and "Error interacting with LLM" not in overview: # Check if LLM provided a useful response
                outputs["overview"] = overview
                if print_output:
                    print("\n=== REPORT OVERVIEW ===\n")
                    print(outputs["overview"])
            else:
                 if print_output:
                    print("\n=== REPORT OVERVIEW ===\n")
                    print("LLM not available or failed to generate report overview.")

        except Exception as e:
            if print_output:
                print("Error generating report overview:", e)
                print("LLM not available or failed to generate report overview.")


        # 5) Get commentary for EDA charts if provided
        if charts_meta:
            try:
                comments = insight_agent.comment_on_eda(charts_meta)
                if comments and all("No LLM available" not in cm and "Error interacting with LLM" not in cm for cm in comments):
                    outputs["comments"] = comments
                    if print_output:
                        print("\n=== CHART COMMENTS ===\n")
                        for title, cm in zip([c.get("title","chart") for c in charts_meta], outputs["comments"]):
                            print(f"- {title}:\n  {cm}\n")
                else:
                     if print_output:
                        print("\n=== CHART COMMENTS ===\n")
                        print("LLM not available or failed to generate chart comments.")


            except Exception as e:
                if print_output:
                    print("Error generating chart comments:", e)
                    print("LLM not available or failed to generate chart comments.")


        return outputs



#Function to analyze threat distribution by department
def analyze_threat_distribution_by_department(df):
    """
    Group the data by department and analyze the counts or proportions of different threat levels or categories within each department.
    """
    print("Analyzing Threat Distribution by Department...")
    threat_level_by_department = df.groupby('Department Affected')['Threat Level'].value_counts().unstack(fill_value=0)
    display("Threat Level Distribution by Department:", threat_level_by_department)
    return threat_level_by_department


# Function to visualize threat distribution
def visualize_threat_distribution(threat_level_by_department):
    """
    Create visualizations for threat distribution.
    """
    print("Visualizing Threat Distribution by Department...")
    # Stacked bar plot for threat levels by department
    threat_level_by_department.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.title('Threat Level Distribution by Department')
    plt.xlabel('Department Affected')
    plt.ylabel('Number of Incidents')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Threat Level')
    plt.tight_layout()
    plt.show()
# Function to investigate high-cost incidents

def investigate_high_cost_incidents(df, top_n=10):
    """
    Identify and examine incidents with the highest reported costs.
    """
    print(f"Investigating Top {top_n} High-Cost Incidents...")
    # Sort the DataFrame by 'Cost' in descending order
    df_sorted_by_cost = df.sort_values(by='Cost', ascending=False)

    # Select the top N incidents with the highest costs
    top_n_high_cost_incidents = df_sorted_by_cost.head(top_n)

    # Display the selected incidents focusing on relevant columns
    display(f"Top {top_n} High-Cost Incidents:", top_n_high_cost_incidents[['Cost', 'Department Affected', 'Threat Level', 'Issue Name', 'Date Reported']])
    return top_n_high_cost_incidents#

# Function to analyze user activity patterns for anomalies
def analyze_user_activity_patterns(df):
    """
    Explore user activity features to identify unusual patterns or outliers.
    """
    print("Analyzing User Activity Patterns for Anomalies...")
    # Select the relevant user activity features
    user_activity_features = ['Session Duration in Second', 'Data Transfer MB', 'Login Attempts', 'Num Files Accessed']
    df_activity = df[user_activity_features]

    # Calculate descriptive statistics
    descriptive_stats = df_activity.describe()
    display("Descriptive Statistics for User Activity Features:", descriptive_stats)

    # Identify potential outliers using the IQR rule
    outliers_summary = {}
    for feature in user_activity_features:
        Q1 = df_activity[feature].quantile(0.25)
        Q3 = df_activity[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_activity[(df_activity[feature] < lower_bound) | (df_activity[feature] > upper_bound)]
        outliers_summary[feature] = {
            'Lower Bound (IQR)': lower_bound,
            'Upper Bound (IQR)': upper_bound,
            'Number of Outliers': len(outliers),
            'Outlier Indices (first 5)': outliers.index.tolist()[:5]
        }

    display("Summary of Potential Outliers (IQR Rule):", pd.DataFrame(outliers_summary))
    return df_activity, outliers_summary

# Function to visualize user activity patterns
def visualize_user_activity_patterns(df, outliers_summary):
    """
    Create visualizations for user activity patterns.
    """
    print("Visualizing User Activity Patterns...")
    # Scatter plot for Session Duration vs Data Transfer
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x='Session Duration in Second', y='Data Transfer MB', data=df, hue='Is Anomaly' if 'Is Anomaly' in df.columns else None, palette='viridis')
    plt.title('Session Duration vs Data Transfer (MB)')
    plt.xlabel('Session Duration in Second')
    plt.ylabel('Data Transfer MB')
    plt.grid(True)
    plt.show()

    # Scatter plot for Login Attempts vs Num Files Accessed
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x='Login Attempts', y='Num Files Accessed', data=df, hue='Is Anomaly' if 'Is Anomaly' in df.columns else None, palette='viridis')
    plt.title('Login Attempts vs Num Files Accessed')
    plt.xlabel('Login Attempts')
    plt.ylabel('Num Files Accessed')
    plt.grid(True)
    plt.show()
#===========================================
def clean_and_format_summary(raw_output: str) -> str:
    """
    Post-process AI output: remove noise, repeated prompts,
    and enforce clean section formatting.
    """
    if not raw_output:
        return "No summary generated."

    # Remove echoed Question/Answer and system noise
    cleaned = re.sub(r"(Question:.*?Answer:)", "", raw_output, flags=re.DOTALL)
    cleaned = re.sub(r"(?i)setting `pad_token_id`.*?\n", "", cleaned)
    cleaned = re.sub(r"(?i)Using local offline.*?\n", "", cleaned)
    cleaned = re.sub(r"(?i)⚠️.*?\n", "", cleaned)

    # Normalize whitespace
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()

    # Ensure all sections exist
    if "Data Analysis Key Findings" not in cleaned:
        cleaned = "Summary:\n\nData Analysis Key Findings\n- " + cleaned
    if "Insights or Next Steps" not in cleaned:
        cleaned += "\n\nInsights or Next Steps\n- To be investigated further."
    if "Additional Observations from Charts" not in cleaned:
        cleaned += "\n\nAdditional Observations from Charts\n- None provided."

    return cleaned

def generate_executive_summary(report_overview: str,
                               chart_comments: list,
                               agent: AIValidatorAgent) -> str:
    """
    Generate a structured executive summary from report overview + chart comments.
    Uses a strict format with 3 sections, then applies post-processing cleanup.
    """
    if not report_overview or "No LLM available" in report_overview:
        return "Executive summary unavailable. No valid report overview provided."

    chart_text = "\n".join([f"- {c}" for c in chart_comments]) if chart_comments else "No additional chart observations."

    prompt = f"""
Rephrase the following REPORT OVERVIEW and CHART COMMENTS into a structured executive summary.

Use these exact section headers (do not change wording):

Summary:
Data Analysis Key Findings
- ...

Insights or Next Steps
- ...

Additional Observations from Charts
- ...

REPORT OVERVIEW:
{report_overview}

CHART COMMENTS:
{chart_text}
"""

    raw_summary = agent.answer(prompt, detail_level="executive")

    return clean_and_format_summary(raw_summary)  # Apply Fix 1 cleanup




# Function to summarize insights dynamically using the AI agent
def summarize_insights(threat_level_by_department, top_n_high_cost_incidents, outliers_summary, agent):
    """
    Synthesize the insights from the previous analyses into a concise, structured executive summary
    using the AI agent.
    """
    print("Summarizing Insights using AI Agent...")

    # Build structured prompt
    prompt = f"""
    You are a senior cybersecurity data analyst. Based on the following analysis outputs,
    write a clear, professional executive summary in the format:

    Summary:
    Data Analysis Key Findings
    [bullet-style or short paragraphs of findings]

    Insights or Next Steps
    [bullet-style or short paragraphs of recommended actions]

    Threat Distribution by Department:
    {threat_level_by_department.to_string()}

    Top High-Cost Incidents (Cost, Department, Threat Level, Issue Name, Date Reported):
    {top_n_high_cost_incidents[['Cost', 'Department Affected', 'Threat Level', 'Issue Name', 'Date Reported']].to_string()}

    Outliers Summary (IQR Rule) for user activity features:
    {pd.DataFrame(outliers_summary).to_string()}

    Requirements:
    - Highlight which departments have the most incidents and most high-level threats.
    - Note patterns in high-cost incidents (departments, threat levels, issue names).
    - Describe outlier findings across user activity features.
    - Conclude with actionable next steps for risk prioritization and investigation.
    - Avoid duplication. Be concise, factual, and well-phrased.
    """

    # Get summary from the AI agent
    summary_insights = agent.answer(prompt, detail_level="executive")

    print(summary_insights)
    return summary_insights


# Update the analysis sub-pipeline to use agent-based summary
def analysis_sub_pipeline(df, agent):
    """
    Sub-pipeline to perform the detailed analysis and visualization steps.
    """
    # 1. Analyze Threat Distribution by Department
    threat_level_by_department = analyze_threat_distribution_by_department(df)
    visualize_threat_distribution(threat_level_by_department)

    # 2. Investigate High-Cost Incidents
    top_10_high_cost_incidents = investigate_high_cost_incidents(df, top_n=10)

    # 3. Analyze User Activity Patterns for Anomalies
    df_activity, outliers_summary = analyze_user_activity_patterns(df)
    visualize_user_activity_patterns(df, outliers_summary)

    # 4. Summarize Insights dynamically using the AI agent
    summary_insights = summarize_insights(
        threat_level_by_department,
        top_10_high_cost_incidents,
        outliers_summary,
        agent
    )

    return threat_level_by_department, top_10_high_cost_incidents, outliers_summary, summary_insights

def eda_main_pipeline():
    agent_validator = AIValidatorAgent(expected_schema=EXPECTED_SCHEMA, knowledge_base=prepare_rag_documents())
    df_ = agent_validator.load_data()

    df = coerce_safe_dtypes(df_)

    extended_knowledge_base = prepare_rag_documents(df)
    agent = AIValidatorAgent(expected_schema=EXPECTED_SCHEMA, knowledge_base=extended_knowledge_base)

    print(f"✅ LLM client initialized in mode: {agent.sdk_mode}")

    report = agent.validate_dataset(df)
    print("\nSummary Statement:", agent.get_summary() + "\n")

    validation_errors = agent.validation_errors(report)
    if not validation_errors:
        print("✅ Proceeding with EDA pipeline...")

        try:
            if "explaratory_data_analysis_pipeline" in globals():
                real_world_normal_and_anomalous_df = explaratory_data_analysis_pipeline(df)

            # Run sub-pipeline (threats, costs, anomalies)
            threat_level_by_department, top_10_high_cost_incidents, outliers_summary, _ = analysis_sub_pipeline(df, agent)

            # Get REPORT OVERVIEW + CHART COMMENTS
            outputs = agent.formatting_eda_insights_request(df, print_output=True)
            report_overview = outputs.get("overview", "")
            chart_comments = outputs.get("comments", [])

            # Stack Fix 1 + Fix 2
            summary_insights = generate_executive_summary(report_overview, chart_comments, agent)


            return df, summary_insights

        except Exception as e:
            print(f"❌ Error running pipeline: {e}")
            return df, "Error generating executive summary."

    else:
        print("⚠️ Validation issues detected, skipping analysis and EDA pipeline.")
        return df, "Validation issues detected, summary skipped."


# Run main pipeline
if __name__ == "__main__":
    df, summary = eda_main_pipeline()
    print("\n--- Final Executive Summary ---")
    print(summary)
