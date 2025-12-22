from datetime import datetime
import numpy as np
import pandas as pd
from CyberThreat_Insight.utils.gdrive_utils import load_csv_from_gdrive_url, load_new_data
from CyberThreat_Insight.production.stacked_ad_classifier_prod import predict_new_data


NEW_DATA_URL = "https://drive.google.com/file/d/1XyNzISx9al6LqMVtEo220LkMpH_jOC22/view?usp=sharing"

# --- Utility Functions ---
def ensure_datetime(df, column):
    df[column] = pd.to_datetime(df[column], errors='coerce')
    return df.dropna(subset=[column])

def filter_by_year(df, column, year):
    return df[df[column].dt.year == year]

# --- Attack Simulations ---
def simulate_phishing(df, verbose=False):
    if verbose: print("[*] Simulating Phishing...")
    targets = df[df["Category"] == "Access Control"].sample(frac=0.1)
    df.loc[targets.index, "Login Attempts"] += np.random.randint(10, 20, len(targets))
    df.loc[targets.index, "Impact Score"] += np.random.randint(10, 20, len(targets))
    df.loc[targets.index, "Threat Score"] += np.random.randint(10, 20, len(targets))
    return df

def simulate_malware(df, verbose=False):
    if verbose: print("[*] Simulating Malware...")
    targets = df[df["Category"] == "System Vulnerability"].sample(frac=0.1)
    df.loc[targets.index, "Num Files Accessed"] += np.random.randint(50, 100, len(targets))
    df.loc[targets.index, "Impact Score"] += np.random.randint(50, 100, len(targets))
    df.loc[targets.index, "Threat Score"] += np.random.randint(50, 100, len(targets))
    return df

def simulate_ddos(df, verbose=False):
    if verbose: print("[*] Simulating DDoS...")
    targets = df[df["Category"] == "Network Security"].sample(frac=0.1)
    df.loc[targets.index, "Session Duration in Second"] += np.random.randint(10000, 20000, len(targets))
    df.loc[targets.index, "Impact Score"] += np.random.randint(10000, 20000, len(targets))
    df.loc[targets.index, "Threat Score"] += np.random.randint(10000, 20000, len(targets))
    return df

def simulate_data_leak(df, verbose=False):
    if verbose: print("[*] Simulating Data Leak...")
    targets = df[df["Category"] == "Data Breach"].sample(frac=0.1)
    df.loc[targets.index, "Data Transfer MB"] += np.random.uniform(500, 1000, len(targets))
    df.loc[targets.index, "Impact Score"] += np.random.uniform(500, 1000, len(targets))
    df.loc[targets.index, "Threat Score"] += np.random.uniform(500, 1000, len(targets))
    return df

def simulate_insider_threat(df, verbose=False):
    if verbose: print("[*] Simulating Insider Threat...")
    df['hour'] = df['Timestamps'].dt.hour
    late_hours = df[(df['hour'] < 5) | (df['hour'] > 23)]
    targets = late_hours.sample(frac=0.1)
    df.loc[targets.index, "Access Restricted Files"] = True
    df.loc[targets.index, "Impact Score"] += np.random.randint(30, 60, len(targets))
    df.loc[targets.index, "Threat Score"] += np.random.randint(30, 60, len(targets))
    return df

def simulate_ransomware(df, verbose=False):
    if verbose: print("[*] Simulating Ransomware...")
    targets = df[df["Category"] == "System Vulnerability"].sample(frac=0.05)
    df.loc[targets.index, "CPU Usage %"] += np.random.uniform(50, 80, len(targets))
    df.loc[targets.index, "Memory Usage MB"] += np.random.uniform(1000, 3000, len(targets))
    df.loc[targets.index, "Num Files Accessed"] += np.random.randint(200, 500, len(targets))
    df.loc[targets.index, "Threat Score"] += np.random.randint(100, 200, len(targets))
    df.loc[targets.index, "Impact Score"] += np.random.randint(100, 200, len(targets))
    return df

#------------------------------------Save the DataFrame to a CSV file--------------------------------------
def save_dataframe_to_drive(df, save_path):
  df.to_csv(save_path, index=False)
  print(f"DataFrame saved to: {save_path}")

# --- Main Simulation Runner ---
def simulate_attack_scenarios(anomalous_flaged_production_df = None,
        file_production_data_folder = "CyberThreat_Insight/cybersecurity_data",
        year_filter=None, attacks_to_simulate=None, verbose=True):

    #anomalous_flaged_production_df = "/content/drive/My Drive/Cybersecurity Data/normal_and_anomalous_flaged_df.csv"
    #file_production_data_folder = "/content/drive/My Drive/Cybersecurity Data/"
    # Load the dataset
    attack_df = pd.read_csv(anomalous_flaged_production_df)

    attack_df = ensure_datetime(attack_df, "Timestamps")

    if year_filter:
        attack_df = filter_by_year(attack_df, "Timestamps", year_filter)
        if verbose: print(f"[i] Filtering data for year {year_filter}...")

    # Default to all if none specified
    all_attacks = {
        "phishing": simulate_phishing,
        "malware": simulate_malware,
        "ddos": simulate_ddos,
        "data_leak": simulate_data_leak,
        "insider": simulate_insider_threat,
        "ransomware": simulate_ransomware
    }

    attacks_to_simulate = attacks_to_simulate or list(all_attacks.keys())

    for attack_name in attacks_to_simulate:
        func = all_attacks.get(attack_name.lower())
        if func:
            simulated_attacks_df = func(attack_df, verbose=verbose)
        elif verbose:
            print(f"[!] Unknown attack type: {attack_name}")

    #return simulated_attacks_df

    save_dataframe_to_drive(simulated_attacks_df, file_production_data_folder+"simulated_attacks_df.csv")
    display(simulated_attacks_df.head())
    return simulated_attacks_df

def get_attacks_data(URL = None):
    anomalous_flaged_production_df = load_new_data(URL, 
                                                   output_dir = "CyberThreat_Insight/cybersecurity_data", 
                                                   filename = "normal_and_anomalous_flaged_df.csv" )
    display(anomalous_flaged_production_df)
    simulated_attacks_df = simulate_attack_scenarios(anomalous_flaged_production_df)
    display(simulated_attacks_df)
    return simulated_attacks_df

if __name__ == "__main__":
    get_attacks_data(NEW_DATA_URL)
