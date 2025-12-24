from datetime import datetime
import numpy as np
import pandas as pd
import random
import socket
import struct

from CyberThreat_Insight.utils.gdrive_utils import load_csv_from_gdrive_url, load_new_data
from CyberThreat_Insight.production.stacked_ad_classifier_prod import predict_new_data


MODEL_DIR = "CyberThreat_Insight/stacked_models_deployment"
DATA_PATH =  "CyberThreat_Insight/cybersecurity_data"
AUGMENTED_DATA_PATH = DATA_PATH + "/x_y_augmented_data_google_drive.csv"
NEW_DATA_URL = "https://drive.google.com/file/d/1Nr9PymyvLfDh3qTfaeKNVbvLwt7lNX6l/view?usp=sharing"

# -------------------- Attack Classes --------------------

class BaseAttack:
    def __init__(self, df):
        self.df = df.copy()
        self.ip_generator = IPAddressGenerator()

    def apply(self):
        raise NotImplementedError("Each attack must implement the apply() method.")

class PhishingAttack(BaseAttack):
    def apply(self):
        targets = self.df[self.df["Category"] == "Access Control"].sample(frac=0.1, random_state=42)
        anomaly_magnitude = 1.0
        self.df.loc[targets.index, "Login Attempts"] += anomaly_magnitude * np.random.poisson(lam=self.df["Login Attempts"].mean(), size=len(targets))
        self.df.loc[targets.index, "Impact Score"] += anomaly_magnitude * np.random.normal(loc=self.df["Impact Score"].mean(), scale=self.df["Impact Score"].std(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Threat Score"] += anomaly_magnitude * np.random.normal(loc=self.df["Threat Score"].mean(), scale=self.df["Threat Score"].std(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Attack Type"] = "Phishing"
        return self.df

class MalwareAttack(BaseAttack):
    def apply(self):
        targets = self.df[self.df["Category"] == "System Vulnerability"].sample(frac=0.1, random_state=42)
        anomaly_magnitude = 1.0
        self.df.loc[targets.index, "Num Files Accessed"] += anomaly_magnitude * np.random.poisson(lam=self.df["Num Files Accessed"].mean(), size=len(targets))
        self.df.loc[targets.index, "Impact Score"] += anomaly_magnitude * np.random.normal(loc=self.df["Impact Score"].mean(), scale=self.df["Impact Score"].std(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Threat Score"] += anomaly_magnitude * np.random.normal(loc=self.df["Threat Score"].mean(), scale=self.df["Threat Score"].std(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Attack Type"] = "Malware"
        return self.df

class DDoSAttack(BaseAttack):
    def apply(self):
        targets = self.df[self.df["Category"] == "Network Security"].sample(frac=0.2, random_state=42)
        anomaly_magnitude = 1.0
        self.df.loc[targets.index, "Session Duration in Second"] += anomaly_magnitude * np.random.exponential(scale=self.df["Session Duration in Second"].mean(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Impact Score"] += anomaly_magnitude * np.random.exponential(scale=self.df["Impact Score"].mean(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Threat Score"] += anomaly_magnitude * np.random.exponential(scale=self.df["Threat Score"].mean(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Login Attempts"] += anomaly_magnitude * np.random.poisson(lam=self.df["Login Attempts"].mean(), size=len(targets))
        self.df.loc[targets.index, "Source IP Address"] = "192.168.1.10"
        self.df.loc[targets.index, "Destination IP Address"] = "192.168.1.10"
        self.df.loc[targets.index, "Attack Type"] = "DDoS"
        return self.df

class DataLeakAttack(BaseAttack):
    def apply(self):
        targets = self.df[self.df["Category"] == "Data Breach"].sample(frac=0.1, random_state=42)
        anomaly_magnitude = 1.0
        transfer_log_mean = np.log(self.df["Data Transfer MB"].mean())
        transfer_log_std = np.log(self.df["Data Transfer MB"].std())
        self.df.loc[targets.index, "Data Transfer MB"] += anomaly_magnitude * np.random.lognormal(mean=transfer_log_mean, sigma=transfer_log_std, size=len(targets))
        self.df.loc[targets.index, "Impact Score"] += anomaly_magnitude * np.random.lognormal(mean=np.log(self.df["Impact Score"].mean()), sigma=transfer_log_std, size=len(targets))
        self.df.loc[targets.index, "Threat Score"] += anomaly_magnitude * np.random.lognormal(mean=np.log(self.df["Threat Score"].mean()), sigma=transfer_log_std, size=len(targets))
        self.df.loc[targets.index, "Attack Type"] = "Data Leak"
        return self.df

class InsiderThreatAttack(BaseAttack):
    def apply(self):
        self.df['hour'] = pd.to_datetime(self.df['Timestamps'], errors='coerce').dt.hour
        late_hours = self.df[(self.df['hour'] < 6) | (self.df['hour'] > 23)]
        targets = late_hours.sample(frac=0.1, random_state=42)
        anomaly_magnitude = 1.0
        transfer_log_mean = np.log(self.df["Data Transfer MB"].mean())
        transfer_log_std = np.log(self.df["Data Transfer MB"].std())
        self.df.loc[targets.index, "Access Restricted Files"] = True
        self.df.loc[targets.index, "Data Transfer MB"] += anomaly_magnitude * np.random.lognormal(mean=transfer_log_mean, sigma=transfer_log_std, size=len(targets))
        self.df.loc[targets.index, "Impact Score"] += anomaly_magnitude * np.random.normal(loc=self.df["Impact Score"].mean(), scale=self.df["Impact Score"].std(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Threat Score"] += anomaly_magnitude * np.random.normal(loc=self.df["Threat Score"].mean(), scale=self.df["Threat Score"].std(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Attack Type"] = "Insider Threat"
        return self.df

class RansomwareAttack(BaseAttack):
    def apply(self):
        targets = self.df[self.df["Category"] == "System Vulnerability"].sample(frac=0.02, random_state=42)
        anomaly_magnitude = 1.0
        self.df.loc[targets.index, "CPU Usage %"] += anomaly_magnitude * np.random.normal(loc=self.df["CPU Usage %"].mean(), scale=self.df["CPU Usage %"].std(), size=len(targets))
        self.df.loc[targets.index, "Memory Usage MB"] += anomaly_magnitude * np.random.lognormal(mean=np.log(self.df["Memory Usage MB"].mean()), sigma=np.log(self.df["Memory Usage MB"].std()), size=len(targets))
        self.df.loc[targets.index, "Num Files Accessed"] += anomaly_magnitude * np.random.poisson(lam=self.df["Num Files Accessed"].mean(), size=len(targets))
        self.df.loc[targets.index, "Threat Score"] += anomaly_magnitude * np.random.normal(loc=self.df["Threat Score"].mean(), scale=self.df["Threat Score"].std(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Impact Score"] += anomaly_magnitude * np.random.normal(loc=self.df["Impact Score"].mean(), scale=self.df["Impact Score"].std(), size=len(targets)).astype(int)
        self.df.loc[targets.index, "Attack Type"] = "Ransomware"
        return self.df





class EarlyAnomalyDetectorClass:
    #def __init__(self):
    #    pass
    def __init__(self, df):
        self.df = df.copy()

    def detect_early_anomalies(self, column='Threat Score'):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        self.df['Actual Anomaly'] = ((self.df[column] < Q1 - 1.5 * IQR) | (self.df[column] > Q3 + 1.5 * IQR)).astype(int)
        #get anomlous dataframe
        df_anomalies = self.df[self.df['Actual Anomaly'] == 1]
        #get normal dataframe
        df_normal = self.df[self.df['Actual Anomaly'] == 0]

        return df_anomalies, df_normal

class DataCombiner:
    def __init__(self, normal_df, anomalous_df):
        self.normal_df = normal_df.copy()
        self.anomalous_df = anomalous_df.copy()

    def combine_data(self):
        combined_df = pd.concat([self.normal_df, self.anomalous_df], ignore_index=True)
        return combined_df

class IPAddressGenerator:
    """A class for generating random IPv4 addresses and pairs."""
    def __init__(self):
        pass
    def generate_random_ip(self):
        """Generates a random IPv4 address."""
        return socket.inet_ntoa(struct.pack('>I', random.randint(1, 0xffffffff)))

    def generate_ip_pair(self):
        """Generates a random source and destination IPv4 address pair."""
        source_ip = self.generate_random_ip()
        destination_ip = self.generate_random_ip()
        return source_ip, destination_ip

# -------------------- Combined Runner --------------------

def run_selected_attacks(df, selected_attacks, verbose=True):
    attack_map = {
        "phishing": PhishingAttack,
        "malware": MalwareAttack,
        "ddos": DDoSAttack,
        "data_leak": DataLeakAttack,
        "insider": InsiderThreatAttack,
        "ransomware": RansomwareAttack
    }
    if df is None:
        raise ValueError("Input DataFrame is None at the start of attack simulation.")

    for attack in selected_attacks:
        if verbose: print(f"[+] Applying {attack.capitalize()} Attack")
        attack_class = attack_map[attack]
        df = attack_class(df).apply()
        if df is None:
            raise ValueError(f"Attack {attack} returned None. Ensure its `.apply()` method returns a DataFrame.")

    return df


#------------------------------Main attacks simulation pipeline----------------------------
def main_attacks_simulation_pipeline(URL = None):
    #anomalous_flaged_production_df = "CyberThreat_Insight/cybersecurity_data/normal_and_anomalous_flaged_df.csv",
    #file_production_data_folder = "CyberThreat_Insight/cybersecurity_data"

    normal_and_anomalous_production_df = load_new_data(URL, 
                                                       output_dir = "CyberThreat_Insight/cybersecurity_data", 
                                                       filename = "normal_and_anomalous_df.csv" )

    selected_attacks=["phishing", "malware", "ddos", "data_leak", "insider", "ransomware"]

    # Load the dataset
    #production_df = pd.read_csv(anomalous_flaged_production_df)
    #production_df.head()


    #detect production data early anomalous
    # Check if production_df is loaded correctly
    #if production_df is not None:
    #    df_anomalies, df_normal = EarlyAnomalyDetectorClass(production_df).detect_early_anomalies()
    #else:
    #    print("Error: production_df is None. Please check the file path.")
    #    return  # Exit the function if data loading failed

    #df_anomalies_copy = df_anomalies.copy()  # Create a copy here
    #display(df_anomalies_copy.head())
    #df = DataCombiner(df_normal, df_anomalies_copy).combine_data()
    #simulate the attacks on anomalous data frame
    #simulated_attacks_df = run_selected_attacks(df_anomalies, selected_attacks, verbose=True)
    simulated_attacks_df = run_selected_attacks(normal_and_anomalous_production_df, selected_attacks, verbose=True)
    #df.head()

    #Combined normal and anomalous data frames
    #combined_normal_and_simulated_attacks_df = DataCombiner(df_normal, simulated_attacks_df).combine_data()
    #combined_normal_and_simulated_attacks_df.head()
    normal_and_simulated_attacks_class_df = predict_new_data(AUGMENTED_DATA_URL = AUGMENTED_DATA_PATH, 
                                                             model_dir = MODEL_DIR, 
                                                             ops_df = simulated_attacks_df)
    #save the combined data frame to google drive
    #save_dataframe_to_drive(normal_and_simulated_attacks_class_df,
    #                        combined_normal_and_simulated_attacks_class_df+"normal_and_simulated_attacks_class_df.csv")
    display(normal_and_simulated_attacks_df.head())

if __name__ == "__main__":

    main_attacks_simulation_pipeline(NEW_DATA_URL)
