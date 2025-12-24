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


# -------------------- Base Attack --------------------

class BaseAttack:
    NUMERIC_COLS = [
        "Login Attempts",
        "Impact Score",
        "Threat Score",
        "Session Duration in Second",
        "CPU Usage %",
        "Memory Usage MB",
        "Num Files Accessed",
        "Data Transfer MB"
    ]

    LIMITS = {
        "CPU Usage %": (0, 100),
        "Memory Usage MB": (0, 256_000),       # 256 GB
        "Data Transfer MB": (0, 1_000_000),    # 1 TB
        "Session Duration in Second": (0, 86_400),
        "Login Attempts": (0, 10_000),
        "Num Files Accessed": (0, 100_000),
        "Impact Score": (0, 100),
        "Threat Score": (0, 100),
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.ip_generator = IPAddressGenerator()
        self._cast_numeric()

    def _cast_numeric(self):
        for col in self.NUMERIC_COLS:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").astype("float64")

    def _bounded_lognormal(self, base_mean, sigma, size):
        """
        Safe multiplicative noise generator
        """
        noise = np.random.lognormal(mean=0.0, sigma=sigma, size=size)
        return base_mean * noise

    def _clip_metrics(self):
        for col, (lo, hi) in self.LIMITS.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].clip(lo, hi)

    def apply(self):
        raise NotImplementedError

class PhishingAttack(BaseAttack):
    def apply(self):
        targets = self.df[self.df["Category"] == "Access Control"].sample(frac=0.1, random_state=42)

        self.df.loc[targets.index, "Login Attempts"] += np.random.poisson(
            lam=self.df["Login Attempts"].mean(), size=len(targets)
        )

        self.df.loc[targets.index, "Impact Score"] += np.random.normal(5, 3, size=len(targets))
        self.df.loc[targets.index, "Threat Score"] += np.random.normal(6, 3, size=len(targets))

        self.df.loc[targets.index, "Attack Type"] = "Phishing"
        self._clip_metrics()
        return self.df

class MalwareAttack(BaseAttack):
    def apply(self):
        targets = self.df[self.df["Category"] == "System Vulnerability"].sample(frac=0.1, random_state=42)

        self.df.loc[targets.index, "Num Files Accessed"] += np.random.poisson(
            lam=self.df["Num Files Accessed"].mean(), size=len(targets)
        )

        self.df.loc[targets.index, "Impact Score"] += np.random.normal(7, 4, size=len(targets))
        self.df.loc[targets.index, "Threat Score"] += np.random.normal(7, 4, size=len(targets))

        self.df.loc[targets.index, "Attack Type"] = "Malware"
        self._clip_metrics()
        return self.df

class DDoSAttack(BaseAttack):
    def apply(self):
        targets = self.df[self.df["Category"] == "Network Security"].sample(frac=0.2, random_state=42)

        self.df.loc[targets.index, "Session Duration in Second"] += np.random.exponential(
            scale=self.df["Session Duration in Second"].mean(), size=len(targets)
        )

        self.df.loc[targets.index, "Login Attempts"] += np.random.poisson(
            lam=self.df["Login Attempts"].mean(), size=len(targets)
        )

        self.df.loc[targets.index, "Impact Score"] += np.random.exponential(8, size=len(targets))
        self.df.loc[targets.index, "Threat Score"] += np.random.exponential(8, size=len(targets))

        self.df.loc[targets.index, "Attack Type"] = "DDoS"
        self._clip_metrics()
        return self.df

class DataLeakAttack(BaseAttack):
    def apply(self):
        targets = self.df[self.df["Category"] == "Data Breach"].sample(frac=0.1, random_state=42)

        mean_transfer = self.df["Data Transfer MB"].mean()
        self.df.loc[targets.index, "Data Transfer MB"] += self._bounded_lognormal(
            mean_transfer, sigma=0.4, size=len(targets)
        )

        self.df.loc[targets.index, "Impact Score"] += np.random.normal(12, 5, size=len(targets))
        self.df.loc[targets.index, "Threat Score"] += np.random.normal(12, 5, size=len(targets))

        self.df.loc[targets.index, "Attack Type"] = "Data Leak"
        self._clip_metrics()
        return self.df

class RansomwareAttack(BaseAttack):
    def apply(self):
        targets = self.df[self.df["Category"] == "System Vulnerability"].sample(frac=0.02, random_state=42)

        mean_mem = self.df["Memory Usage MB"].mean()
        self.df.loc[targets.index, "Memory Usage MB"] += self._bounded_lognormal(
            mean_mem, sigma=0.5, size=len(targets)
        )

        self.df.loc[targets.index, "CPU Usage %"] += np.random.normal(20, 10, size=len(targets))
        self.df.loc[targets.index, "Num Files Accessed"] += np.random.poisson(
            lam=self.df["Num Files Accessed"].mean(), size=len(targets)
        )

        self.df.loc[targets.index, "Threat Score"] += np.random.normal(15, 5, size=len(targets))
        self.df.loc[targets.index, "Impact Score"] += np.random.normal(15, 5, size=len(targets))

        self.df.loc[targets.index, "Attack Type"] = "Ransomware"
        self._clip_metrics()
        return self.df

def sanitize_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans numeric features for ML inference while preserving metadata columns.
    """

    df_clean = df.copy()

    # Select numeric columns only
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    # Replace inf / -inf
    df_clean[numeric_cols] = df_clean[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    # Fill NaNs using column medians
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
        df_clean[numeric_cols].median()
    )

    # Cast ONLY numeric columns to float32
    df_clean[numeric_cols] = df_clean[numeric_cols].astype("float32")

    return df_clean



class InsiderThreatAttack(BaseAttack):
    def apply(self):
        self.df["hour"] = pd.to_datetime(self.df["Timestamps"], errors="coerce").dt.hour
        late_hours = self.df[(self.df["hour"] < 6) | (self.df["hour"] > 23)]
        targets = late_hours.sample(frac=0.1, random_state=42)

        mean_transfer = self.df["Data Transfer MB"].mean()
        self.df.loc[targets.index, "Access Restricted Files"] = True
        self.df.loc[targets.index, "Data Transfer MB"] += self._bounded_lognormal(
            mean_transfer, sigma=0.3, size=len(targets)
        )

        self.df.loc[targets.index, "Impact Score"] += np.random.normal(10, 4, size=len(targets))
        self.df.loc[targets.index, "Threat Score"] += np.random.normal(10, 4, size=len(targets))

        self.df.loc[targets.index, "Attack Type"] = "Insider Threat"
        self._clip_metrics()
        return self.df


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
def main_attacks_simulation_pipeline(URL=None):
    """
    End-to-end attack simulation + stacked model inference pipeline
    """

    print("[INFO] Loading operational dataset from Google Drive ...")

    ops_df = load_new_data(
        URL,
        output_dir=DATA_PATH,
        filename="normal_and_anomalous_df.csv"
    )

    print(f"[INFO] Dataset loaded | shape={ops_df.shape}")

    # ---------------- Attack Selection ----------------
    selected_attacks = [
        "phishing",
        "malware",
        "ddos",
        "data_leak",
        "insider",
        "ransomware"
    ]

    print("[INFO] Running selected attack simulations ...")

    simulated_attacks_df = run_selected_attacks(
        ops_df,
        selected_attacks,
        verbose=True
    )

    print(f"[INFO] Simulation complete | shape={simulated_attacks_df.shape}")

        # ---------------- ML SAFETY GATE ----------------
    print("[INFO] Sanitizing simulated data for ML inference ...")

    simulated_attacks_df = sanitize_for_ml(simulated_attacks_df)

    # ---------------- SCHEMA VALIDATION ----------------
    required_cols = ["Impact Score", "Threat Score", "Attack Type"]
    missing = set(required_cols) - set(simulated_attacks_df.columns)

    if missing:
        raise ValueError(f"[ERROR] Missing required columns: {missing}")

    print("[INFO] Schema validation passed")
    simulated_attacks_df[["Impact Score", "Threat Score"]] = (
    simulated_attacks_df[["Impact Score", "Threat Score"]]
    .astype("float32")
    )

    # Save augmented dataset
    output_path = (
        f"{DATA_PATH}/simulated_with_predictions_"
        f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    simulated_attacks_df.to_csv(output_path, index=False)
    print(f"[INFO] Results saved to {output_path}")

    # ---------------- Prediction ----------------
    print("[INFO] Running stacked anomaly classifier ...")
    predict_new_data(NEW_DATA_URL = None, AUGMENTED_DATA_URL = None, model_dir = None, ops_df = None, label_col="Threat Level"):
    predictions_df = predict_new_data(
        NEW_DATA_URL = URL,
        AUGMENTED_DATA_URL=AUGMENTED_DATA_PATH,
        model_dir=MODEL_DIR,
        ops_df=simulated_attacks_df
    )

    print("[INFO] Prediction complete")

    # ---------------- PERSIST RESULTS ----------------
    output_path = (
        f"{DATA_PATH}/simulated_with_predictions_"
        f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    )

    predictions_df.to_csv(output_path, index=False)
    print(f"[INFO] Results saved to {output_path}")

    print(predictions_df.head())

    return predictions_df


if __name__ == "__main__":
    main_attacks_simulation_pipeline(NEW_DATA_URL)
