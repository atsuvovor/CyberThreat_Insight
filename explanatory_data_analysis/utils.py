import os
import sys
import pandas as pd
import requests
from io import StringIO

def running_in_colab():
    """Detect if the script is running in Google Colab."""
    return "google.colab" in sys.modules


def safe_mount_drive():
    """
    Mount Google Drive ONLY if:
    1. Running in Colab
    2. User actually needs Drive access
    """
    if running_in_colab():
        from google.colab import drive
        if not os.path.exists("/content/drive"):
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
        else:
            print("Google Drive already mounted.")
    else:
        print("Not running in Colab — skipping Google Drive mount.")


def load_dataset(path):
    """
    Universal dataset loader. Handles:
    - Google Drive paths
    - Local paths
    - Public Google Drive links
    - HTTP(s) URLs
    """

    # --- 1. Public Google Drive Link ---
    if "drive.google.com" in path:
        print("Detected shared Google Drive URL.")
        file_id = None

        if "id=" in path:
            file_id = path.split("id=")[-1]
        elif "file/d/" in path:
            file_id = path.split("file/d/")[1].split("/")[0]

        if file_id:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(download_url)
            return pd.read_csv(StringIO(response.text))
        else:
            raise ValueError("Invalid Google Drive link format.")

    # --- 2. HTTP/HTTPS URL ---
    if path.startswith("http://") or path.startswith("https://"):
        print("Detected URL. Downloading file...")
        response = requests.get(path)
        return pd.read_csv(StringIO(response.text))

    # --- 3. Google Drive Local Path in Colab ---
    if path.startswith("/content/drive"):
        safe_mount_drive()
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found in Google Drive: {path}")
        print("Loading from Google Drive...")
        return pd.read_csv(path)

    # --- 4. Local File Path ---
    if os.path.exists(path):
        print("Loading local file...")
        return pd.read_csv(path)

    raise FileNotFoundError(f"Unable to load file: {path}")
