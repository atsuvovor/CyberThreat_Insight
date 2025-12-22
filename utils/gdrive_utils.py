import os
import re
import requests
import pandas as pd

def log(msg):
    print(f"[INFO] {msg}")
    
def load_csv_from_gdrive_url(
    gdrive_url: str,
    output_dir: str = "CyberThreat_Insight/cybersecurity_data",
    filename: str = "x_y_augmented_data_google_drive.csv"
) -> str:
    """
    Extracts a Google Drive file ID from a URL, downloads the CSV,
    and saves it locally.

    Parameters
    ----------
    gdrive_url : str
        Shared Google Drive file URL
    output_dir : str
        Target directory inside the repository
    filename : str
        Output CSV filename

    Returns
    -------
    str
        Absolute path to the downloaded CSV
    """

    if not gdrive_url:
        raise ValueError("Google Drive URL is None or empty")
            
    # -------------------------
    # Extract file ID
    # -------------------------
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)"
    ]

    file_id = None
    for pattern in patterns:
        match = re.search(pattern, gdrive_url)
        if match:
            file_id = match.group(1)
            break

    if not file_id:
        raise ValueError("Could not extract Google Drive file ID from URL.")

    # -------------------------
    # Download file
    # -------------------------
    os.makedirs(output_dir, exist_ok=True)

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    output_path = os.path.join(output_dir, filename)

    response = requests.get(download_url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"[INFO] Google Drive CSV downloaded to: {output_path}")
    return output_path


def load_new_data(URL, output_dir = "CyberThreat_Insight/cybersecurity_data", filename = None ):
    """
    Loads the dataset from URL end return the datasets.
    """
    log("Loading operational dataset from Google Drive ...")       
    if URL is not None:
          data_path = load_csv_from_gdrive_url(URL, output_dir, filename)     
          new_data = pd.read_csv(data_path)
    return new_data
