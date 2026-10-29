"""
Data acquisition module for the UCI AI4I 2020 Predictive Maintenance dataset.
Safely downloads, verifies, and extracts the dataset into data/ai4i2020.csv.
"""

import io
import sys
import zipfile
import requests
from pathlib import Path

# Support running directly or as a module
try:
    from src.config import DATA_DIR, RAW_DATA_PATH
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.config import DATA_DIR, RAW_DATA_PATH

# Primary and fallback URLs
UCI_ZIP_URLS = [
    "https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip",
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
]


def download_dataset(target_path: Path = RAW_DATA_PATH, force: bool = False) -> Path:
    """
    Downloads the official UCI AI4I 2020 Predictive Maintenance dataset archive,
    safely validates the ZIP structure, and extracts ai4i2020.csv.

    Parameters
    ----------
    target_path : Path
        Destination path for the extracted CSV file.
    force : bool
        If True, re-downloads even if the file already exists.

    Returns
    -------
    Path
        The path to the extracted CSV file.
    """
    if target_path.exists() and not force:
        print(f"[INFO] Dataset already exists at {target_path} ({target_path.stat().st_size} bytes).")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Predictive-Maintenance-Intelligence-System/1.0"}

    download_success = False
    last_error = None

    for url in UCI_ZIP_URLS:
        try:
            print(f"[INFO] Attempting download from: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Validate ZIP format in memory
            content_bytes = response.content
            if not zipfile.is_zipfile(io.BytesIO(content_bytes)):
                raise ValueError("Downloaded content is not a valid ZIP archive.")

            with zipfile.ZipFile(io.BytesIO(content_bytes)) as z:
                # Find the target CSV inside the ZIP
                csv_members = [m for m in z.namelist() if m.endswith(".csv")]
                if not csv_members:
                    raise FileNotFoundError("No CSV file found inside the downloaded archive.")

                # Select the ai4i2020 CSV
                target_member = None
                for m in csv_members:
                    if "ai4i2020" in m.lower():
                        target_member = m
                        break
                if not target_member:
                    target_member = csv_members[0]

                print(f"[INFO] Extracting '{target_member}' to {target_path}")
                extracted_data = z.read(target_member)
                with open(target_path, "wb") as f_out:
                    f_out.write(extracted_data)

            download_success = True
            break
        except Exception as e:
            last_error = e
            print(f"[WARNING] Failed downloading from {url}: {e}")

    if not download_success:
        raise RuntimeError(
            f"Failed to download the UCI AI4I 2020 dataset from all sources. "
            f"Last error: {last_error}\n"
            f"Please verify internet connectivity or manually download the dataset from "
            f"https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset "
            f"and place 'ai4i2020.csv' in '{target_path.parent}'."
        )

    print(f"[SUCCESS] Dataset successfully saved to {target_path}")
    return target_path


if __name__ == "__main__":
    download_dataset()
