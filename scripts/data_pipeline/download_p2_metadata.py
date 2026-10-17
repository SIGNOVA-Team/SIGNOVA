"""
SIGNOVA - P2 Metadata and Annotations Downloader
Automates downloading of official annotations, metadata, and code archives for:
  - P2: How2Sign (Continuous ASL) -> data/how2sign/raw/
  - P2: ArabSign (Continuous ArSL) -> data/arabsign/raw/
"""

import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

HOW2SIGN_FILES = {
    "how2sign_train.csv": "https://drive.usercontent.google.com/download?id=1kE10vaZh_LQvZtBvT-jxT5nR9DXU6H7N&export=download",
    "how2sign_val.csv": "https://drive.usercontent.google.com/download?id=12xVToQ2vm6pnXbgTyk2VqfnIxrMsORF1&export=download",
    "how2sign_test.csv": "https://drive.usercontent.google.com/download?id=1B6WVb3Mtwof3mx3Pnf-UOOyXRpFaT9Hw&export=download",
    "how2sign_realigned_train.csv": "https://drive.usercontent.google.com/download?id=1BWt2ASmOIUM8tWnCuRtl9AdQsp4zsjPN&export=download",
    "how2sign_realigned_val.csv": "https://drive.usercontent.google.com/download?id=1hcPqXfHIHHGHUQYfT3eFARTIgi0M1C9_&export=download",
    "how2sign_realigned_test.csv": "https://drive.usercontent.google.com/download?id=1OsTvMsVFOMk54r65v2gcxOottV41ZEUo&export=download",
}

ARABSIGN_CODE_ZIP = "https://github.com/Hamzah-Luqman/ArabSign/raw/main/ArabSign-EncoderDec.zip"


def download_file(url, target_path):
    print(f"Downloading {os.path.basename(target_path)}...")
    urllib.request.urlretrieve(url, target_path)
    print(f"Saved: {target_path} ({os.path.getsize(target_path)} bytes)")


def setup_p2_data(data_root):
    h2s_dir = os.path.join(data_root, "how2sign", "raw")
    arab_dir = os.path.join(data_root, "arabsign", "raw")
    os.makedirs(h2s_dir, exist_ok=True)
    os.makedirs(arab_dir, exist_ok=True)

    # 1. Download How2Sign metadata CSVs
    print("\n--- [1/2] Fetching How2Sign Official Text & Annotation CSVs ---")
    for fname, url in HOW2SIGN_FILES.items():
        dst = os.path.join(h2s_dir, fname)
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            try:
                download_file(url, dst)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"Failed to download {fname}: {e}")
        else:
            print(f"Already exists: {fname}")

    # 2. Download ArabSign Code & Splits Zip
    print("\n--- [2/2] Fetching ArabSign Code, Splits & GroundTruth Archive ---")
    gt_dst = os.path.join(arab_dir, "groundTruth.txt")
    splits_dst = os.path.join(arab_dir, "author_splits")
    if not os.path.exists(gt_dst) or not os.path.exists(splits_dst):
        zip_dst = os.path.join(arab_dir, "temp_ArabSign.zip")
        try:
            download_file(ARABSIGN_CODE_ZIP, zip_dst)
            print("Extracting ArabSign ground truth and author splits...")
            with zipfile.ZipFile(zip_dst, "r") as z:
                z.extractall(os.path.join(arab_dir, "_temp_ext"))

            src_data = os.path.join(
                arab_dir, "_temp_ext", "ArabSign-EncoderDec", "EncoderDec", "data"
            )
            if os.path.exists(src_data):
                shutil.copy(os.path.join(src_data, "groundTruth.txt"), arab_dir)
                os.makedirs(splits_dst, exist_ok=True)
                for f in os.listdir(src_data):
                    if f.endswith(".csv"):
                        shutil.copy(os.path.join(src_data, f), splits_dst)
            # Cleanup temporary extraction
            shutil.rmtree(os.path.join(arab_dir, "_temp_ext"), ignore_errors=True)
            if os.path.exists(zip_dst):
                os.remove(zip_dst)
            print("ArabSign raw files organized successfully.")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"Failed to download/extract ArabSign: {e}")
    else:
        print("ArabSign raw files already exist.")

    print("\nAll P2 metadata assets downloaded and organized.")


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    setup_p2_data(data_dir)
