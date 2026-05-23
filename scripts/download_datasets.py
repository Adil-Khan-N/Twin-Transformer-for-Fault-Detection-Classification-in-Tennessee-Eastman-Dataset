import os
import shutil
import subprocess
import zipfile
import pandas as pd

def download_and_prepare_dataset():
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    os.makedirs(target_dir, exist_ok=True)

    mapping = {
        "TEP_FaultFree_Training.RData": "fault_free_training.csv",
        "TEP_FaultFree_Testing.RData": "fault_free_testing.csv",
        "TEP_Faulty_Training.RData": "faulty_training.csv",
        "TEP_Faulty_Testing.RData": "faulty_testing.csv",
    }

    all_csv_exist = all(os.path.exists(os.path.join(target_dir, csv_name)) for csv_name in mapping.values())
    if all_csv_exist:
        print("⚡ All CSV files already exist in datasets/. Skipping download and conversion.")
        return

    print("📥 Downloading dataset from Kaggle...")
    download_location = None
    try:
        import kagglehub
        download_location = kagglehub.dataset_download("averkij/tennessee-eastman-process-simulation-dataset")
        print("✅ Dataset downloaded at:", download_location)
    except Exception as e:
        print(f"⚠️ kagglehub download failed: {e}")
        print("⏳ Falling back to Kaggle CLI if available.")
        try:
            subprocess.check_call([
                "kaggle",
                "datasets",
                "download",
                "-d",
                "averkij/tennessee-eastman-process-simulation-dataset",
                "-p",
                target_dir,
                "--unzip",
            ])
            download_location = target_dir
            print("✅ Kaggle CLI download and unzip completed.")
        except Exception as cli_error:
            raise RuntimeError(
                "Unable to download dataset automatically. Install kagglehub or configure the Kaggle CLI and retry."
            ) from cli_error

    if download_location and os.path.isfile(download_location) and download_location.lower().endswith(".zip"):
        extracted_dir = os.path.join(target_dir, "downloaded")
        os.makedirs(extracted_dir, exist_ok=True)
        with zipfile.ZipFile(download_location, "r") as zf:
            zf.extractall(extracted_dir)
        download_location = extracted_dir
        print(f"✅ Extracted download to {download_location}")

    source_dir = download_location if os.path.isdir(download_location) else os.path.dirname(download_location)
    if source_dir is None:
        source_dir = target_dir

    try:
        import pyreadr
    except ImportError as e:
        raise ImportError(
            "pyreadr is required to convert RData files to CSV. Install it with `pip install pyreadr`."
        ) from e

    for fname, csv_name in mapping.items():
        src = os.path.join(source_dir, fname)
        csv_dst = os.path.join(target_dir, csv_name)

        if not os.path.exists(src):
            print(f"⚠️ Expected file {fname} not found in {source_dir}. Skipping this file.")
            continue

        if not os.path.exists(csv_dst):
            print(f"🔄 Converting {fname} → {csv_name}")
            result = pyreadr.read_r(src)
            if not result:
                raise ValueError(f"Unable to read {src} with pyreadr.")
            for key, df in result.items():
                df.to_csv(csv_dst, index=False)
                print(f"✅ Saved {csv_dst} ({len(df)} rows, {len(df.columns)} cols)")
        else:
            print(f"⚡ {csv_name} already exists, skipping conversion.")

    print("🎉 Dataset ready in datasets/ (CSV available)")

if __name__ == "__main__":
    download_and_prepare_dataset()
