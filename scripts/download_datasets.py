# scripts/download_dataset.py
import os
import kagglehub
import shutil
import pyreadr  # to read RData files
import pandas as pd

def download_and_prepare_dataset():
    # Define target folder in repo
    target_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
    os.makedirs(target_dir, exist_ok=True)

    # Mapping of RData → CSV filenames
    mapping = {
        "TEP_FaultFree_Training.RData": "fault_free_training.csv",
        "TEP_FaultFree_Testing.RData": "fault_free_testing.csv",
        "TEP_Faulty_Training.RData": "faulty_training.csv",
        "TEP_Faulty_Testing.RData": "faulty_testing.csv"
    }

    # Check if all CSVs exist
    all_csv_exist = all(os.path.exists(os.path.join(target_dir, csv_name)) for csv_name in mapping.values())
    if all_csv_exist:
        print("⚡ All CSV files already exist in datasets/. Skipping download and conversion.")
        return

    # Download dataset from Kaggle
    print("📥 Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("averkij/tennessee-eastman-process-simulation-dataset")
    print("✅ Dataset downloaded at:", path)

    # Copy + Convert
    for fname in os.listdir(path):
        src = os.path.join(path, fname)
        dst = os.path.join(target_dir, fname)

        # Copy RData files if not already present
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"📂 Copied {fname} → {target_dir}")
        else:
            print(f"⚡ {fname} already exists, skipping copy.")

        # Convert to CSV if in mapping and CSV not already present
        if fname in mapping:
            csv_dst = os.path.join(target_dir, mapping[fname])
            if not os.path.exists(csv_dst):
                print(f"🔄 Converting {fname} → {mapping[fname]}")
                result = pyreadr.read_r(src)   # returns a dict of dataframes
                for key in result.keys():
                    df = result[key]
                    df.to_csv(csv_dst, index=False)
                    print(f"✅ Saved {csv_dst} ({len(df)} rows, {len(df.columns)} cols)")
            else:
                print(f"⚡ {mapping[fname]} already exists, skipping conversion.")

    print("🎉 Dataset ready in datasets/ (CSV available)")

if __name__ == "__main__":
    download_and_prepare_dataset()
