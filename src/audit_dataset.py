import os
import json
import random
from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd
import numpy as np

# Set fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

RAW_DATA_DIR = Path(r"D:/AppleLeaf_Project/data/raw/AppleLeaf9")
SPLITS_DIR = Path(r"D:/AppleLeaf_Project/data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Directory mapping to canonical class labels
CLASS_FOLDER_MAP = {
    "Alternaria leaf spot": "Alternaria leaf spot",
    "Brown spot": "Brown spot",
    "Frogeye leaf spot": "Frogeye leaf spot",
    "Grey spot": "Grey spot",
    "Health": "Healthy",
    "Mosaic": "Mosaic",
    "Powdery mildew": "Powdery mildew",
    "Rust": "Rust",
    "Scab": "Scab"
}

def audit_and_split():
    print("=" * 70)
    print("PHASE 1: APPLELEAF9 INGESTION, AUDIT & STRATIFIED SPLIT")
    print("=" * 70)

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    records = []
    corrupt_files = []
    
    print("\nScanning class directories and verifying image integrity...")
    for folder_name, canonical_class in CLASS_FOLDER_MAP.items():
        folder_path = RAW_DATA_DIR / folder_name
        if not folder_path.exists():
            print(f"[WARNING] Missing directory: {folder_path}")
            continue

        file_list = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        print(f"  • Checking '{folder_name}' (Label: '{canonical_class}'): {len(file_list)} files found")

        for file_path in file_list:
            try:
                file_size_kb = file_path.stat().st_size / 1024.0
                if file_size_kb == 0:
                    corrupt_files.append((str(file_path), "Zero-byte file"))
                    continue

                with Image.open(file_path) as img:
                    img.verify() # Fast integrity check
                
                # Re-open to read dimensions and mode safely
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    format_str = img.format

                records.append({
                    "file_path": str(file_path).replace("\\", "/"),
                    "relative_path": str(file_path.relative_to(RAW_DATA_DIR.parent)).replace("\\", "/"),
                    "filename": file_path.name,
                    "folder": folder_name,
                    "class_name": canonical_class,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 3),
                    "channels": len(mode),
                    "mode": mode,
                    "format": format_str,
                    "size_kb": round(file_size_kb, 2)
                })

            except Exception as e:
                corrupt_files.append((str(file_path), str(e)))

    print(f"\nAudit Complete:")
    print(f"  • Total Valid Images: {len(records)}")
    print(f"  • Total Corrupted / Unreadable: {len(corrupt_files)}")

    if corrupt_files:
        print("[WARNING] Found corrupt files:")
        for cf, err in corrupt_files[:5]:
            print(f"    - {cf}: {err}")

    df = pd.DataFrame(records)
    
    # -------------------------------------------------------------
    # Class Statistics
    # -------------------------------------------------------------
    class_counts = df["class_name"].value_counts().to_dict()
    total_imgs = len(df)
    
    print("\n" + "=" * 70)
    print("CLASS BALANCE & FREQUENCY ANALYSIS")
    print("=" * 70)
    print(f"{'Class Name':<25} | {'Count':<8} | {'Percentage':<10} | {'Imbalance Ratio':<15}")
    print("-" * 70)
    
    max_count = max(class_counts.values())
    class_stats = {}
    for c_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_imgs) * 100
        ratio = max_count / count
        class_stats[c_name] = {
            "count": int(count),
            "percentage": round(pct, 2),
            "imbalance_ratio": round(ratio, 2)
        }
        print(f"{c_name:<25} | {count:<8} | {pct:>8.2f}% | {ratio:>13.2f}x")

    # Image Resolution Stats
    print("\n" + "=" * 70)
    print("IMAGE RESOLUTION & SIZE DISTRIBUTION")
    print("=" * 70)
    print(f"Width (px)  : min={df['width'].min()}, max={df['width'].max()}, median={df['width'].median():.0f}, mean={df['width'].mean():.1f}")
    print(f"Height (px) : min={df['height'].min()}, max={df['height'].max()}, median={df['height'].median():.0f}, mean={df['height'].mean():.1f}")
    print(f"Size (KB)   : min={df['size_kb'].min():.1f}, max={df['size_kb'].max():.1f}, mean={df['size_kb'].mean():.1f}")
    print(f"Color Modes : {dict(df['mode'].value_counts())}")

    # -------------------------------------------------------------
    # Stratified 70 / 15 / 15 Split
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING STRATIFIED TRAIN / VAL / TEST SPLITS (70 / 15 / 15)")
    print("=" * 70)

    train_indices = []
    val_indices = []
    test_indices = []

    # Assign class numerical IDs (0 to 8)
    unique_classes = sorted(list(class_counts.keys()))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    df["label"] = df["class_name"].map(class_to_idx)

    # Perform stratified sampling per class
    for c_name in unique_classes:
        c_df = df[df["class_name"] == c_name]
        c_indices = c_df.index.tolist()
        random.shuffle(c_indices)

        n_total = len(c_indices)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        
        c_train = c_indices[:n_train]
        c_val = c_indices[n_train:n_train + n_val]
        c_test = c_indices[n_train + n_val:]

        train_indices.extend(c_train)
        val_indices.extend(c_val)
        test_indices.extend(c_test)

    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    test_df = df.loc[test_indices].copy()

    df.loc[train_indices, "split"] = "train"
    df.loc[val_indices, "split"] = "val"
    df.loc[test_indices, "split"] = "test"

    print(f"Split Totals:")
    print(f"  • Train Set : {len(train_df)} ({len(train_df)/total_imgs*100:.1f}%)")
    print(f"  • Val Set   : {len(val_df)} ({len(val_df)/total_imgs*100:.1f}%)")
    print(f"  • Test Set  : {len(test_df)} ({len(test_df)/total_imgs*100:.1f}%)")
    print(f"  • Total Sum : {len(train_df) + len(val_df) + len(test_df)}")

    # Save manifest files
    full_manifest_path = SPLITS_DIR / "full_dataset_manifest.csv"
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"
    summary_path = SPLITS_DIR / "dataset_summary.json"

    df.to_csv(full_manifest_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary_data = {
        "dataset_name": "AppleLeaf9",
        "total_images": total_imgs,
        "classes": unique_classes,
        "class_to_idx": class_to_idx,
        "class_statistics": class_stats,
        "splits": {
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "ratio": "70/15/15"
        },
        "resolution_stats": {
            "width_mean": round(float(df["width"].mean()), 1),
            "height_mean": round(float(df["height"].mean()), 1),
            "median_width": int(df["width"].median()),
            "median_height": int(df["height"].median())
        },
        "random_seed": RANDOM_SEED
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\nArtifacts Saved Successfully to '{SPLITS_DIR}':")
    print(f"  • {full_manifest_path.name}")
    print(f"  • {train_path.name}")
    print(f"  • {val_path.name}")
    print(f"  • {test_path.name}")
    print(f"  • {summary_path.name}")

if __name__ == "__main__":
    audit_and_split()
