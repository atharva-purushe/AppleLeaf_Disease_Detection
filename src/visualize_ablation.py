import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from src.config import (
    SPLITS_DIR,
    EXPERIMENTS_DIR,
    CLASS_NAMES
)
from src.enhancement import LeafEnhancementEngine

def compute_rms_contrast(img_rgb: np.ndarray) -> float:
    """Computes Root-Mean-Square (RMS) contrast on luminance."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return float(np.std(gray))

def compute_entropy(img_rgb: np.ndarray) -> float:
    """Computes Shannon image entropy."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def generate_visual_ablation():
    print("=" * 70)
    print("PHASE 2: VISUAL & QUANTITATIVE ABLATION ANALYSIS")
    print("=" * 70)

    val_csv_path = SPLITS_DIR / "val.csv"
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Missing val manifest: {val_csv_path}")

    df_val = pd.read_csv(val_csv_path)
    engine = LeafEnhancementEngine()

    # Pick 1 representative image per class
    sample_records = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        c_subset = df_val[df_val["class_name"] == c_name]
        if len(c_subset) > 0:
            sample_row = c_subset.iloc[0]
            sample_records.append((c_name, sample_row["file_path"]))

    n_samples = len(sample_records)
    print(f"Selected {n_samples} representative samples across all 9 classes.")

    # Prepare plot: 9 rows, 3 columns: Raw, Enhanced, Difference Heatmap
    fig, axes = plt.subplots(n_samples, 3, figsize=(14, n_samples * 3.2), dpi=150)
    plt.subplots_adjust(hspace=0.25, wspace=0.15)

    metrics_table = []

    print("\nProcessing samples and calculating contrast metrics...")
    for idx, (class_name, img_path) in enumerate(sample_records):
        raw_pil = Image.open(img_path).convert("RGB")
        raw_np = np.array(raw_pil)
        
        # Enhanced image
        enh_np = engine.enhance(raw_np)

        # Difference map in grayscale
        diff_np = cv2.absdiff(raw_np, enh_np)
        diff_gray = cv2.cvtColor(diff_np, cv2.COLOR_RGB2GRAY)

        # Compute metrics
        raw_rms = compute_rms_contrast(raw_np)
        enh_rms = compute_rms_contrast(enh_np)
        raw_ent = compute_entropy(raw_np)
        enh_ent = compute_entropy(enh_np)

        rms_delta = ((enh_rms - raw_rms) / (raw_rms + 1e-6)) * 100
        ent_delta = ((enh_ent - raw_ent) / (raw_ent + 1e-6)) * 100

        metrics_table.append({
            "class": class_name,
            "raw_rms": round(raw_rms, 4),
            "enh_rms": round(enh_rms, 4),
            "rms_delta_pct": round(rms_delta, 2),
            "raw_entropy": round(raw_ent, 4),
            "enh_entropy": round(enh_ent, 4),
            "entropy_delta_pct": round(ent_delta, 2)
        })

        # Column 1: Raw
        ax_raw = axes[idx, 0]
        ax_raw.imshow(raw_np)
        ax_raw.set_title(f"{class_name}\n[Raw Input] RMS={raw_rms:.3f}", fontsize=10, fontweight="bold", color="#1E293B")
        ax_raw.axis("off")

        # Column 2: Enhanced (CLAHE L*a*b* + Bilateral)
        ax_enh = axes[idx, 1]
        ax_enh.imshow(enh_np)
        ax_enh.set_title(f"Enhanced (L*a*b* CLAHE)\nRMS={enh_rms:.3f} (+{rms_delta:.1f}%)", fontsize=10, fontweight="bold", color="#15803D")
        ax_enh.axis("off")

        # Column 3: Amplified Lesion / Feature Delta Heatmap
        ax_diff = axes[idx, 2]
        im_diff = ax_diff.imshow(diff_gray, cmap="magma")
        ax_diff.set_title(f"Extracted Contrast Delta\nEntropy Δ: +{ent_delta:.1f}%", fontsize=10, fontweight="bold", color="#B91C1C")
        ax_diff.axis("off")

    output_fig_path = EXPERIMENTS_DIR / "ablation_visual_comparison.png"
    plt.tight_layout()
    plt.savefig(output_fig_path, bbox_inches="tight", dpi=150)
    plt.close()

    metrics_df = pd.DataFrame(metrics_table)
    metrics_csv_path = EXPERIMENTS_DIR / "ablation_metrics_summary.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    print("\n" + "=" * 70)
    print("QUANTITATIVE ENHANCEMENT VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"{'Class Name':<25} | {'RMS (Raw)':<10} | {'RMS (Enh)':<10} | {'RMS Δ':<8} | {'Entropy Δ':<10}")
    print("-" * 70)
    for m in metrics_table:
        print(f"{m['class']:<25} | {m['raw_rms']:<10.4f} | {m['enh_rms']:<10.4f} | {m['rms_delta_pct']:>6.1f}% | {m['entropy_delta_pct']:>8.1f}%")

    avg_rms_gain = metrics_df["rms_delta_pct"].mean()
    avg_ent_gain = metrics_df["entropy_delta_pct"].mean()
    print("-" * 70)
    print(f"Mean Contrast Improvement : +{avg_rms_gain:.2f}% across all classes")
    print(f"Mean Entropy Improvement  : +{avg_ent_gain:.2f}% across all classes")
    print(f"\nArtifacts Saved Successfully:")
    print(f"  • Visual comparison figure : {output_fig_path}")
    print(f"  • Quantitative metrics     : {metrics_csv_path}")

if __name__ == "__main__":
    generate_visual_ablation()
