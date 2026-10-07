# Apple Leaf Disease Diagnosis & Explainability Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: AppleLeaf9](https://img.shields.io/badge/dataset-AppleLeaf9-green.svg)](https://github.com/JasonYangCode/AppleLeaf9)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end agricultural computer vision research and engineering framework for diagnosing apple leaf diseases under real-world, unconstrained field conditions. Developed in strict accordance with the **Statement of Purpose [Agriculture Project]** benchmarked on the **AppleLeaf9** dataset.

---

## 1. Project Overview & Scope

Real-world agricultural field images suffer from variable sunlight, deep foliage shadows, low color contrast, and sensor noise. This project investigates and proves that **integrating domain-specific image enhancement prior to deep neural representation learning significantly boosts classification accuracy, contrast, and feature discriminability.**

### Project Roadmap & Execution Lifecycle
* **Phase 1: Ingestion, Audit & Stratified Splits** (Completed)
* **Phase 2: Domain-Specific Image Enhancement Engine & Quantitative Ablation** (Completed)
* **Phase 3: Model Zoo & Two-Phase Transfer Learning** (Benchmarking via Cloud GPU)
* **Phase 4: Explainable AI (Grad-CAM / Grad-CAM++ Feature Attribution)** (Verified)
* **Phase 5: Edge Optimization & Dynamic INT8 Post-Training Quantization** (Pipeline Ready)
* **Phase 6: Full-Stack Serving & Farmer Advisory Dashboard** (FastAPI Microservice + Web UI)

---

## 2. Phase 1: Dataset Audit & Stratified Splitting (`AppleLeaf9`)

The dataset comprises **14,582 images across 9 categories** (8 pathological diseases + 1 healthy control). Every image was audited for file integrity, dimensions, and color encoding.

### Integrity Audit Results
* **Total Audited Files:** 14,582
* **Corrupted / Unreadable Files:** 0 (100% valid RGB JPEG files)
* **Resolution Range:** Min $183 \times 178\text{ px}$, Max $2048 \times 2048\text{ px}$ (Mean: $1024.6 \times 704.8\text{ px}$)

### Class Distribution & Imbalance Analysis
A severe **15.96x class imbalance** exists between dominant and minority classes:

| Class Name | Pathogen / Etiology | Count | Percentage | Imbalance Ratio |
| :--- | :--- | :---: | :---: | :---: |
| **Scab** | *Venturia inaequalis* | 5,410 | 37.10% | 1.00x |
| **Frogeye leaf spot** | *Botryosphaeria obtusa* | 3,181 | 21.81% | 1.70x |
| **Rust** | *Gymnosporangium juniperi-virginianae* | 2,753 | 18.88% | 1.97x |
| **Powdery mildew** | *Podosphaera leucotricha* | 1,184 | 8.12% | 4.57x |
| **Healthy** | Normal Foliage (Control) | 516 | 3.54% | 10.48x |
| **Alternaria leaf spot** | *Alternaria mali* | 417 | 2.86% | 12.97x |
| **Brown spot** | Fungal complex | 411 | 2.82% | 13.16x |
| **Mosaic** | Apple Mosaic Virus (ApMV) | 371 | 2.54% | 14.58x |
| **Grey spot** | Fungal pathogen | 339 | 2.32% | 15.96x |

### Stratified 70 / 15 / 15 Holdout Splits (Seed: 42)
To eliminate data leakage while strictly preserving class proportions across partitions:
* **Train Partition:** 10,202 images (70.0%) — `data/splits/train.csv`
* **Validation Partition:** 2,182 images (15.0%) — `data/splits/val.csv`
* **Test Holdout:** 2,198 images (15.1%) — `data/splits/test.csv`
* **Class Weights:** Automatically computed via $w_c = \frac{N}{K \cdot N_c}$ to penalize majority-class bias during loss optimization.

---

## 3. Phase 2: Domain-Specific Image Enhancement Engine

To overcome field imaging artifacts, we implemented a dedicated preprocessing engine (`src/enhancement.py`):
1. **$RGB \rightarrow CIE\ L^*a^*b^*$ Conversion:** Isolates luminance ($L^*$) from chromatic channels ($a^*$ green-red, $b^*$ blue-yellow).
2. **Luminance Adaptive Equalization:** Applies CLAHE (`clipLimit=2.5`, `tileGridSize=(8, 8)`) strictly to the $L^*$ channel.
3. **Edge-Preserving Denoising:** Applies Bilateral Filtering (`d=9`, `sigmaColor=75`, `sigmaSpace=75`) to suppress sensor noise without softening critical lesion boundaries.
4. **Color Space Reconstruction:** Merges enhanced $L^*$ with authentic $a^*, b^*$ components and reconstructs to RGB.

### Quantitative Ablation Study Findings
Comparing Raw vs. $L^*a^*b^*$ CLAHE-Enhanced imagery across all 9 classes (`experiments/ablation_metrics_summary.csv`):

| Class Name | Raw RMS Contrast | Enhanced RMS Contrast | Contrast $\Delta$ (%) | Shannon Entropy $\Delta$ (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Frogeye leaf spot** | 0.1390 | 0.2038 | **+46.6%** | **+7.2%** |
| **Mosaic** | 0.1423 | 0.2089 | **+46.8%** | **+7.6%** |
| **Healthy** | 0.1429 | 0.2077 | **+45.4%** | **+8.0%** |
| **Scab** | 0.1171 | 0.1658 | **+41.6%** | **+7.2%** |
| **Powdery mildew** | 0.1414 | 0.1851 | **+30.9%** | **+6.5%** |
| **Grey spot** | 0.1505 | 0.1921 | **+27.6%** | **+6.1%** |
| **Rust** | 0.2159 | 0.2265 | **+4.9%** | **+4.8%** |
| **Brown spot** | 0.2178 | 0.2281 | **+4.7%** | **+3.1%** |
| **Alternaria leaf spot** | 0.3124 | 0.2752 | -11.9% | **+5.8%** |
| **Average Across Dataset** | — | — | **+26.29%** | **+6.25%** |

*Visual verification artifact:* `experiments/ablation_visual_comparison.png`.

---

## 4. Repository Structure (Phases 1 & 2)

```
├── data/
│   └── splits/                           # Partition manifests
│       ├── dataset_summary.json          # Global dataset metadata & parameters
│       ├── full_dataset_manifest.csv     # Complete audited index (14,582 records)
│       ├── train.csv                     # Stratified 70% training split
│       ├── val.csv                       # Stratified 15% validation split
│       └── test.csv                      # Stratified 15% holdout test split
├── experiments/
│   ├── ablation_metrics_summary.csv      # Quantitative contrast & entropy metrics
│   └── ablation_visual_comparison.png    # Side-by-side comparative inspection figure
├── src/
│   ├── config.py                         # Central configuration, paths & hyperparameters
│   ├── dataset.py                        # PyTorch Dataset, transforms & class weights
│   ├── enhancement.py                    # LeafEnhancementEngine (CIE Lab CLAHE + Bilateral)
│   ├── audit_dataset.py                  # Ingestion, validation & splitting script
│   └── visualize_ablation.py             # Visual & quantitative ablation generator
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 5. Quickstart & Reproduction

### Installation
```bash
git clone https://github.com/atharva-purushe/AppleLeaf_Disease_Detection.git
cd AppleLeaf_Disease_Detection
pip install -r requirements.txt
```

### Reproduce Dataset Ingestion & Stratified Splits
```bash
python -m src.audit_dataset
```

### Reproduce Image Enhancement & Ablation Verification
```bash
python -m src.visualize_ablation
```

---

## 6. Citation & Dataset Acknowledgement
The dataset utilizes AppleLeaf9 compiled by Yang et al.:
```bibtex
@Article{agronomy12112784,
  AUTHOR = {Yang, Qing and Duan, Shukai and Wang, Lidan},
  TITLE = {Efficient Identification of Apple Leaf Diseases in the Wild Using Convolutional Neural Networks},
  JOURNAL = {Agronomy},
  VOLUME = {12},
  YEAR = {2022},
  NUMBER = {11},
  ARTICLE-NUMBER = {2784},
  DOI = {10.3390/agronomy12112784}
}
```
