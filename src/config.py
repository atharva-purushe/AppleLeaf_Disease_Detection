import os
from pathlib import Path
import torch

# Base Project Directories
BASE_DIR = Path(r"D:/AppleLeaf_Project")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "AppleLeaf9"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"

# Ensure runtime directories exist
for d in [PROCESSED_DIR, SPLITS_DIR, EXPERIMENTS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset Class Specifications (9 Categories)
CLASS_NAMES = [
    "Alternaria leaf spot",
    "Brown spot",
    "Frogeye leaf spot",
    "Grey spot",
    "Healthy",
    "Mosaic",
    "Powdery mildew",
    "Rust",
    "Scab"
]

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Agronomic Disease Description & Action Mapping
AGRONOMIC_ADVISORY = {
    "Healthy": {
        "etiology": "Normal Foliage (Baseline)",
        "symptoms": "Uniform green leaf blade with intact venation and no necrotic/chlorotic lesions.",
        "action": "Maintain routine orchard sanitation, standard nutritional spraying, and periodic foliage scouting."
    },
    "Alternaria leaf spot": {
        "etiology": "Fungal pathogen (Alternaria mali)",
        "symptoms": "Small brown/black circular spots (2-5mm) often surrounded by a characteristic purple halo.",
        "action": "Apply protectant fungicides (mancozeb, captan or strobilurin-based). Prune dense canopy to improve airflow."
    },
    "Brown spot": {
        "etiology": "Fungal complex",
        "symptoms": "Irregular necrotic brown lesions on leaf margins with localized drying and ragged edges.",
        "action": "Rake and destroy fallen leaves. Apply copper-based or triazole fungicide during early spring growth."
    },
    "Frogeye leaf spot": {
        "etiology": "Fungal pathogen (Botryosphaeria obtusa)",
        "symptoms": "Circular lesions featuring pale tan/gray centers bordered by distinct dark purple margins.",
        "action": "Prune out dead spurs and cankered branches (major inoculum reservoir). Apply broad-spectrum fungicide."
    },
    "Grey spot": {
        "etiology": "Fungal pathogen",
        "symptoms": "Diffuse grayish-brown lesions with surrounding chlorosis along the midrib and leaf margin.",
        "action": "Ensure adequate spacing between tree canopies; apply sulfur or preventive fungicide sprays."
    },
    "Mosaic": {
        "etiology": "Viral infection (Apple Mosaic Virus - ApMV)",
        "symptoms": "Bright yellow/cream vein-banding patterns, irregular mosaic blotches, and localized chlorosis.",
        "action": "No chemical cure exists for viral infection. Rogue and remove severely infected trees; use certified virus-free rootstocks."
    },
    "Powdery mildew": {
        "etiology": "Fungal pathogen (Podosphaera leucotricha)",
        "symptoms": "Superficial white/gray talcum-like powdery mycelial coating on young leaves and stunted shoots.",
        "action": "Prune infected terminal shoots. Apply sterol inhibitor fungicides, potassium bicarbonate, or sulfur."
    },
    "Rust": {
        "etiology": "Fungal pathogen (Gymnosporangium juniperi-virginianae - Cedar Apple Rust)",
        "symptoms": "Bright yellow/orange-rust lesions on upper leaf surface; cup-shaped aecia on lower leaf surface.",
        "action": "Remove nearby alternative host red cedars within 1-2 miles if possible. Apply myclobutanil or propiconazole."
    },
    "Scab": {
        "etiology": "Fungal pathogen (Venturia inaequalis)",
        "symptoms": "Olive-green to velvety dark brown/black scabby lesions, puckered leaves, premature defoliation.",
        "action": "Critical disease priority. Apply pre-bloom flusilazole / difenoconazole sprays and protectant captan."
    }
}

# Image Preprocessing & Enhancement Hyperparameters
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE = (8, 8)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Hardware Execution
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 # Safe default on Windows to avoid spawn multiprocessing conflicts

# Reproducibility Seed
RANDOM_SEED = 42
