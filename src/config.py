"""
Global configuration for the Multi-Modal Chest X-Ray Intelligence System.
All paths, model IDs, and hyperparameters are centralized here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RENDERED_REPORTS_DIR = DATA_DIR / "rendered_reports"
QA_DATASET_PATH = DATA_DIR / "qa_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_GEN_RESULTS_DIR = RESULTS_DIR / "report_generation"
QA_RESULTS_DIR = RESULTS_DIR / "qa_results"
COMPARISON_RESULTS_DIR = RESULTS_DIR / "comparison"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RENDERED_REPORTS_DIR,
          REPORT_GEN_RESULTS_DIR, QA_RESULTS_DIR, COMPARISON_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Credentials ─────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

# ─── Model IDs ───────────────────────────────────────────────────────────────
MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"
COLPALI_MODEL_ID = "vidore/colpali-v1.2"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# ─── Dataset ─────────────────────────────────────────────────────────────────
KAGGLE_DATASET = "simhadrisadaram/mimic-cxr-dataset"
DATASET_SUBSET_SIZE = 1000  # Number of samples to use
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.2  # 20% for testing

# ─── MedGemma Settings ──────────────────────────────────────────────────────
MEDGEMMA_MAX_NEW_TOKENS = 1024
MEDGEMMA_QUANTIZATION = "4bit"  # "4bit", "8bit", or "none"
MEDGEMMA_TORCH_DTYPE = "bfloat16"

# ─── ColPali Settings ───────────────────────────────────────────────────────
COLPALI_INDEX_NAME = "cxr_report_index"
COLPALI_TOP_K = 3  # Number of documents to retrieve

# ─── CLIP Settings ───────────────────────────────────────────────────────────
CLIP_EMBEDDING_DIM = 512
CLIP_TOP_K = 3

# ─── Report Generation ──────────────────────────────────────────────────────
REPORT_SECTIONS = [
    "findings",
    "impression",
    "pathologies_detected",
    "support_devices",
    "heart_and_mediastinum",
    "lungs_and_pleura",
    "bones_and_soft_tissues",
]

# ─── QA Categories ───────────────────────────────────────────────────────────
QA_CATEGORIES = [
    "no_finding",
    "cardiomegaly",
    "pleural_effusion",
    "pneumothorax",
    "atelectasis",
    "consolidation",
    "edema",
    "lung_opacity",
    "support_devices",
    "fracture",
    "summary",
    "location",
    "severity",
    "comparison",
]

# ─── Evaluation ──────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE = 100  # Number of samples for detailed evaluation
BLEU_MAX_NGRAM = 4
