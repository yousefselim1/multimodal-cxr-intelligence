"""
Dataset loader for MIMIC-CXR dataset from Kaggle.
Handles downloading, loading, and basic validation.
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config - handle both local and Kaggle notebook environments
try:
    from src.config import (
        RAW_DATA_DIR, KAGGLE_DATASET, HF_TOKEN,
        KAGGLE_USERNAME, KAGGLE_KEY
    )
except ImportError:
    # When running in Kaggle notebook, paths may differ
    RAW_DATA_DIR = Path("./data/raw")
    KAGGLE_DATASET = "simhadrisadaram/mimic-cxr-dataset"
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")


def download_dataset_kaggle(output_dir=None):
    """
    Download MIMIC-CXR dataset from Kaggle using the Kaggle API.
    
    Prerequisites:
        - Kaggle API token (kaggle.json) in ~/.kaggle/ 
        - Or KAGGLE_USERNAME and KAGGLE_KEY environment variables
    
    Args:
        output_dir: Directory to save the dataset. Defaults to RAW_DATA_DIR.
    
    Returns:
        Path to the downloaded dataset directory.
    """
    if output_dir is None:
        output_dir = RAW_DATA_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set Kaggle credentials if available from env
    if KAGGLE_USERNAME and KAGGLE_KEY:
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        logger.info(f"Downloading {KAGGLE_DATASET} to {output_dir}...")
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(output_dir),
            unzip=True
        )
        logger.info("Download complete!")
        return output_dir
        
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        logger.info(
            "Alternative: Download manually from "
            f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"and extract to {output_dir}"
        )
        raise


def find_dataset_files(data_dir=None):
    """
    Scan the data directory to find CSV/parquet files and image directories.
    
    Returns:
        dict with keys: 'csv_files', 'parquet_files', 'image_dirs', 'image_files'
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    data_dir = Path(data_dir)
    
    result = {
        "csv_files": list(data_dir.rglob("*.csv")),
        "parquet_files": list(data_dir.rglob("*.parquet")),
        "image_dirs": [],
        "image_files": [],
    }
    
    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}
    for ext in image_extensions:
        result["image_files"].extend(list(data_dir.rglob(f"*{ext}")))
    
    # Find directories containing images
    image_parents = set(f.parent for f in result["image_files"])
    result["image_dirs"] = list(image_parents)
    
    logger.info(f"Found {len(result['csv_files'])} CSV files")
    logger.info(f"Found {len(result['parquet_files'])} parquet files")
    logger.info(f"Found {len(result['image_files'])} image files")
    logger.info(f"Found {len(result['image_dirs'])} image directories")
    
    return result


def load_dataset(data_dir=None):
    """
    Load the MIMIC-CXR dataset into a pandas DataFrame.
    Automatically detects CSV or parquet format.
    
    Expected columns (will be verified):
        - image path column (various possible names)
        - text/report column (the report text)
    
    Returns:
        pd.DataFrame with standardized columns: 'image_path', 'report_text'
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR
    data_dir = Path(data_dir)
    
    files = find_dataset_files(data_dir)
    
    df = None
    
    # Try CSV first
    if files["csv_files"]:
        for csv_file in files["csv_files"]:
            logger.info(f"Loading CSV: {csv_file}")
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
                logger.info(f"Columns: {list(df.columns)}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
    
    # Try parquet
    if df is None and files["parquet_files"]:
        for pq_file in files["parquet_files"]:
            logger.info(f"Loading parquet: {pq_file}")
            try:
                df = pd.read_parquet(pq_file)
                logger.info(f"Loaded {len(df)} rows from {pq_file.name}")
                logger.info(f"Columns: {list(df.columns)}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {pq_file}: {e}")
    
    if df is None:
        raise FileNotFoundError(
            f"No loadable dataset files found in {data_dir}. "
            "Please download the dataset first."
        )
    
    # Standardize column names
    df = standardize_columns(df, data_dir)
    
    return df


def standardize_columns(df, data_dir=None):
    """
    Map various possible column names to our standard schema:
        - image_path: path to the chest X-ray image
        - report_text: the radiology report text
    
    Args:
        df: Raw DataFrame
        data_dir: Base directory for resolving relative image paths
    
    Returns:
        DataFrame with standardized columns
    """
    # Common column name mappings
    image_col_candidates = [
        "image_path", "image", "img_path", "file_path", "filepath",
        "path", "filename", "dicom_id", "image_id", "img",
        "image_file", "file_name", "image_name"
    ]
    
    text_col_candidates = [
        "text", "report", "report_text", "findings", "impression",
        "radiology_report", "clinical_report", "narrative",
        "report_content", "description"
    ]
    
    columns_lower = {c.lower().strip(): c for c in df.columns}
    
    # Find image column
    image_col = None
    for candidate in image_col_candidates:
        if candidate in columns_lower:
            image_col = columns_lower[candidate]
            break
    
    # Find text column
    text_col = None
    for candidate in text_col_candidates:
        if candidate in columns_lower:
            text_col = columns_lower[candidate]
            break
    
    if image_col is None:
        logger.warning(
            f"Could not find image column. Available columns: {list(df.columns)}"
        )
        # Try to use the first column that looks like a path
        for col in df.columns:
            sample = str(df[col].iloc[0]) if len(df) > 0 else ""
            if any(ext in sample.lower() for ext in [".jpg", ".png", ".dcm", "/"]):
                image_col = col
                logger.info(f"Auto-detected image column: {col}")
                break
    
    if text_col is None:
        logger.warning(
            f"Could not find text column. Available columns: {list(df.columns)}"
        )
        # Try to use the column with longest average string length
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) > 0:
            avg_lens = {
                col: df[col].astype(str).str.len().mean() 
                for col in str_cols
            }
            text_col = max(avg_lens, key=avg_lens.get)
            logger.info(f"Auto-detected text column: {text_col} (avg len: {avg_lens[text_col]:.0f})")
    
    # Rename columns
    rename_map = {}
    if image_col and image_col != "image_path":
        rename_map[image_col] = "image_path"
    if text_col and text_col != "report_text":
        rename_map[text_col] = "report_text"
    
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {rename_map}")
    
    # Verify required columns exist
    required = ["report_text"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
    
    # Resolve image paths if data_dir is provided
    if "image_path" in df.columns and data_dir is not None:
        df["image_path"] = df["image_path"].apply(
            lambda p: resolve_image_path(p, data_dir)
        )
    
    return df


def resolve_image_path(path_str, data_dir):
    """
    Resolve a potentially relative image path to an absolute path.
    Tries multiple strategies to find the actual file.
    """
    if pd.isna(path_str):
        return None
    
    path_str = str(path_str).strip()
    path = Path(path_str)
    
    # Already absolute and exists
    if path.is_absolute() and path.exists():
        return str(path)
    
    # Try relative to data_dir
    candidates = [
        Path(data_dir) / path_str,
        Path(data_dir) / Path(path_str).name,
        Path(data_dir) / "images" / Path(path_str).name,
        Path(data_dir) / "files" / path_str,
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    
    # Return as-is if we can't resolve (will be filtered later)
    return path_str


def get_dataset_stats(df):
    """
    Print basic statistics about the loaded dataset.
    """
    stats = {
        "total_samples": len(df),
        "columns": list(df.columns),
        "report_text_present": df["report_text"].notna().sum() if "report_text" in df.columns else 0,
        "avg_report_length": df["report_text"].astype(str).str.len().mean() if "report_text" in df.columns else 0,
    }
    
    if "image_path" in df.columns:
        stats["image_path_present"] = df["image_path"].notna().sum()
        # Check how many images actually exist on disk
        stats["images_found"] = sum(
            1 for p in df["image_path"].dropna() 
            if Path(str(p)).exists()
        )
    
    logger.info("=" * 50)
    logger.info("Dataset Statistics:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)
    
    return stats


if __name__ == "__main__":
    # Quick test
    print("Attempting to load dataset...")
    try:
        df = load_dataset()
        get_dataset_stats(df)
        print("\nSample report:")
        print(df["report_text"].iloc[0][:500])
    except FileNotFoundError:
        print("Dataset not found. Run download_dataset_kaggle() first.")
