"""
Data preprocessing for MIMIC-CXR dataset.
Handles cleaning, subsetting, and splitting.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import (
        PROCESSED_DATA_DIR, DATASET_SUBSET_SIZE,
        RANDOM_SEED, TEST_SPLIT_RATIO
    )
except ImportError:
    PROCESSED_DATA_DIR = Path("./data/processed")
    DATASET_SUBSET_SIZE = 1000
    RANDOM_SEED = 42
    TEST_SPLIT_RATIO = 0.2


def clean_report_text(text):
    """
    Clean a radiology report text by removing common artifacts.
    
    Removes:
        - DICOM headers and metadata lines
        - De-identification markers (e.g., [**...**])
        - Excessive whitespace
        - Empty section headers
        - Non-UTF8 characters
    
    Args:
        text: Raw report text string
    
    Returns:
        Cleaned report text string
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove de-identification markers like [**2021-01-01**] or [**Name**]
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
    
    # Remove common DICOM/metadata headers
    header_patterns = [
        r'(?i)^(wet read|final report|addendum|indication|history|technique|comparison):?\s*',
        r'(?i)^(clinical information|clinical history|reason for exam):?\s*',
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip lines that are just underscores or dashes (separators)
        if re.match(r'^[_\-=]{3,}$', line):
            continue
        
        # Skip very short lines that look like headers without content
        if len(line) < 3 and not line[0].isalpha():
            continue
        
        cleaned_lines.append(line)
    
    text = ' '.join(cleaned_lines)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove any remaining non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
    
    return text


def filter_valid_samples(df):
    """
    Filter DataFrame to keep only valid samples with:
        - Non-empty report text
        - Minimum report length (at least 20 characters)
        - Valid image path (if column exists)
    
    Args:
        df: DataFrame with 'report_text' and optionally 'image_path'
    
    Returns:
        Filtered DataFrame
    """
    initial_count = len(df)
    
    # Filter non-empty reports
    df = df[df["report_text"].notna() & (df["report_text"].str.len() > 0)]
    logger.info(f"After removing empty reports: {len(df)} / {initial_count}")
    
    # Filter minimum length
    df = df[df["report_text"].str.len() >= 20]
    logger.info(f"After minimum length filter: {len(df)}")
    
    # Filter valid image paths if column exists
    if "image_path" in df.columns:
        # Check which image files actually exist
        valid_images = df["image_path"].apply(
            lambda p: p is not None and Path(str(p)).exists()
        )
        existing_count = valid_images.sum()
        
        if existing_count > 0:
            df = df[valid_images]
            logger.info(f"After image path validation: {len(df)}")
        else:
            logger.warning(
                "No image files found on disk. Keeping all rows "
                "(images may need to be downloaded separately)."
            )
    
    return df.reset_index(drop=True)


def create_subset(df, subset_size=None, seed=None):
    """
    Create a random subset of the dataset.
    
    Args:
        df: Full DataFrame
        subset_size: Number of samples (defaults to config)
        seed: Random seed (defaults to config)
    
    Returns:
        Subset DataFrame
    """
    if subset_size is None:
        subset_size = DATASET_SUBSET_SIZE
    if seed is None:
        seed = RANDOM_SEED
    
    if len(df) <= subset_size:
        logger.info(f"Dataset size ({len(df)}) <= subset size ({subset_size}). Using full dataset.")
        return df
    
    df_subset = df.sample(n=subset_size, random_state=seed).reset_index(drop=True)
    logger.info(f"Created subset: {len(df_subset)} samples from {len(df)} total")
    
    return df_subset


def split_dataset(df, test_ratio=None, seed=None):
    """
    Split dataset into train and test sets.
    
    Args:
        df: DataFrame to split
        test_ratio: Fraction for test set (defaults to config)
        seed: Random seed (defaults to config)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    if test_ratio is None:
        test_ratio = TEST_SPLIT_RATIO
    if seed is None:
        seed = RANDOM_SEED
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    logger.info(f"Split: {len(train_df)} train, {len(test_df)} test")
    
    return train_df, test_df


def preprocess_pipeline(df, subset_size=None):
    """
    Run the full preprocessing pipeline:
        1. Clean report texts
        2. Filter valid samples
        3. Create subset
        4. Split into train/test
        5. Save processed files
    
    Args:
        df: Raw DataFrame from load_dataset()
        subset_size: Number of samples for subset
    
    Returns:
        Tuple of (train_df, test_df, full_subset_df)
    """
    logger.info("=" * 50)
    logger.info("Starting preprocessing pipeline...")
    
    # Step 1: Clean reports
    logger.info("Step 1: Cleaning report texts...")
    df["report_text"] = df["report_text"].apply(clean_report_text)
    
    # Step 2: Filter
    logger.info("Step 2: Filtering valid samples...")
    df = filter_valid_samples(df)
    
    # Step 3: Subset
    logger.info("Step 3: Creating subset...")
    df_subset = create_subset(df, subset_size)
    
    # Step 4: Add unique IDs
    df_subset["sample_id"] = [f"cxr_{i:05d}" for i in range(len(df_subset))]
    
    # Step 5: Split
    logger.info("Step 4: Splitting dataset...")
    train_df, test_df = split_dataset(df_subset)
    
    # Step 6: Save
    logger.info("Step 5: Saving processed files...")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    df_subset.to_csv(PROCESSED_DATA_DIR / "dataset_full.csv", index=False)
    train_df.to_csv(PROCESSED_DATA_DIR / "dataset_train.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "dataset_test.csv", index=False)
    
    logger.info(f"Saved to {PROCESSED_DATA_DIR}")
    logger.info("=" * 50)
    
    # Print summary stats
    logger.info(f"Full subset: {len(df_subset)} samples")
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    logger.info(f"Avg report length: {df_subset['report_text'].str.len().mean():.0f} chars")
    logger.info(f"Min report length: {df_subset['report_text'].str.len().min()} chars")
    logger.info(f"Max report length: {df_subset['report_text'].str.len().max()} chars")
    
    return train_df, test_df, df_subset


if __name__ == "__main__":
    from src.data.load_dataset import load_dataset
    
    df = load_dataset()
    train_df, test_df, full_df = preprocess_pipeline(df)
    
    print("\nSample cleaned report:")
    print(train_df["report_text"].iloc[0][:500])
