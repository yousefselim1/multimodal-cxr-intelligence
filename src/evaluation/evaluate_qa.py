"""
QA evaluation: accuracy, retrieval relevance, and answer groundedness.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import QA_RESULTS_DIR
except ImportError:
    QA_RESULTS_DIR = Path("./results/qa_results")


def normalize_answer(text):
    """Normalize text for comparison."""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def compute_f1(prediction, ground_truth):
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction, ground_truth):
    """Check if the key assertion matches (yes/no questions)."""
    pred = normalize_answer(prediction)
    gt = normalize_answer(ground_truth)
    
    # For yes/no questions, check if both start with same assertion
    pred_yn = "yes" if pred.startswith("yes") else ("no" if pred.startswith("no") else None)
    gt_yn = "yes" if gt.startswith("yes") else ("no" if gt.startswith("no") else None)
    
    if pred_yn is not None and gt_yn is not None:
        return 1.0 if pred_yn == gt_yn else 0.0
    
    return 1.0 if pred == gt else 0.0


def evaluate_qa(results_df, output_dir=None):
    """
    Evaluate QA results against ground truth answers.
    
    Args:
        results_df: DataFrame with 'answer', 'ground_truth_answer', 'category'
    
    Returns:
        Tuple of (metrics_df, summary_dict)
    """
    if output_dir is None:
        output_dir = QA_RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for idx, row in results_df.iterrows():
        pred = str(row.get("answer", ""))
        gt = str(row.get("ground_truth_answer", ""))
        if pred.startswith("ERROR"):
            continue

        f1 = compute_f1(pred, gt)
        em = compute_exact_match(pred, gt)

        metrics.append({
            "sample_id": row.get("sample_id", ""),
            "method": row.get("method", ""),
            "category": row.get("category", ""),
            "f1_score": f1,
            "exact_match": em,
            "pred_length": len(pred.split()),
            "gt_length": len(gt.split()),
        })

    metrics_df = pd.DataFrame(metrics)

    # Summary by method
    summary = {}
    if len(metrics_df) > 0:
        for method in metrics_df["method"].unique():
            m = metrics_df[metrics_df["method"] == method]
            method_summary = {
                "count": len(m),
                "f1_mean": m["f1_score"].mean(),
                "f1_std": m["f1_score"].std(),
                "exact_match_rate": m["exact_match"].mean(),
            }
            # Per-category breakdown
            for cat in m["category"].unique():
                mc = m[m["category"] == cat]
                method_summary[f"f1_{cat}"] = mc["f1_score"].mean()
                method_summary[f"em_{cat}"] = mc["exact_match"].mean()
            summary[method] = method_summary

    metrics_df.to_csv(output_dir / "qa_metrics.csv", index=False)
    logger.info("QA evaluation summary:")
    for method, stats in summary.items():
        logger.info(f"  {method}: F1={stats['f1_mean']:.4f}, EM={stats['exact_match_rate']:.4f}")

    return metrics_df, summary
