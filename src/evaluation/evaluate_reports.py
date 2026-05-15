"""
Report generation evaluation using BLEU, ROUGE, and manual review framework.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import REPORT_GEN_RESULTS_DIR, BLEU_MAX_NGRAM
except ImportError:
    REPORT_GEN_RESULTS_DIR = Path("./results/report_generation")
    BLEU_MAX_NGRAM = 4


def compute_bleu(reference, hypothesis, max_ngram=None):
    """Compute BLEU score between reference and hypothesis."""
    if max_ngram is None:
        max_ngram = BLEU_MAX_NGRAM
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0

        weights = tuple([1.0 / max_ngram] * max_ngram)
        smoothing = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoothing)
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        return 0.0


def compute_rouge(reference, hypothesis):
    """Compute ROUGE scores between reference and hypothesis."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1_f": scores['rouge1'].fmeasure,
            "rouge2_f": scores['rouge2'].fmeasure,
            "rougeL_f": scores['rougeL'].fmeasure,
        }
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}


def evaluate_reports(results_df, output_dir=None):
    """
    Evaluate generated reports against ground truth.
    
    Args:
        results_df: DataFrame with 'report' and 'ground_truth' columns
        output_dir: Directory to save evaluation results
    
    Returns:
        DataFrame with per-sample metrics and summary dict
    """
    if output_dir is None:
        output_dir = REPORT_GEN_RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for idx, row in results_df.iterrows():
        generated = str(row.get("report", ""))
        reference = str(row.get("ground_truth", ""))
        if not generated or not reference or generated.startswith("ERROR"):
            continue

        bleu = compute_bleu(reference, generated)
        rouge = compute_rouge(reference, generated)

        metrics.append({
            "sample_id": row.get("sample_id", ""),
            "method": row.get("method", ""),
            "bleu4": bleu,
            **rouge,
            "gen_length": len(generated.split()),
            "ref_length": len(reference.split()),
        })

    metrics_df = pd.DataFrame(metrics)

    # Summary statistics
    summary = {}
    if len(metrics_df) > 0:
        for method in metrics_df["method"].unique():
            m = metrics_df[metrics_df["method"] == method]
            summary[method] = {
                "count": len(m),
                "bleu4_mean": m["bleu4"].mean(),
                "bleu4_std": m["bleu4"].std(),
                "rouge1_mean": m["rouge1_f"].mean(),
                "rouge2_mean": m["rouge2_f"].mean(),
                "rougeL_mean": m["rougeL_f"].mean(),
                "avg_gen_length": m["gen_length"].mean(),
            }

    # Save
    metrics_df.to_csv(output_dir / "report_metrics.csv", index=False)
    logger.info(f"Report evaluation summary:")
    for method, stats in summary.items():
        logger.info(f"  {method}: BLEU-4={stats['bleu4_mean']:.4f}, ROUGE-L={stats['rougeL_mean']:.4f}")

    return metrics_df, summary


def create_manual_review_table(results_df, n_samples=10, output_dir=None):
    """Create a table for manual review of generated reports."""
    if output_dir is None:
        output_dir = REPORT_GEN_RESULTS_DIR
    output_dir = Path(output_dir)

    sample = results_df.sample(n=min(n_samples, len(results_df)), random_state=42)
    review = sample[["sample_id", "method", "ground_truth", "report"]].copy()
    review["clinical_accuracy"] = ""  # To be filled manually
    review["completeness"] = ""
    review["hallucination_count"] = ""
    review["overall_score"] = ""

    review.to_csv(output_dir / "manual_review_template.csv", index=False)
    logger.info(f"Manual review template saved ({len(review)} samples)")
    return review
