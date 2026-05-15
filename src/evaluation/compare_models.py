"""
Model comparison: generates side-by-side comparison tables and visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import COMPARISON_RESULTS_DIR
except ImportError:
    COMPARISON_RESULTS_DIR = Path("./results/comparison")


def compare_report_generation(report_metrics_dict, output_dir=None):
    """
    Compare report generation across methods.
    
    Args:
        report_metrics_dict: {method_name: summary_dict} from evaluate_reports
    
    Returns:
        DataFrame comparison table
    """
    if output_dir is None:
        output_dir = COMPARISON_RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, stats in report_metrics_dict.items():
        rows.append({
            "Method": method,
            "Samples": stats.get("count", 0),
            "BLEU-4": f"{stats.get('bleu4_mean', 0):.4f}",
            "ROUGE-1": f"{stats.get('rouge1_mean', 0):.4f}",
            "ROUGE-2": f"{stats.get('rouge2_mean', 0):.4f}",
            "ROUGE-L": f"{stats.get('rougeL_mean', 0):.4f}",
            "Avg Length": f"{stats.get('avg_gen_length', 0):.0f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "report_comparison.csv", index=False)
    logger.info(f"Report comparison table:\n{df.to_string(index=False)}")
    return df


def compare_qa_performance(qa_metrics_dict, output_dir=None):
    """
    Compare QA performance across methods.
    
    Args:
        qa_metrics_dict: {method_name: summary_dict} from evaluate_qa
    
    Returns:
        DataFrame comparison table
    """
    if output_dir is None:
        output_dir = COMPARISON_RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, stats in qa_metrics_dict.items():
        rows.append({
            "Method": method,
            "Samples": stats.get("count", 0),
            "F1 Score": f"{stats.get('f1_mean', 0):.4f}",
            "F1 Std": f"{stats.get('f1_std', 0):.4f}",
            "Exact Match": f"{stats.get('exact_match_rate', 0):.4f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "qa_comparison.csv", index=False)
    logger.info(f"QA comparison table:\n{df.to_string(index=False)}")
    return df


def compare_retrieval(colpali_results, clip_results, output_dir=None):
    """Compare ColPali vs CLIP retrieval quality."""
    if output_dir is None:
        output_dir = COMPARISON_RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "Retriever": "ColPali",
            "Type": "Visual Document Retrieval",
            "Medical Specialization": "No (general documents)",
            "Input": "Document page images",
            "Avg Score": f"{np.mean([r.get('score', 0) for r in colpali_results]):.4f}" if colpali_results else "N/A",
        },
        {
            "Retriever": "CLIP",
            "Type": "Image-Text Similarity",
            "Medical Specialization": "No (general images)",
            "Input": "Images + Text embeddings",
            "Avg Score": f"{np.mean([r.get('score', 0) for r in clip_results]):.4f}" if clip_results else "N/A",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "retrieval_comparison.csv", index=False)
    return df


def generate_full_comparison_report(report_metrics, qa_metrics, output_dir=None):
    """Generate a comprehensive markdown comparison report."""
    if output_dir is None:
        output_dir = COMPARISON_RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md = "# Model Comparison Results\n\n"
    md += "## Report Generation\n\n"

    if report_metrics:
        report_df = compare_report_generation(report_metrics, output_dir)
        md += report_df.to_markdown(index=False) + "\n\n"

    md += "## QA Performance\n\n"

    if qa_metrics:
        qa_df = compare_qa_performance(qa_metrics, output_dir)
        md += qa_df.to_markdown(index=False) + "\n\n"

    md += "## Key Observations\n\n"
    md += "- **MedGemma** is the primary generation model for both modes.\n"
    md += "- **ColPali** provides visual document retrieval for RAG.\n"
    md += "- **CLIP** serves as a baseline retrieval comparison.\n"
    md += "- RAG-augmented generation is expected to reduce hallucination.\n\n"
    md += "## Limitations\n\n"
    md += "- This is an academic demo, not a clinical diagnostic tool.\n"
    md += "- Evaluation is on a dataset subset.\n"
    md += "- ColPali was not fine-tuned on medical documents.\n"

    with open(output_dir / "comparison_report.md", "w") as f:
        f.write(md)

    logger.info(f"Full comparison report saved to {output_dir / 'comparison_report.md'}")
    return md


def create_comparison_visualizations(report_metrics, qa_metrics, output_dir=None):
    """Create comparison charts."""
    if output_dir is None:
        output_dir = COMPARISON_RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        # Report Generation Comparison
        if report_metrics:
            methods = list(report_metrics.keys())
            bleu_scores = [report_metrics[m].get("bleu4_mean", 0) for m in methods]
            rouge_scores = [report_metrics[m].get("rougeL_mean", 0) for m in methods]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.bar(methods, bleu_scores, color=["#2196F3", "#FF9800", "#4CAF50"][:len(methods)])
            ax1.set_title("BLEU-4 Score by Method")
            ax1.set_ylabel("BLEU-4")
            ax1.set_ylim(0, max(bleu_scores) * 1.3 if bleu_scores else 1)

            ax2.bar(methods, rouge_scores, color=["#2196F3", "#FF9800", "#4CAF50"][:len(methods)])
            ax2.set_title("ROUGE-L Score by Method")
            ax2.set_ylabel("ROUGE-L")
            ax2.set_ylim(0, max(rouge_scores) * 1.3 if rouge_scores else 1)

            plt.tight_layout()
            plt.savefig(output_dir / "report_comparison_chart.png", dpi=150, bbox_inches="tight")
            plt.close()

        # QA Comparison
        if qa_metrics:
            methods = list(qa_metrics.keys())
            f1_scores = [qa_metrics[m].get("f1_mean", 0) for m in methods]
            em_scores = [qa_metrics[m].get("exact_match_rate", 0) for m in methods]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.bar(methods, f1_scores, color=["#2196F3", "#FF9800", "#4CAF50"][:len(methods)])
            ax1.set_title("F1 Score by Method")
            ax1.set_ylabel("F1")

            ax2.bar(methods, em_scores, color=["#2196F3", "#FF9800", "#4CAF50"][:len(methods)])
            ax2.set_title("Exact Match Rate by Method")
            ax2.set_ylabel("EM Rate")

            plt.tight_layout()
            plt.savefig(output_dir / "qa_comparison_chart.png", dpi=150, bbox_inches="tight")
            plt.close()

        logger.info("Comparison charts saved.")
    except ImportError:
        logger.warning("matplotlib not available, skipping charts.")
