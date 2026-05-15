"""
Mode 1: Report Generation Pipeline.
Takes a chest X-ray image and generates a structured radiology report using MedGemma.
Optionally augments with RAG-retrieved context from similar reports.
"""

import time
import pandas as pd
from PIL import Image
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import REPORT_GEN_RESULTS_DIR
except ImportError:
    REPORT_GEN_RESULTS_DIR = Path("./results/report_generation")


class ReportGenerationPipeline:
    """
    Chest X-ray Report Generation Pipeline.
    
    Supports two modes:
    1. Direct: MedGemma generates report from image alone
    2. RAG-augmented: Retrieves similar reports first, then generates with context
    """

    def __init__(self, medgemma_model=None, colpali_retriever=None, clip_model=None):
        self.medgemma = medgemma_model
        self.colpali = colpali_retriever
        self.clip = clip_model

    def generate_report_direct(self, image, max_new_tokens=None):
        """
        Generate a report directly from the image using MedGemma only.
        
        Args:
            image: PIL Image or path to CXR image
            max_new_tokens: Max generation length
        
        Returns:
            dict with 'report', 'method', 'time_seconds'
        """
        start = time.time()
        report = self.medgemma.generate_report(image, max_new_tokens)
        elapsed = time.time() - start

        return {
            "report": report,
            "method": "medgemma_direct",
            "time_seconds": elapsed,
        }

    def generate_report_with_rag(self, image, retriever_type="colpali",
                                  query=None, top_k=3, max_new_tokens=None):
        """
        Generate a report augmented with retrieved similar reports.
        
        Args:
            image: PIL Image or path to CXR image
            retriever_type: "colpali" or "clip"
            query: Optional text query for retrieval (defaults to generic)
            top_k: Number of reports to retrieve
            max_new_tokens: Max generation length
        
        Returns:
            dict with 'report', 'method', 'retrieved_context', 'time_seconds'
        """
        start = time.time()

        if query is None:
            query = "chest x-ray findings radiology report"

        # Retrieve similar reports
        context = ""
        results = []
        if retriever_type == "colpali" and self.colpali is not None:
            context, results = self.colpali.get_retrieved_context(query, top_k)
        elif retriever_type == "clip" and self.clip is not None:
            context, results = self.clip.get_retrieved_context(query, top_k)

        # Generate report with context
        if context:
            augmented_prompt = (
                f"Use the following similar radiology reports as reference:\n\n"
                f"{context}\n\n"
                f"Now analyze the provided chest X-ray image and generate a "
                f"structured radiology report. Use the reference reports for "
                f"style and terminology, but base your findings on the actual image."
            )
            report = self.medgemma.answer_question(
                question=augmented_prompt, image=image, max_new_tokens=max_new_tokens
            )
        else:
            report = self.medgemma.generate_report(image, max_new_tokens)

        elapsed = time.time() - start

        return {
            "report": report,
            "method": f"rag_{retriever_type}",
            "retrieved_context": context,
            "retrieved_results": results,
            "time_seconds": elapsed,
        }

    def batch_generate(self, df, method="direct", output_dir=None, max_samples=None):
        """
        Generate reports for a batch of samples.
        
        Args:
            df: DataFrame with 'image_path' and 'sample_id' columns
            method: "direct" or "rag_colpali" or "rag_clip"
            output_dir: Directory to save results
            max_samples: Limit number of samples
        
        Returns:
            DataFrame with generated reports
        """
        if output_dir is None:
            output_dir = REPORT_GEN_RESULTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if max_samples:
            df = df.head(max_samples)

        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating reports ({method})"):
            image_path = row.get("image_path", "")
            sample_id = row.get("sample_id", f"cxr_{idx:05d}")
            ground_truth = row.get("report_text", "")

            try:
                if not image_path or not Path(str(image_path)).exists():
                    logger.warning(f"Image not found: {image_path}, skipping {sample_id}")
                    continue

                if method == "direct":
                    result = self.generate_report_direct(image_path)
                elif method == "rag_colpali":
                    result = self.generate_report_with_rag(image_path, "colpali")
                elif method == "rag_clip":
                    result = self.generate_report_with_rag(image_path, "clip")
                else:
                    result = self.generate_report_direct(image_path)

                result["sample_id"] = sample_id
                result["ground_truth"] = ground_truth
                result["image_path"] = str(image_path)
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {sample_id}: {e}")
                results.append({
                    "sample_id": sample_id, "report": f"ERROR: {e}",
                    "method": method, "ground_truth": ground_truth,
                    "image_path": str(image_path), "time_seconds": 0,
                })

        results_df = pd.DataFrame(results)
        output_path = output_dir / f"reports_{method}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results_df)} reports to {output_path}")

        return results_df
