"""
Mode 2: RAG-based QA Pipeline.
Takes a question (+ optional CXR image), retrieves relevant context using
ColPali or CLIP, and generates a grounded answer using MedGemma.
"""

import time
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import QA_RESULTS_DIR
except ImportError:
    QA_RESULTS_DIR = Path("./results/qa_results")


class QARAGPipeline:
    """
    RAG-based Question Answering Pipeline for chest X-rays.
    
    Flow: Question → Retrieve relevant reports → Generate grounded answer
    """

    def __init__(self, medgemma_model=None, colpali_retriever=None, clip_model=None):
        self.medgemma = medgemma_model
        self.colpali = colpali_retriever
        self.clip = clip_model

    def answer_with_rag(self, question, image=None, retriever_type="colpali",
                         top_k=3, max_new_tokens=512):
        """
        Answer a question using RAG: retrieve context then generate.
        
        Args:
            question: The medical question
            image: Optional CXR image (PIL or path)
            retriever_type: "colpali" or "clip"
            top_k: Number of documents to retrieve
            max_new_tokens: Max answer length
        
        Returns:
            dict with 'answer', 'context', 'retrieved_results', 'method', 'time_seconds'
        """
        start = time.time()

        # Step 1: Retrieve relevant context
        context = ""
        results = []
        if retriever_type == "colpali" and self.colpali is not None:
            context, results = self.colpali.get_retrieved_context(question, top_k)
        elif retriever_type == "clip" and self.clip is not None:
            context, results = self.clip.get_retrieved_context(question, top_k)

        # Step 2: Generate answer with MedGemma
        answer = self.medgemma.answer_question(
            question=question, image=image, context=context,
            max_new_tokens=max_new_tokens,
        )

        elapsed = time.time() - start

        return {
            "answer": answer,
            "context": context,
            "retrieved_results": results,
            "method": f"rag_{retriever_type}",
            "time_seconds": elapsed,
        }

    def answer_direct(self, question, image=None, max_new_tokens=512):
        """
        Answer a question directly with MedGemma (no retrieval).
        
        Args:
            question: The medical question
            image: Optional CXR image
            max_new_tokens: Max answer length
        
        Returns:
            dict with 'answer', 'method', 'time_seconds'
        """
        start = time.time()
        answer = self.medgemma.answer_question(
            question=question, image=image, max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - start

        return {
            "answer": answer,
            "method": "medgemma_direct",
            "context": "",
            "retrieved_results": [],
            "time_seconds": elapsed,
        }

    def batch_answer(self, qa_df, method="rag_colpali", output_dir=None, max_samples=None):
        """
        Answer a batch of QA pairs.
        
        Args:
            qa_df: DataFrame with 'question', 'answer' (ground truth), 'image_path'
            method: "rag_colpali", "rag_clip", or "direct"
            output_dir: Directory to save results
            max_samples: Limit number of samples
        
        Returns:
            DataFrame with generated answers
        """
        if output_dir is None:
            output_dir = QA_RESULTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if max_samples:
            qa_df = qa_df.head(max_samples)

        results = []
        for idx, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc=f"QA ({method})"):
            question = row.get("question", "")
            ground_truth = row.get("answer", "")
            image_path = row.get("image_path", "")
            sample_id = row.get("sample_id", "")
            category = row.get("category", "")

            try:
                image = None
                if image_path and Path(str(image_path)).exists():
                    image = image_path

                if method == "direct":
                    result = self.answer_direct(question, image)
                elif method == "rag_colpali":
                    result = self.answer_with_rag(question, image, "colpali")
                elif method == "rag_clip":
                    result = self.answer_with_rag(question, image, "clip")
                else:
                    result = self.answer_direct(question, image)

                result["sample_id"] = sample_id
                result["question"] = question
                result["ground_truth_answer"] = ground_truth
                result["category"] = category
                # Remove large objects for CSV storage
                result.pop("retrieved_results", None)
                results.append(result)

            except Exception as e:
                logger.error(f"Error on {sample_id}: {e}")
                results.append({
                    "sample_id": sample_id, "question": question,
                    "answer": f"ERROR: {e}", "ground_truth_answer": ground_truth,
                    "method": method, "category": category, "time_seconds": 0,
                })

        results_df = pd.DataFrame(results)
        output_path = output_dir / f"qa_results_{method}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(results_df)} QA results to {output_path}")

        return results_df
