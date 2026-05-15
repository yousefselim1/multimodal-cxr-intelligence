"""
ColPali document retriever for RAG-based QA.
Uses Byaldi wrapper to index rendered report pages and retrieve relevant documents.
"""

import os
from PIL import Image
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import COLPALI_MODEL_ID, COLPALI_INDEX_NAME, COLPALI_TOP_K, RENDERED_REPORTS_DIR
except ImportError:
    COLPALI_MODEL_ID = "vidore/colpali-v1.2"
    COLPALI_INDEX_NAME = "cxr_report_index"
    COLPALI_TOP_K = 3
    RENDERED_REPORTS_DIR = Path("./data/rendered_reports")


class ColPaliRetriever:
    """
    ColPali-based document retriever using Byaldi.
    Indexes rendered report page images and retrieves relevant pages for queries.
    """

    def __init__(self, model_id=None, index_name=None, top_k=None):
        self.model_id = model_id or COLPALI_MODEL_ID
        self.index_name = index_name or COLPALI_INDEX_NAME
        self.top_k = top_k or COLPALI_TOP_K
        self.model = None
        self._loaded = False
        self._indexed = False
        self.doc_id_to_info = {}

    def load(self):
        """Load ColPali model from pretrained checkpoint."""
        if self._loaded:
            logger.info("ColPali already loaded.")
            return

        from byaldi import RAGMultiModalModel
        logger.info(f"Loading ColPali model: {self.model_id}...")
        self.model = RAGMultiModalModel.from_pretrained(self.model_id)
        self._loaded = True
        logger.info("ColPali loaded successfully.")

    def load_index(self, index_path=None):
        """Load a previously built index."""
        if not self._loaded:
            self.load()

        index_name = index_path or self.index_name
        try:
            from byaldi import RAGMultiModalModel
            self.model = RAGMultiModalModel.from_index(index_name)
            self._indexed = True
            
            # Load document mapping if exists
            mapping_path = Path(f"{index_name}_mapping.json")
            if mapping_path.exists():
                with open(mapping_path, "r") as f:
                    self.doc_id_to_info = json.load(f)
            
            logger.info(f"Loaded index: {index_name}")
        except Exception as e:
            logger.warning(f"Could not load index: {e}")

    def build_index(self, report_pages_dir=None, df=None, overwrite=True):
        """
        Build a ColPali index from rendered report page images.

        Args:
            report_pages_dir: Directory containing rendered report PNG images
            df: Optional DataFrame with sample_id and report_text for mapping
            overwrite: Whether to overwrite existing index
        """
        if not self._loaded:
            self.load()

        if report_pages_dir is None:
            report_pages_dir = RENDERED_REPORTS_DIR
        report_pages_dir = Path(report_pages_dir)

        if not report_pages_dir.exists() or not list(report_pages_dir.glob("*.png")):
            raise FileNotFoundError(
                f"No rendered report pages found in {report_pages_dir}. "
                "Run render_reports.render_all_reports() first."
            )

        logger.info(f"Building ColPali index from {report_pages_dir}...")
        self.model.index(
            input_path=str(report_pages_dir),
            index_name=self.index_name,
            store_collection_with_index=False,
            overwrite=overwrite,
        )
        self._indexed = True

        # Build document ID to sample info mapping
        if df is not None:
            image_files = sorted(report_pages_dir.glob("*.png"))
            for doc_id, img_file in enumerate(image_files):
                sample_id = img_file.stem.replace("_report", "")
                row = df[df["sample_id"] == sample_id]
                if len(row) > 0:
                    self.doc_id_to_info[str(doc_id)] = {
                        "sample_id": sample_id,
                        "report_text": row.iloc[0].get("report_text", ""),
                        "image_path": row.iloc[0].get("image_path", ""),
                        "rendered_path": str(img_file),
                    }

            # Save mapping
            mapping_path = Path(f"{self.index_name}_mapping.json")
            with open(mapping_path, "w") as f:
                json.dump(self.doc_id_to_info, f, indent=2)
            logger.info(f"Saved document mapping ({len(self.doc_id_to_info)} docs)")

        logger.info(f"Index built: {self.index_name}")

    def search(self, query, top_k=None):
        """
        Search the index for relevant report pages.

        Args:
            query: Text query string
            top_k: Number of results to return

        Returns:
            List of dicts with: doc_id, page_num, score, sample_id, report_text, rendered_path
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        k = top_k or self.top_k
        raw_results = self.model.search(query, k=k)

        results = []
        for r in raw_results:
            doc_id = str(r.get("doc_id", r.get("doc_id", "")))
            result = {
                "doc_id": doc_id,
                "page_num": r.get("page_num", 1),
                "score": r.get("score", 0.0),
            }
            # Add info from mapping
            if doc_id in self.doc_id_to_info:
                result.update(self.doc_id_to_info[doc_id])
            results.append(result)

        return results

    def get_retrieved_context(self, query, top_k=None):
        """
        Search and return concatenated report text from top results.

        Args:
            query: Text query
            top_k: Number of results

        Returns:
            Tuple of (context_string, results_list)
        """
        results = self.search(query, top_k)
        context_parts = []
        for i, r in enumerate(results):
            report_text = r.get("report_text", "")
            if report_text:
                context_parts.append(f"[Report {i+1}] {report_text}")

        context = "\n\n".join(context_parts)
        return context, results

    def get_retrieved_images(self, query, top_k=None):
        """
        Search and return the rendered report page images.

        Returns:
            List of PIL Images of retrieved report pages
        """
        results = self.search(query, top_k)
        images = []
        for r in results:
            rendered_path = r.get("rendered_path", "")
            if rendered_path and Path(rendered_path).exists():
                images.append(Image.open(rendered_path).convert("RGB"))
        return images, results

    def unload(self):
        """Free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        self._indexed = False
        logger.info("ColPali unloaded.")


if __name__ == "__main__":
    retriever = ColPaliRetriever()
    retriever.load()
    print("ColPali loaded successfully!")
    retriever.unload()
