"""
CLIP model wrapper for baseline retrieval comparison.
Encodes images and text into a shared embedding space for similarity search.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import CLIP_MODEL_ID, CLIP_TOP_K
except ImportError:
    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    CLIP_TOP_K = 3


class CLIPModel:
    """
    CLIP-based retrieval model for baseline comparison with ColPali.
    Encodes both CXR images and report text for similarity-based retrieval.
    """

    def __init__(self, model_id=None, device=None, top_k=None):
        self.model_id = model_id or CLIP_MODEL_ID
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k or CLIP_TOP_K
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False
        # Index storage
        self.text_embeddings = None
        self.image_embeddings = None
        self.index_metadata = []

    def load(self):
        """Load CLIP model and processor."""
        if self._loaded:
            return

        from transformers import CLIPModel as HFCLIPModel, CLIPProcessor, CLIPTokenizer

        logger.info(f"Loading CLIP: {self.model_id}...")
        self.model = HFCLIPModel.from_pretrained(self.model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id)
        self.model.eval()
        self._loaded = True
        logger.info(f"CLIP loaded on {self.device}")

    def encode_image(self, image):
        """Encode a single image to CLIP embedding."""
        if not self._loaded:
            self.load()
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    def encode_text(self, text):
        """Encode text to CLIP embedding."""
        if not self._loaded:
            self.load()
        # Truncate long text to CLIP's max token length (77)
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    def build_index(self, df, use_images=True, use_text=True):
        """
        Build CLIP embedding index from DataFrame.

        Args:
            df: DataFrame with 'report_text', 'image_path', 'sample_id'
            use_images: Whether to encode CXR images
            use_text: Whether to encode report text
        """
        if not self._loaded:
            self.load()

        from tqdm import tqdm

        self.index_metadata = []
        text_embs = []
        image_embs = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="CLIP indexing"):
            sample_id = row.get("sample_id", f"cxr_{idx:05d}")
            report_text = row.get("report_text", "")
            image_path = row.get("image_path", "")

            meta = {
                "sample_id": sample_id,
                "report_text": report_text,
                "image_path": str(image_path) if image_path else "",
            }
            self.index_metadata.append(meta)

            # Encode text
            if use_text and report_text:
                # CLIP has 77 token limit, use first portion of report
                text_emb = self.encode_text(report_text[:300])
                text_embs.append(text_emb)
            else:
                text_embs.append(np.zeros(512))

            # Encode image
            if use_images and image_path and Path(str(image_path)).exists():
                try:
                    img_emb = self.encode_image(image_path)
                    image_embs.append(img_emb)
                except Exception as e:
                    logger.warning(f"Failed to encode image {image_path}: {e}")
                    image_embs.append(np.zeros(512))
            else:
                image_embs.append(np.zeros(512))

        self.text_embeddings = np.array(text_embs)
        self.image_embeddings = np.array(image_embs)

        logger.info(f"CLIP index built: {len(self.index_metadata)} entries")
        logger.info(f"Text embeddings shape: {self.text_embeddings.shape}")
        logger.info(f"Image embeddings shape: {self.image_embeddings.shape}")

    def search_by_text(self, query, top_k=None, search_in="text"):
        """
        Search the index using a text query.

        Args:
            query: Text query string
            top_k: Number of results
            search_in: "text" to search text embeddings, "image" for image embeddings

        Returns:
            List of dicts with score, sample_id, report_text
        """
        k = top_k or self.top_k
        query_emb = self.encode_text(query)

        if search_in == "text" and self.text_embeddings is not None:
            similarities = np.dot(self.text_embeddings, query_emb)
        elif search_in == "image" and self.image_embeddings is not None:
            similarities = np.dot(self.image_embeddings, query_emb)
        else:
            return []

        top_indices = np.argsort(similarities)[::-1][:k]
        results = []
        for idx in top_indices:
            result = {
                "score": float(similarities[idx]),
                "rank": len(results) + 1,
            }
            result.update(self.index_metadata[idx])
            results.append(result)
        return results

    def search_by_image(self, image, top_k=None, search_in="image"):
        """Search the index using an image query."""
        k = top_k or self.top_k
        query_emb = self.encode_image(image)

        if search_in == "image" and self.image_embeddings is not None:
            similarities = np.dot(self.image_embeddings, query_emb)
        elif search_in == "text" and self.text_embeddings is not None:
            similarities = np.dot(self.text_embeddings, query_emb)
        else:
            return []

        top_indices = np.argsort(similarities)[::-1][:k]
        results = []
        for idx in top_indices:
            result = {"score": float(similarities[idx]), "rank": len(results) + 1}
            result.update(self.index_metadata[idx])
            results.append(result)
        return results

    def get_retrieved_context(self, query, top_k=None):
        """Get concatenated report text from top search results."""
        results = self.search_by_text(query, top_k)
        parts = []
        for i, r in enumerate(results):
            if r.get("report_text"):
                parts.append(f"[Report {i+1}] {r['report_text']}")
        return "\n\n".join(parts), results

    def save_index(self, path):
        """Save the CLIP index to disk."""
        data = {
            "text_embeddings": self.text_embeddings,
            "image_embeddings": self.image_embeddings,
            "metadata": self.index_metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"CLIP index saved to {path}")

    def load_index(self, path):
        """Load a previously saved CLIP index."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.text_embeddings = data["text_embeddings"]
        self.image_embeddings = data["image_embeddings"]
        self.index_metadata = data["metadata"]
        logger.info(f"CLIP index loaded: {len(self.index_metadata)} entries")

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CLIP unloaded.")


if __name__ == "__main__":
    clip = CLIPModel()
    clip.load()
    emb = clip.encode_text("chest x-ray with pleural effusion")
    print(f"Text embedding shape: {emb.shape}")
    clip.unload()
