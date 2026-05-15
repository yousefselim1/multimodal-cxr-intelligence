"""
MedGemma 1.5 4B model wrapper.
Handles loading (with 4-bit quantization), report generation, and QA answer generation.
Designed to run on Kaggle T4x2 GPU (32GB VRAM).
"""

import torch
from PIL import Image
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.config import MEDGEMMA_MODEL_ID, HF_TOKEN, MEDGEMMA_MAX_NEW_TOKENS, MEDGEMMA_QUANTIZATION
    from src.utils.prompts import (
        REPORT_GENERATION_SYSTEM, REPORT_GENERATION_PROMPT,
        QA_SYSTEM, QA_WITH_CONTEXT_PROMPT, QA_WITHOUT_CONTEXT_PROMPT,
    )
except ImportError:
    MEDGEMMA_MODEL_ID = "google/medgemma-1.5-4b-it"
    HF_TOKEN = ""
    MEDGEMMA_MAX_NEW_TOKENS = 1024
    MEDGEMMA_QUANTIZATION = "4bit"
    REPORT_GENERATION_SYSTEM = "You are an expert radiologist assistant."
    REPORT_GENERATION_PROMPT = "Analyze this chest X-ray and generate a structured report."
    QA_SYSTEM = "You are a medical AI assistant."
    QA_WITH_CONTEXT_PROMPT = "Context: {context}\nQuestion: {question}\nAnswer:"
    QA_WITHOUT_CONTEXT_PROMPT = "Question: {question}\nAnswer:"


class MedGemmaModel:
    """Wrapper for MedGemma 1.5 4B model with quantization support."""

    def __init__(self, model_id=None, quantization=None, device=None, hf_token=None):
        self.model_id = model_id or MEDGEMMA_MODEL_ID
        self.quantization = quantization or MEDGEMMA_QUANTIZATION
        self.hf_token = hf_token or HF_TOKEN
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        """Load MedGemma model with quantization."""
        if self._loaded:
            logger.info("Model already loaded.")
            return

        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info(f"Loading {self.model_id} with {self.quantization} quantization...")

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "token": self.hf_token,
        }

        if self.quantization == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        elif self.quantization == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto"

        self.processor = AutoProcessor.from_pretrained(self.model_id, token=self.hf_token)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        self._loaded = True
        logger.info(f"MedGemma loaded successfully on {self.device}")

    def _generate(self, messages, max_new_tokens=None):
        """Internal generation method."""
        if not self._loaded:
            self.load()

        max_tokens = max_new_tokens or MEDGEMMA_MAX_NEW_TOKENS

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        decoded = self.processor.decode(output[0][input_len:], skip_special_tokens=True)
        return decoded.strip()

    def generate_report(self, image, max_new_tokens=None):
        """
        Generate a structured radiology report from a chest X-ray image.

        Args:
            image: PIL Image or path to image file
            max_new_tokens: Maximum tokens to generate

        Returns:
            str: Generated structured report
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": REPORT_GENERATION_PROMPT},
            ]}
        ]
        return self._generate(messages, max_new_tokens)

    def answer_question(self, question, image=None, context=None, max_new_tokens=None):
        """
        Answer a medical question with optional image and retrieved context.

        Args:
            question: The question to answer
            image: Optional PIL Image or path
            context: Optional retrieved context string
            max_new_tokens: Maximum tokens to generate

        Returns:
            str: Generated answer
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        content = []
        if image is not None:
            content.append({"type": "image", "image": image})

        if context:
            prompt = QA_WITH_CONTEXT_PROMPT.format(context=context, question=question)
        else:
            prompt = QA_WITHOUT_CONTEXT_PROMPT.format(question=question)

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        return self._generate(messages, max_new_tokens)

    def generate_qa_from_report(self, report_text, max_new_tokens=512):
        """Use MedGemma to generate QA pairs from a report (LLM-augmented QA)."""
        from src.utils.prompts import QA_GENERATION_PROMPT

        prompt = QA_GENERATION_PROMPT.format(report_text=report_text)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self._generate(messages, max_new_tokens)

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MedGemma unloaded, GPU memory freed.")


if __name__ == "__main__":
    model = MedGemmaModel()
    model.load()
    print("MedGemma loaded successfully!")
    # Quick test with a text-only query
    answer = model.answer_question("What is cardiomegaly?")
    print(f"Answer: {answer}")
    model.unload()
