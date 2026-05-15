# 🏥 Multi-Modal Chest X-Ray Intelligence System

**DSAI 413 — Assignment 2: Multi-Modal Chest X-Ray Intelligence System (Dual-Mode: Report Generation & QA)**

> ⚠️ **Disclaimer:** This is an academic demo project. It has NOT been clinically validated and should NOT be used for medical diagnosis or treatment decisions.

---

## Overview

A multi-modal AI system for chest X-ray analysis with two independent modes:

1. **Report Generation Mode** — Upload a chest X-ray image → receive a structured radiology report
2. **QA Mode** — Ask clinical questions about chest X-rays → receive grounded, evidence-based answers using RAG (Retrieval-Augmented Generation)

## Models Used

| Model | Role | Type |
|---|---|---|
| **MedGemma 1.5 4B** | Report generator + QA answer generator | Vision-Language Model (medical) |
| **ColPali v1.2** | Document retriever for RAG | Visual Document Retrieval |
| **CLIP ViT-B/32** | Baseline retriever for comparison | Image-Text Similarity |

## Architecture

```
Mode 1 (Report Generation):
  CXR Image → MedGemma → Structured Report

Mode 2 (RAG QA):
  Question → ColPali/CLIP Retriever → Retrieved Context
  Question + Context + CXR Image → MedGemma → Grounded Answer
```

## Dataset

- **Source:** [MIMIC-CXR Dataset (Kaggle)](https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset)
- **QA Dataset:** Generated from report text using rule-based templates (7 question types)
- See `reports/assignment_report.md` for detailed QA dataset creation methodology

## Quick Start

### Prerequisites
- Python 3.9+
- NVIDIA GPU with 16GB+ VRAM (or use Kaggle/Colab)
- HuggingFace account with MedGemma access

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd Assignment_2_MulitiMedia

# Install dependencies
pip install -r requirements.txt

# Set up credentials
cp .env.example .env
# Edit .env with your HF_TOKEN and Kaggle credentials
```

### Run the Demo

```bash
python app.py
```

This launches a Gradio web interface at `http://localhost:7860`.

### Running on Kaggle

1. Upload the code to a Kaggle notebook
2. Enable GPU (T4 x2)
3. Set HuggingFace token as a Kaggle secret
4. Run the notebook cells

## Project Structure

```
├── app.py                      # Gradio demo application
├── requirements.txt            # Python dependencies
├── src/
│   ├── config.py               # Global configuration
│   ├── data/
│   │   ├── load_dataset.py     # Dataset download & loading
│   │   ├── preprocess.py       # Data cleaning & splitting
│   │   └── create_qa_dataset.py # QA pair generation
│   ├── models/
│   │   ├── medgemma_model.py   # MedGemma wrapper
│   │   ├── colpali_retriever.py # ColPali retriever
│   │   └── clip_model.py       # CLIP baseline
│   ├── pipelines/
│   │   ├── report_generation.py # Mode 1 pipeline
│   │   └── qa_rag.py           # Mode 2 pipeline
│   ├── evaluation/
│   │   ├── evaluate_reports.py # BLEU/ROUGE metrics
│   │   ├── evaluate_qa.py      # QA accuracy metrics
│   │   └── compare_models.py   # Model comparison
│   └── utils/
│       ├── prompts.py          # Prompt templates
│       └── render_reports.py   # Report page rendering
├── data/                       # Dataset files (gitignored)
├── results/                    # Evaluation results
├── reports/
│   └── assignment_report.md    # Assignment report
└── notebooks/                  # Jupyter notebooks
```

## Evaluation Metrics

### Report Generation
- BLEU-4 score
- ROUGE-1, ROUGE-2, ROUGE-L
- Manual clinical review

### QA Performance
- Token-level F1 score
- Exact Match rate
- Per-category breakdown

## References

- MedGemma: [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
- ColPali: [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2)
- CLIP: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- MIMIC-CXR Dataset: [Kaggle](https://www.kaggle.com/datasets/simhadrisadaram/mimic-cxr-dataset)
- HuggingFace Multimodal RAG Cookbook: [Link](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms)

## License

Academic use only. MedGemma is governed by the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).
