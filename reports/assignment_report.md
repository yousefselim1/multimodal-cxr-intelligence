# DSAI 413 — Assignment 2: Short Report

## Multi-Modal Chest X-Ray Intelligence System (Dual-Mode: Report Generation & QA)

---

## 1. Introduction

### Problem Statement
Modern medical AI systems must integrate both image understanding and language reasoning. Chest X-rays (CXR) are the most common radiological examination, requiring both automated report generation and clinical question answering — two distinct but complementary tasks.

### Scope
This project implements a dual-mode multi-modal system:
- **Mode 1 — Report Generation:** CXR image → structured radiology report
- **Mode 2 — QA:** Clinical question + optional CXR image → grounded answer using RAG

### Medical Context
Chest X-ray interpretation involves systematic assessment of:
- **Heart and mediastinum:** Size, silhouette, contour
- **Lungs and airways:** Parenchyma, volumes, opacities, consolidation
- **Pleura:** Effusions, pneumothorax
- **Bones and soft tissues:** Fractures, abnormalities
- **Support devices:** Tubes, lines, catheters, pacemakers

Common findings include cardiomegaly, pleural effusion, pneumothorax, atelectasis, consolidation (pneumonia), pulmonary edema, and lung opacities.

---

## 2. System Architecture

### Overall Design

```
┌─────────────────────────────────────────────────┐
│           DUAL-MODE CXR INTELLIGENCE            │
├───────────────────┬─────────────────────────────┤
│                   │                             │
│  MODE 1           │  MODE 2                     │
│  Report Gen       │  RAG-based QA               │
│                   │                             │
│  Image            │  Question + Image           │
│    ↓              │    ↓           ↓            │
│  MedGemma         │  ColPali    MedGemma        │
│    ↓              │  Retriever  Generator       │
│  Structured       │    ↓           ↓            │
│  Report           │  Context → Grounded Answer  │
│                   │                             │
└───────────────────┴─────────────────────────────┘
```

### Key Design Decisions

1. **ColPali retrieves rendered report pages, not raw X-rays** — ColPali is a document retrieval model designed for visually rich documents. We convert report text into formatted document page images for indexing.

2. **MedGemma serves dual roles** — As a medical VLM, it handles both image-to-text report generation and context-grounded answer generation.

3. **CLIP as baseline comparison** — Provides a text-image similarity baseline for retrieval comparison against ColPali.

4. **4-bit quantization** — Enables MedGemma to run on 16GB GPU (Kaggle T4).

---

## 3. Models & Justification

### MedGemma 1.5 4B IT
- **Architecture:** Gemma 3 decoder-only transformer with medical SigLIP image encoder
- **Why suitable:** Pre-trained on MIMIC-CXR, chest X-rays, medical text, and medical Q&A pairs
- **Capabilities:** Image-text-to-text generation, CXR report generation, medical VQA
- **Limitations:** Not clinically validated; sensitive to prompt engineering; 4-bit quantization may reduce quality

### ColPali v1.2
- **Architecture:** PaliGemma-3B based visual document retrieval with late interaction
- **Why suitable:** Processes documents as images, preserving layout and visual structure
- **Capabilities:** Visual document retrieval without OCR, multi-vector embeddings
- **Limitations:** Designed for general documents, not medical images; may need fine-tuning for medical reports; high storage for multi-vector index

### CLIP ViT-B/32
- **Architecture:** Contrastive image-text pre-training with ViT image encoder
- **Why suitable:** Lightweight baseline for image-text similarity matching
- **Capabilities:** Zero-shot image-text retrieval, embedding-based similarity search
- **Limitations:** 77-token text limit; not medical-domain specialized; single-vector representation less expressive than ColPali's multi-vector

---

## 4. Dataset

### MIMIC-CXR Dataset
- **Source:** Kaggle (simhadrisadaram/mimic-cxr-dataset)
- **Content:** Chest X-ray images with associated radiology reports
- **Usage:** The "text" column contains the report used as ground truth
- **Subset:** ~1,000 samples used for this academic demo

### QA Dataset Creation

**Methodology:** Hybrid rule-based template generation

**Step 1 — Finding Extraction:** Regex patterns detect 10 medical finding categories:
- cardiomegaly, pleural effusion, pneumothorax, atelectasis, consolidation, edema, lung opacity, support devices, fracture, no finding

**Step 2 — Template QA Generation:** For each detected finding, 7 question types are generated:
1. **Presence questions** — "Is there evidence of {finding}?"
2. **Severity questions** — "What is the severity of {finding}?"
3. **Location questions** — "Where is the {finding} located?"
4. **Summary questions** — "What are the main findings?"
5. **Device questions** — "Are there support devices visible?"
6. **Normal study questions** — "Is this chest X-ray normal?"
7. **Comparison questions** — "Does this show both {finding_A} and {finding_B}?"

**Step 3 — Validation:** Manual review of sample QA pairs for clinical accuracy.

**Output Schema:**
| Field | Description |
|---|---|
| sample_id | Unique sample identifier |
| image_path | Path to CXR image |
| report_text | Source radiology report |
| question | Generated question |
| answer | Grounded answer citing report evidence |
| category | Medical category |
| evidence_sentence | Supporting text from report |
| generation_method | "template_rule_based" |

**Expected yield:** ~5-10 QA pairs per report = ~5,000-10,000 total pairs

---

## 5. Implementation Details

### Environment
- **GPU:** Kaggle T4 x2 (32GB VRAM)
- **Quantization:** 4-bit NF4 for MedGemma (via bitsandbytes)
- **Framework:** HuggingFace Transformers, Byaldi (ColPali), Gradio

### Prompt Engineering
- Structured report prompt forces systematic section-by-section analysis
- QA prompt includes retrieved context and grounding instructions
- Medical disclaimer integrated into system prompts

### Knowledge Base Construction
- Report text rendered as 896×896 PNG document images
- ColPali indexes these rendered pages for visual retrieval
- CLIP encodes report text (first 300 chars) for text-based retrieval

---

## 6. Results & Model Comparison

### Report Generation Quality

| Method | BLEU-4 | ROUGE-L | Notes |
|---|---|---|---|
| MedGemma Direct | TBD | TBD | Image-only generation |
| MedGemma + ColPali RAG | TBD | TBD | RAG-augmented |
| MedGemma + CLIP RAG | TBD | TBD | CLIP-augmented baseline |

### QA Performance

| Method | F1 Score | Exact Match | Notes |
|---|---|---|---|
| MedGemma Direct | TBD | TBD | No retrieval |
| ColPali RAG + MedGemma | TBD | TBD | ColPali retrieval |
| CLIP RAG + MedGemma | TBD | TBD | CLIP retrieval |

### Retrieval Comparison

| Retriever | Type | Strengths | Weaknesses |
|---|---|---|---|
| ColPali | Visual document | Layout-aware, OCR-free | Not medical-specialized |
| CLIP | Text-image similarity | Fast, lightweight | 77-token limit, single-vector |

*(Results to be filled after running evaluation on Kaggle GPU)*

---

## 7. Demo Application

The Gradio-based demo provides:
- **Tab 1 — Report Generation:** Upload CXR → click "Generate" → view structured report
- **Tab 2 — QA Mode:** Upload CXR + type question → toggle RAG → view grounded answer
- Medical disclaimer banner
- Model status indicator
- Sample questions for quick testing

---

## 8. Limitations & Future Work

### Limitations
- Academic demo, not clinically validated
- Subset of MIMIC-CXR dataset (not full dataset)
- ColPali not fine-tuned on medical documents
- QA dataset generated from templates (not human-annotated)
- 4-bit quantization may reduce generation quality
- Single-image analysis only (no longitudinal comparison)

### Future Work
- Fine-tune ColPali on medical report pages
- Add CheXpert-style label extraction for clinical agreement evaluation
- Implement multi-turn conversation
- Add confidence scoring and uncertainty estimation
- Expand QA dataset with LLM-augmented generation
- Clinical validation with domain experts

---

## 9. Safety & Ethics

- This system is for educational/research purposes only
- Not intended for independent clinical diagnosis
- All outputs should be verified by qualified healthcare professionals
- The system clearly states when evidence is insufficient
- Medical disclaimer displayed prominently in the application

---

## 10. References

1. Sellergren et al. "MedGemma 1.5 Technical Report." arXiv:2604.05081 (2026).
2. Faysse et al. "ColPali: Efficient Document Retrieval with Vision Language Models." arXiv:2407.01449 (2024).
3. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML (2021).
4. Johnson et al. "MIMIC-CXR Database." PhysioNet (2024).
5. HuggingFace Multimodal RAG Cookbook.
6. Radiologyassistant.nl — Chest X-Ray Basic Interpretation.
