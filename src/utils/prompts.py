"""
Prompt templates for the Multi-Modal Chest X-Ray Intelligence System.
"""

# ─── Report Generation Prompts ───────────────────────────────────────────────

REPORT_GENERATION_SYSTEM = """You are an expert radiologist assistant. Analyze the provided chest X-ray image and generate a structured radiology report. Be precise, clinical, and evidence-based. Only describe what you can observe."""

REPORT_GENERATION_PROMPT = """Analyze this chest X-ray image and generate a structured radiology report with the following sections:

**FINDINGS:**
Describe all observable findings systematically:
- Heart and mediastinum (heart size, mediastinal contour)
- Lungs and airways (parenchyma, volumes, opacities)
- Pleura (effusions, pneumothorax)
- Bones and soft tissues (fractures, abnormalities)
- Support devices (tubes, lines, catheters if present)

**IMPRESSION:**
Provide a concise clinical impression summarizing the key findings.

**PATHOLOGIES DETECTED:**
List each pathology detected with severity (none/mild/moderate/severe):
- Cardiomegaly: 
- Pleural Effusion: 
- Pneumothorax: 
- Atelectasis: 
- Consolidation/Pneumonia: 
- Pulmonary Edema: 
- Lung Opacity: 
- Support Devices: 
- Fractures: 

Be systematic and thorough. If a finding is absent, explicitly state it is not observed."""

# ─── QA Prompts ──────────────────────────────────────────────────────────────

QA_SYSTEM = """You are a medical AI assistant specializing in chest X-ray interpretation. Answer questions based on the provided context and image. Always ground your answers in the evidence provided. If the evidence is insufficient, say so clearly."""

QA_WITH_CONTEXT_PROMPT = """Based on the following retrieved medical context and the chest X-ray image, answer the question.

**Retrieved Context:**
{context}

**Question:** {question}

**Instructions:**
1. Answer the question directly and concisely.
2. Cite evidence from the retrieved context to support your answer.
3. If the context does not contain enough information, state that clearly.
4. Do not make unsupported medical claims.

**Answer:**"""

QA_WITHOUT_CONTEXT_PROMPT = """Based on the chest X-ray image provided, answer the following question.

**Question:** {question}

**Instructions:**
1. Answer based only on what you can observe in the image.
2. Be precise and clinical in your response.
3. If you cannot determine the answer from the image alone, state that clearly.

**Answer:**"""

# ─── QA Dataset LLM Generation Prompt ────────────────────────────────────────

QA_GENERATION_PROMPT = """Given the following radiology report, generate 3 clinically relevant question-answer pairs. Each answer must be directly supported by the report text.

**Radiology Report:**
{report_text}

Generate exactly 3 QA pairs in this format:
Q1: [question]
A1: [answer grounded in the report]

Q2: [question]
A2: [answer grounded in the report]

Q3: [question]
A3: [answer grounded in the report]

Make questions diverse: include presence/absence, severity, location, and clinical significance questions."""
