"""
Multi-Modal Chest X-Ray Intelligence System — Gradio Demo Application.
Dual-mode: Report Generation & RAG-based QA.

⚠️ DISCLAIMER: This is an academic demo. Not for clinical use.
"""

import gradio as gr
from PIL import Image
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Global Model State ─────────────────────────────────────────────────────
medgemma = None
colpali = None
clip_model = None
models_loaded = False


def load_models():
    """Load all models (called once on startup)."""
    global medgemma, colpali, clip_model, models_loaded

    if models_loaded:
        return "Models already loaded."

    status_parts = []

    # Load MedGemma
    try:
        from src.models.medgemma_model import MedGemmaModel
        medgemma = MedGemmaModel()
        medgemma.load()
        status_parts.append("✅ MedGemma loaded")
    except Exception as e:
        status_parts.append(f"❌ MedGemma failed: {e}")
        logger.error(f"MedGemma load error: {e}")

    # Load ColPali
    try:
        from src.models.colpali_retriever import ColPaliRetriever
        colpali = ColPaliRetriever()
        colpali.load()
        # Try to load existing index
        try:
            colpali.load_index()
            status_parts.append("✅ ColPali loaded (with index)")
        except Exception:
            status_parts.append("✅ ColPali loaded (no index)")
    except Exception as e:
        status_parts.append(f"❌ ColPali failed: {e}")
        logger.error(f"ColPali load error: {e}")

    # Load CLIP
    try:
        from src.models.clip_model import CLIPModel
        clip_model = CLIPModel()
        clip_model.load()
        status_parts.append("✅ CLIP loaded")
    except Exception as e:
        status_parts.append(f"❌ CLIP failed: {e}")
        logger.error(f"CLIP load error: {e}")

    models_loaded = True
    status = "\n".join(status_parts)
    logger.info(f"Model loading complete:\n{status}")
    return status


def generate_report(image):
    """Mode 1: Generate structured report from CXR image."""
    if medgemma is None:
        return "⚠️ MedGemma not loaded. Click 'Load Models' first."

    if image is None:
        return "⚠️ Please upload a chest X-ray image."

    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        report = medgemma.generate_report(image)
        return report
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return f"❌ Error generating report: {e}"


def generate_report_rag(image):
    """Mode 1 with RAG: Generate report augmented with retrieved context."""
    if medgemma is None:
        return "⚠️ MedGemma not loaded.", ""

    if image is None:
        return "⚠️ Please upload a chest X-ray image.", ""

    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        from src.pipelines.report_generation import ReportGenerationPipeline
        pipeline = ReportGenerationPipeline(medgemma, colpali, clip_model)
        result = pipeline.generate_report_with_rag(image, retriever_type="colpali")

        report = result.get("report", "No report generated.")
        context = result.get("retrieved_context", "No context retrieved.")

        return report, context
    except Exception as e:
        logger.error(f"RAG report error: {e}")
        return f"❌ Error: {e}", ""


def answer_question(image, question, use_rag, retriever_choice):
    """Mode 2: Answer a question about the CXR."""
    if medgemma is None:
        return "⚠️ MedGemma not loaded. Click 'Load Models' first.", ""

    if not question:
        return "⚠️ Please enter a question.", ""

    try:
        if image is not None and not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        from src.pipelines.qa_rag import QARAGPipeline
        pipeline = QARAGPipeline(medgemma, colpali, clip_model)

        if use_rag:
            if retriever_choice == "ColPali" and colpali is not None:
                result = pipeline.answer_with_rag(question, image, "colpali")
            elif retriever_choice == "CLIP" and clip_model is not None:
                result = pipeline.answer_with_rag(question, image, "clip")
            else:
                result = {
                    "answer": "⚠️ Selected retriever is not loaded. Try direct mode or load models again.",
                    "context": "No retrieval used."
                }
        else:
            result = pipeline.answer_direct(question, image)

        answer = result.get("answer", "No answer generated.")
        context = result.get("context", "No retrieval used.")

        return answer, context
    except Exception as e:
        logger.error(f"QA error: {e}")
        return f"❌ Error: {e}", ""


# ─── Build Gradio Interface ─────────────────────────────────────────────────

DISCLAIMER = (
    "⚠️ **ACADEMIC DEMO — NOT FOR CLINICAL USE** ⚠️\n\n"
    "This system is a student project for DSAI 413. It has NOT been clinically "
    "validated and should NOT be used for medical diagnosis or treatment decisions. "
    "Always consult a qualified healthcare professional."
)

with gr.Blocks(
    title="Multi-Modal CXR Intelligence System",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    ),
    css="""
    .disclaimer { background-color: #fff3cd; border: 1px solid #ffc107; 
                   border-radius: 8px; padding: 12px; margin-bottom: 16px; }
    .header { text-align: center; margin-bottom: 20px; }
    """
) as app:

    # Header
    gr.Markdown(
        """
        # 🏥 Multi-Modal Chest X-Ray Intelligence System
        ### Dual-Mode: Report Generation & RAG-based QA
        **Models:** MedGemma 1.5 4B · ColPali · CLIP
        """,
        elem_classes=["header"],
    )

    gr.Markdown(DISCLAIMER, elem_classes=["disclaimer"])
    gr.Markdown(
        """
        **Demo flow:** Load models → try Report Generation → try QA with ColPali RAG → optionally compare with CLIP retrieval.  
        The retrieved context box is shown so the video demo can prove that RAG is being used.
        """
    )

    # Model loading
    with gr.Row():
        load_btn = gr.Button("🔄 Load Models", variant="primary", scale=1)
        model_status = gr.Textbox(label="Model Status", lines=4, interactive=False, scale=3)
    load_btn.click(fn=load_models, outputs=model_status)

    # Tabs for dual mode
    with gr.Tabs():

        # ─── Tab 1: Report Generation ────────────────────────────────────
        with gr.TabItem("📋 Report Generation (Mode 1)"):
            gr.Markdown("Upload a chest X-ray image to generate a structured radiology report.")

            with gr.Row():
                with gr.Column(scale=1):
                    report_image = gr.Image(
                        label="Upload Chest X-Ray",
                        type="pil",
                        height=400,
                    )
                    with gr.Row():
                        report_btn = gr.Button("📝 Generate Report (Direct)", variant="primary")
                        report_rag_btn = gr.Button("📝 Generate Report (RAG)", variant="secondary")

                with gr.Column(scale=1):
                    report_output = gr.Textbox(
                        label="Generated Report",
                        lines=20,
                        show_copy_button=True,
                    )
                    report_context = gr.Textbox(
                        label="Retrieved Context (RAG mode)",
                        lines=8,
                        visible=True,
                    )

            report_btn.click(fn=generate_report, inputs=report_image, outputs=report_output)
            report_rag_btn.click(fn=generate_report_rag, inputs=report_image,
                                  outputs=[report_output, report_context])

        # ─── Tab 2: QA Mode ─────────────────────────────────────────────
        with gr.TabItem("❓ Question Answering (Mode 2)"):
            gr.Markdown(
                "Ask a clinical question about a chest X-ray. "
                "RAG mode retrieves relevant reports for grounded answers."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    qa_image = gr.Image(
                        label="Upload Chest X-Ray (Optional)",
                        type="pil",
                        height=300,
                    )
                    qa_question = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., Is there pleural effusion in this chest X-ray?",
                        lines=2,
                    )
                    qa_use_rag = gr.Checkbox(
                        label="Use RAG (retrieve relevant reports)",
                        value=True,
                    )
                    qa_retriever = gr.Dropdown(
                        choices=["ColPali", "CLIP"],
                        value="ColPali",
                        label="Retriever for RAG / comparison",
                    )
                    qa_btn = gr.Button("🔍 Get Answer", variant="primary")

                    # Sample questions
                    gr.Markdown("**Sample Questions:**")
                    gr.Examples(
                        examples=[
                            ["Is there cardiomegaly?"],
                            ["What are the main findings?"],
                            ["Is there pleural effusion?"],
                            ["Are there any support devices visible?"],
                            ["Is this chest X-ray normal?"],
                        ],
                        inputs=qa_question,
                    )

                with gr.Column(scale=1):
                    qa_answer = gr.Textbox(
                        label="Answer",
                        lines=10,
                        show_copy_button=True,
                    )
                    qa_context = gr.Textbox(
                        label="Retrieved Context",
                        lines=8,
                    )

            qa_btn.click(
                fn=answer_question,
                inputs=[qa_image, qa_question, qa_use_rag, qa_retriever],
                outputs=[qa_answer, qa_context],
            )

    # Footer
    gr.Markdown(
        """
        ---
        **DSAI 413 — Assignment 2** | Multi-Modal Chest X-Ray Intelligence System  
        Models: MedGemma 1.5 4B (Google) · ColPali v1.2 (Vidore) · CLIP (OpenAI)  
        Dataset: MIMIC-CXR | Framework: Gradio  
        *This is an academic project. Not intended for clinical diagnosis.*
        """
    )


if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)
