import logging
from typing import List
from transformers import pipeline

logger = logging.getLogger(__name__)

try:
    rag_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
except Exception as e:
    logger.error("Failed to load local transformer model: %s", e)
    rag_pipeline = None

def generate_answer(question: str, context_chunks: List[str]) -> str:
    if rag_pipeline is None:
        return "LLM not available. Check your model or internet connection."

    context = "\n".join(context_chunks)
    prompt = (
        f"Answer the question based on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    logger.info("Sending prompt to local transformer model...")

    try:
        result = rag_pipeline(prompt, max_new_tokens=256, do_sample=False)
        return result[0]["generated_text"]
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        return "Failed to generate answer. Please try again later."
