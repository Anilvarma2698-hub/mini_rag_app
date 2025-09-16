import os
import logging
from typing import List, Tuple
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def read_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_docx(file) -> str:
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_txt(file) -> str:
    return file.read().decode("utf-8")


def load_documents(uploaded_files) -> Tuple[List[str], List[str]]:
    chunks = []
    metadatas = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file in uploaded_files:
        filename = file.name
        ext = os.path.splitext(filename)[1].lower()

        logger.info(f"Reading file: {filename}")

        try:
            if ext == ".pdf":
                raw_text = read_pdf(file)
            elif ext == ".docx":
                raw_text = read_docx(file)
            elif ext == ".txt":
                raw_text = read_txt(file)
            else:
                continue


            split_chunks = splitter.split_text(raw_text)
            chunks.extend(split_chunks)
            metadatas.extend([filename] * len(split_chunks))

        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")

    return chunks, metadatas
