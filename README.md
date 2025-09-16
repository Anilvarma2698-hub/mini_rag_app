# Mini RAG Application

This is a simple Retrieval-Augmented Generation (RAG) system that allows you to upload documents (PDF, DOCX), query them in natural language, and get contextual answers powered by LLMs.

## Features

- Upload PDF/DOCX/txt files
- Vector search with chunking
- Uses local Hugging Face model for answers
- Streamlit interface
- Confidence scores + citations

## Requirements

- Python 3.8+

## Installation

```bash
git checkout interview_project
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Example

1. Upload `Machine_Learning_Basics.txt` from docs folder 
2. Ask:  
   ```
   What is supervised learning according to the document?
   ```

You’ll see:
- A concise LLM-generated answer
- Retrieved chunks + confidence scores
- Citations for document origin
