import logging
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="rag_docs",
    embedding_function=sentence_transformer_ef
)

def embed_and_store(chunks, metadata):
    logger.info("Storing embeddings in vector DB...")

    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        collection.add(
            documents=[chunk],
            ids=[f"chunk_{i}"],
            metadatas=[{"source": meta}]
        )

    logger.info(f"Stored {len(chunks)} chunks.")


def clear_vector_store():
    """Clear all documents from the vector database."""
    logger.info("Clearing vector store collection...")

    existing = collection.get()
    ids = existing.get("ids", [])

    if not ids:
        logger.info("No documents found in the vector store.")
        return

    collection.delete(ids=ids)
    logger.info("Vector store reset complete.")

