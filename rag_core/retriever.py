import logging
from .vector_store import collection

logger = logging.getLogger(__name__)

def retrieve_context(query, top_k=3):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    scores = [1.0 / (1.0 + d) for d in distances]

    chunks_with_source = [
        f"[{meta['source']}] {doc}" for doc, meta in zip(documents, metadatas)
    ]

    return chunks_with_source, scores
