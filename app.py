import streamlit as st
import logging
from rag_core.document_loader import load_documents
from rag_core.vector_store import embed_and_store
from rag_core.retriever import retrieve_context
from rag_core.llm_generator import generate_answer

# Setup logging to monitor application behavior
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Streamlit app layout and title
st.set_page_config(page_title="Mini RAG App", layout="centered")
st.title("Mini RAG Application")

st.markdown("Upload documents and ask questions. We'll find the answer!")

# Upload area for documents (PDF, DOCX, TXT)
uploaded_files = st.file_uploader(
    "Upload your documents (PDF or DOCX OR TXT)",type=["pdf", "docx", "txt"] , accept_multiple_files=True
)

if uploaded_files:
    # Display loading spinner during processing with st.spinner("Processing documents..."):
        all_chunks, metadata = load_documents(uploaded_files)
        embed_and_store(all_chunks, metadata)
st.success("Documents processed and stored in vector DB!")


# Optional: Reset the vector DB
if st.button("Reset Vector DB"):
    from rag_core.vector_store import clear_vector_store
    clear_vector_store()
    st.success("Vector store has been reset. Upload fresh documents now.")

# Input box for user to ask natural language questions
query = st.text_input("Ask a question about your documents")

if query:
    # Display loading spinner during processing\nwith st.spinner("Retrieving context and generating answer..."):
        top_chunks, scores = retrieve_context(query)
        answer = generate_answer(query, top_chunks)

        st.markdown("### Final Answer")
        st.write(answer)

        st.markdown("### Retrieved Contexts with Confidence Scores")
        for i, (chunk, score) in enumerate(zip(top_chunks, scores)):
            st.markdown(f"**Passage {i+1} (Score: {score:.4f})**")
            st.code(chunk)
