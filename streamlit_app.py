
import streamlit as st
import google.generativeai as genai
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
from langchain_core.runnables import RunnableLambda

# Page configuration
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .langchain-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Medical RAG System Class with LangChain integration
class MedicalRAGSystem:
    def __init__(self, vector_store_path="medical_rag_system"):
        self.vector_store_path = vector_store_path
        self.loaded = False

        try:
            # Load FAISS index
            index_path = os.path.join(vector_store_path, "medical_faiss.index")
            self.index = faiss.read_index(index_path)

            # Load metadata
            metadata_path = os.path.join(vector_store_path, "vector_metadata.pkl")
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)

            self.chunks = data['chunks']
            self.metadata = data['metadata']

            # Load embedding model
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            self.loaded = True

        except Exception as e:
            st.error(f"Error loading Medical RAG System: {e}")
            self.loaded = False

    def retrieve_medical_context(self, query, top_k=5):
        if not self.loaded:
            return []

        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)

            distances, indices = self.index.search(query_embedding, top_k)

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunks):
                    result = {
                        'content': self.chunks[idx],
                        'similarity_score': float(distance),
                        'metadata': self.metadata[idx]
                    }
                    results.append(result)

            return results

        except Exception as e:
            st.error(f"Error in retrieval: {e}")
            return []

    def generate_medical_answer(self, query, context_chunks, api_key):
        if not api_key:
            return "Please enter your Gemini API key"

        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")

            if context_chunks:
                context_text = "\n\n".join([
                    f"[Medical Note {i+1} - {chunk['metadata'].get('medical_specialty', 'General Medicine')}]\n"
                    f"{chunk['content']}"
                    for i, chunk in enumerate(context_chunks)
                ])

                prompt = f"""You are a medical assistant. Answer based ONLY on this medical context:

MEDICAL CONTEXT:
{context_text}

QUESTION: {query}

IMPORTANT: Use only information from the context. If context doesn't have relevant info, say "I cannot find specific information about this in the available medical records. If you do not found anything than make sure you get some detials from llm regarding the question and at the end gives recommendation to meet with specilalist"

ANSWER:"""
            else:
                prompt = f"""You are a medical assistant. Answer this medical question:

QUESTION: {query}

Provide a helpful medical answer.

ANSWER:"""

            # Create LangChain Runnable
            def gemini_predict(prompt: str):
                response = model.generate_content(prompt)
                return response.text

            llm = RunnableLambda(gemini_predict)
            return llm.invoke(prompt)

        except Exception as e:
            return f"Error: {str(e)}"

# Streamlit App Interface
st.markdown('<div class="main-header"> Medical RAG Assistant <span class="langchain-badge">LangChain</span></div>', unsafe_allow_html=True)
st.markdown("**Ask medical questions based on thousands of clinical transcriptions**")

# Sidebar for configuration
with st.sidebar:
    st.header(" Configuration")

    # API Key input
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get free API key from https://aistudio.google.com/"
    )

    # LangChain information
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("**LangChain Components Used:**")
    st.write("• `RunnableLambda` - LangChain core")
    st.write("• Direct Gemini API integration")
    st.write("• Medical prompt engineering")
    st.markdown('</div>', unsafe_allow_html=True)

    # System info
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.write("**System Information:**")
    st.write("• Medical text chunks: 29,713")
    st.write("• Medical specialties: 39")
    st.write("• Vector database: FAISS")
    st.write("• AI model: Google Gemini 2.0 Flash")
    st.markdown('</div>', unsafe_allow_html=True)

    # Initialize button
    if st.button(" Initialize Medical System", use_container_width=True):
        with st.spinner("Loading medical database..."):
            try:
                rag_system = MedicalRAGSystem()
                st.session_state.rag_system = rag_system
                st.session_state.api_key = api_key
                st.success(" Medical RAG System Ready!")
                st.info("Using LangChain `RunnableLambda` for Gemini integration")
            except Exception as e:
                st.error(f" Error: {e}")

# Main content area
if 'rag_system' not in st.session_state:
    st.info(" Welcome! Please enter your Gemini API key and initialize the system in the sidebar.")

    # Show sample questions
    st.markdown("**Sample medical questions you can ask:**")
    st.write("• What are symptoms of allergic rhinitis?")
    st.write("• How is asthma typically treated?")
    st.write("• What medications are used for hypertension?")
    st.write("• Describe common migraine symptoms")
else:
    # Question input
    question = st.text_input(
        "Ask your medical question:",
        placeholder="e.g., What are common treatments for allergies?",
        key="question_input"
    )

    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        num_sources = st.slider("Number of sources to use", 1, 5, 3)
    with col2:
        st.write("**Using LangChain Runnable**")
        st.write("For Gemini integration")

    # Process question
    if question and st.session_state.get('api_key'):
        with st.spinner(" Searching medical database..."):
            # Retrieve context
            sources = st.session_state.rag_system.retrieve_medical_context(question, top_k=num_sources)

            # Generate answer using LangChain pattern
            answer = st.session_state.rag_system.generate_medical_answer(
                question, sources, st.session_state.api_key
            )

        # Display answer
        st.markdown("###  Medical Answer (via LangChain)")
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.write(answer)
        st.markdown('</div>', unsafe_allow_html=True)

        # Display sources
        if sources:
            with st.expander(f" Medical Sources ({len(sources)} found)"):
                for i, source in enumerate(sources):
                    specialty = source['metadata'].get('medical_specialty', 'Medical Note')
                    similarity = source['similarity_score']

                    st.markdown('<div class="source-box">', unsafe_allow_html=True)
                    st.write(f"**Source {i+1}** | **Specialty:** {specialty} | **Relevance:** {similarity:.3f}")
                    st.write(f"**Content:** {source['content'][:300]}...")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No specific medical sources found - answer is based on general medical knowledge")

# Footer
st.markdown("---")
st.markdown("**Medical Disclaimer:** This system provides information from medical records for educational purposes only. It is not a substitute for professional medical advice.")
st.markdown("*Built with Streamlit, FAISS, LangChain, and Google Gemini*")
