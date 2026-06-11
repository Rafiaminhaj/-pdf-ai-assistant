import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

st.set_page_config(
    page_title="PDF AI Assistant - RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    .chat-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #f8f9fa;
        border-left: 5px solid #17a2b8;
        margin-bottom: 15px;
    }
    .source-box {
        background-color: #f1f3f5;
        border-radius: 4px;
        padding: 8px;
        margin-top: 5px;
        font-size: 13px;
        border-left: 3px solid #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>📄 PDF AI Assistant - RAG Chatbot</h1>", unsafe_allow_html=True)
st.write("Upload any PDF document and chat with an AI assistant that retrieves facts directly from the document context.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
hf_token = st.sidebar.text_input("Hugging Face API Token (Free)", type="password", help="Get a free token from huggingface.co/settings/tokens")
st.sidebar.markdown("[Get Free HF Token](https://huggingface.co/settings/tokens)")

chunk_size = st.sidebar.slider("Chunk Size", min_value=200, max_value=1000, value=500, step=100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=20, max_value=100, value=50, step=10)
num_chunks = st.sidebar.slider("Retrieve Chunks (k)", min_value=1, max_value=5, value=3)

# File Uploader
uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

# Process PDF
@st.cache_resource(show_spinner="Embedding document chunks into FAISS vector store...")
def process_pdf(pdf_path, _chunk_size, _chunk_overlap):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    
    # Embed and index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

db = None
if uploaded_file is not None:
    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name
        
    db = process_pdf(temp_pdf_path, chunk_size, chunk_overlap)
    st.success("✅ PDF processed successfully! Ask your questions below.")
else:
    # Use dummy text helper if no file uploaded
    st.info("💡 Please upload a PDF file to begin. (You can download the sample PDF generated in your project folder to test).")

# Chat System
if db is not None:
    question = st.text_input("Ask a question about the PDF:")
    if question:
        if not hf_token:
            st.warning("⚠️ Please provide a Hugging Face API Token in the sidebar for free cloud LLM generation.")
        else:
            with st.spinner("AI is retrieving context and generating answer..."):
                try:
                    # 1. Retrieve top-k chunks
                    docs = db.similarity_search(question, k=num_chunks)
                    context = "\n---\n".join([doc.page_content for doc in docs])
                    
                    # 2. Setup prompt for LLM
                    prompt = f"""<|system|>
You are a helpful assistant. Use ONLY the provided Context to answer the Question. If the answer cannot be found in the Context, respond with 'I am sorry, but that information is not available in the document.' Keep your answer short and precise.
Context:
{context}
<|user|>
Question: {question}
<|assistant|>
"""
                    
                    # 3. Call HuggingFace Cloud Inference
                    client = InferenceClient("Qwen/Qwen2.5-1.5B-Instruct", token=hf_token)
                    output_text = client.text_generation(prompt, max_new_tokens=256)
                    
                    # Display Answer
                    st.markdown("### 💬 AI Answer")
                    st.success(output_text.strip())
                    
                    # Display References
                    st.markdown("### 🔍 Source References")
                    for i, doc in enumerate(docs):
                        page = doc.metadata.get("page", 0) + 1
                        st.markdown(f"""
                            <div class='source-box'>
                                <strong>Source {i+1} (Page {page}):</strong><br>{doc.page_content}
                            </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error querying LLM: {e}")
