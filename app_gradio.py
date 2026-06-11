import gradio as gr
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# Global database placeholder
vector_db = None

def load_pdf(pdf_file, chunk_size, chunk_overlap):
    global vector_db
    if pdf_file is None:
        return "⚠️ Please upload a PDF file first."
    
    try:
        # Load PDF using LangChain
        loader = PyPDFLoader(pdf_file.name)
        docs = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(docs)
        
        # Embed chunks and index inside FAISS Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        return f"✅ PDF processed successfully! Split into {len(chunks)} semantic chunks. Ask your questions now."
    except Exception as e:
        return f"❌ Error loading PDF: {str(e)}"

def answer_question(question, hf_token, num_chunks):
    global vector_db
    if vector_db is None:
        return "⚠️ Please upload and process a PDF first.", ""
    if not hf_token.strip():
        return "⚠️ Please provide a Hugging Face API Token in the input field.", ""
    
    try:
        # 1. Similarity Search context retrieval
        docs = vector_db.similarity_search(question, k=num_chunks)
        context = "\n---\n".join([doc.page_content for doc in docs])
        
        # 2. Setup prompt structure
        prompt = f"""<|system|>
You are a helpful assistant. Use ONLY the provided Context to answer the Question. If the answer cannot be found in the Context, respond with 'I am sorry, but that information is not available in the document.' Keep your answer short and precise.
Context:
{context}
<|user|>
Question: {question}
<|assistant|>
"""
        # 3. Query free cloud LLM (Qwen) via HuggingFace Hub client
        client = InferenceClient("Qwen/Qwen2.5-1.5B-Instruct", token=hf_token.strip())
        output_text = client.text_generation(prompt, max_new_tokens=256)
        
        # Format source chunks references in HTML
        refs_html = "<div style='margin-top: 10px;'>"
        for idx, doc in enumerate(docs):
            page = doc.metadata.get("page", 0) + 1
            refs_html += f"""
            <div style='background-color: rgba(23, 162, 184, 0.08); border-left: 4px solid #17a2b8; padding: 10px; margin-bottom: 8px; border-radius: 4px; font-size: 13px;'>
                <strong>Source {idx+1} (Page {page}):</strong> {doc.page_content}
            </div>
            """
        refs_html += "</div>"
            
        return output_text.strip(), refs_html
    except Exception as e:
        return f"❌ Error querying model: {str(e)}", ""

# Build Gradio Web Dashboard
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate")) as demo:
    gr.Markdown("# 📄 PDF AI Assistant - RAG Chatbot")
    gr.Markdown("Upload any PDF document, index its text chunks, and ask questions. The AI assistant retrieves matching context using a local FAISS database and answers without hallucinations using a quantized Qwen LLM.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Step 1: Upload Document & Authentication")
            pdf_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
            hf_token_input = gr.Textbox(label="Hugging Face API Token (Free)", type="password", placeholder="Paste your hf_... token here")
            gr.Markdown("[Get a free HF Token here](https://huggingface.co/settings/tokens)")
            
            with gr.Accordion("Advanced Vector Index Settings", open=False):
                chunk_size_input = gr.Slider(minimum=200, maximum=1000, value=500, step=100, label="Chunk Size")
                chunk_overlap_input = gr.Slider(minimum=20, maximum=100, value=50, step=10, label="Chunk Overlap")
                num_chunks_input = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Retrieve Chunks (k)")
                
            process_btn = gr.Button("⚙️ Index PDF Chunks", variant="primary")
            status_output = gr.Textbox(label="System Status", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Step 2: Chat with PDF")
            question_input = gr.Textbox(label="Enter your question:", placeholder="e.g., What are the standard working hours?")
            ask_btn = gr.Button("🚀 Ask Assistant", variant="primary")
            
            answer_output = gr.Textbox(label="AI Generated Response", interactive=False)
            sources_output = gr.HTML(label="Retrieved Source References")
            
    # Wire click actions
    process_btn.click(
        fn=load_pdf,
        inputs=[pdf_input, chunk_size_input, chunk_overlap_input],
        outputs=status_output
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, hf_token_input, num_chunks_input],
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch()
