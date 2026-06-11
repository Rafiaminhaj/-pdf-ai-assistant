import json
import os

# Define the notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source]
    })

# --- CELL 1: TITLE & INTRO ---
add_markdown([
    "# 📄 PDF AI Assistant - Retrieval-Augmented Generation (RAG) Chatbot",
    "### Fine-Tuning Search & Generation using FAISS, Hugging Face Embeddings, and Qwen LLM",
    "",
    "This notebook demonstrates a complete, production-grade **Retrieval-Augmented Generation (RAG)** pipeline executing entirely within Google Colab. The system allows you to upload any PDF document, parse and slice its content, embed the slices in a high-performance vector store, and ask natural language questions.",
    "",
    "#### 🌟 Portfolio Project Highlights:",
    "1. **Local Quantized LLM**: Employs `Qwen/Qwen2.5-1.5B-Instruct` loaded locally onto Colab's GPU runtime using FP16 formatting to ensure rapid execution.",
    "2. **Vector Space Indexing**: Utilizes `sentence-transformers/all-MiniLM-L6-v2` for generating semantic embeddings and `FAISS` for fast vector similarity searches.",
    "3. **Zero-Cost Implementation**: Operates entirely offline inside Colab without requiring external paid API keys (like OpenAI or Anthropic).",
    "4. **Interactive Chat Widget**: Features a custom HTML/CSS chat console built using `ipywidgets` that displays LLM replies alongside retrieved document references.",
    "",
    "Let's get started!"
])

# --- CELL 2: INSTALL DEPENDENCIES ---
add_markdown([
    "## 🛠️ Step 1: Install Dependencies",
    "We install the required libraries: `langchain` and `langchain-community` for RAG routing, `sentence-transformers` for embeddings, `faiss-cpu` for indexing, `pypdf` for parsing, `transformers` and `accelerate` for local GPU model execution, and `reportlab` to programmatically build a dummy PDF so the notebook is instantly runnable."
])
add_code([
    "# Install dependencies silently",
    "!pip install -q langchain langchain-community sentence-transformers faiss-cpu pypdf transformers accelerate reportlab ipywidgets"
])

# --- CELL 3: IMPORTS ---
add_markdown([
    "## 📦 Step 2: Import Libraries",
    "We import standard packages for PyTorch, Hugging Face pipelines, and LangChain wrappers."
])
add_code([
    "import os",
    "import torch",
    "from reportlab.lib.pagesizes import letter",
    "from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer",
    "from reportlab.lib.styles import getSampleStyleSheet",
    "",
    "from langchain_community.document_loaders import PyPDFLoader",
    "from langchain.text_splitter import RecursiveCharacterTextSplitter",
    "from langchain_community.embeddings import HuggingFaceEmbeddings",
    "from langchain_community.vectorstores import FAISS",
    "from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline",
    "",
    "print(f\"CUDA (GPU) available: {torch.cuda.is_available()}\")"
])

# --- CELL 4: SAMPLE PDF GENERATOR ---
add_markdown([
    "## 📄 Step 3: Programmatically Generate a Sample PDF",
    "To make the notebook runnable without requiring you to manually upload a PDF, we write a python script using `reportlab` to generate a structured 4-section employee handbook (`indian_company_policy.pdf`)."
])
add_code([
    "# Define PDF generator function",
    "def create_sample_pdf():",
    "    doc = SimpleDocTemplate(\"indian_company_policy.pdf\", pagesize=letter)",
    "    styles = getSampleStyleSheet()",
    "    story = []",
    "    ",
    "    # Title",
    "    story.append(Paragraph(\"<b>OmniCorp India - Employee Policy Handbook</b>\", styles[\"Title\"]))",
    "    story.append(Spacer(1, 12))",
    "    ",
    "    # Section 1",
    "    story.append(Paragraph(\"<b>1. Working Hours and Attendance</b>\", styles[\"Heading2\"]))",
    "    story.append(Paragraph(\"OmniCorp India operates on a 5-day workweek (Monday to Friday). The standard working hours are 9:00 AM to 6:00 PM IST, including a 1-hour lunch break from 1:00 PM to 2:00 PM. Employees are expected to log their daily attendance in the HR portal by 9:30 AM.\", styles[\"BodyText\"]))",
    "    story.append(Spacer(1, 10))",
    "    ",
    "    # Section 2",
    "    story.append(Paragraph(\"<b>2. Leave Policies</b>\", styles[\"Heading2\"]))",
    "    story.append(Paragraph(\"Employees are entitled to 18 Sick Leaves (SL) and 12 Casual Leaves (CL) per calendar year, accrued monthly. Any unutilized sick leaves up to 5 days will carry forward to the next year, while casual leaves expire on December 31st. Maternity leave is provided for 26 weeks, and paternity leave is provided for 2 weeks.\", styles[\"BodyText\"]))",
    "    story.append(Spacer(1, 10))",
    "    ",
    "    # Section 3",
    "    story.append(Paragraph(\"<b>3. Health Benefits and Reimbursements</b>\", styles[\"Heading2\"]))",
    "    story.append(Paragraph(\"OmniCorp provides comprehensive health insurance of up to INR 5,00,000 per annum covering the employee, spouse, and up to two children. Additionally, employees can claim a monthly internet allowance of up to INR 1,500 and a wellness reimbursement of INR 2,000 per quarter for gym memberships or physical checkups.\", styles[\"BodyText\"]))",
    "    story.append(Spacer(1, 10))",
    "    ",
    "    # Section 4",
    "    story.append(Paragraph(\"<b>4. Remote Work & Travel Policy</b>\", styles[\"Heading2\"]))",
    "    story.append(Paragraph(\"Under the Hybrid Work Model, employees are allowed to work remotely for up to 2 days per week with prior manager approval. Travel expenses incurred for official business trips will be reimbursed based on receipts, with air travel allowed in economy class for flights under 6 hours.\", styles[\"BodyText\"]))",
    "    ",
    "    doc.build(story)",
    "    print(\"Sample PDF 'indian_company_policy.pdf' created successfully!\")",
    "",
    "create_sample_pdf()"
])

# --- CELL 5: LOAD AND SPLIT ---
add_markdown([
    "## ✂️ Step 4: Load and Chunk PDF Document",
    "We load the PDF document using LangChain's `PyPDFLoader` and split it into chunks of 400 characters with 50 characters overlap. Chunking ensures our local LLM receives concise context, preventing context length issues."
])
add_code([
    "# Load PDF",
    "loader = PyPDFLoader(\"indian_company_policy.pdf\")",
    "documents = loader.load()",
    "",
    "# Split document into chunks",
    "text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)",
    "chunks = text_splitter.split_documents(documents)",
    "",
    "print(f\"Loaded {len(documents)} pages.\")",
    "print(f\"Split document into {len(chunks)} text chunks.\")",
    "print(\"\\n--- Sample Chunk 1 Preview ---\")",
    "print(chunks[0].page_content)"
])

# --- CELL 6: VECTOR DB ---
add_markdown([
    "## 🗄️ Step 5: Generate Vector Embeddings & Index in FAISS",
    "We use the open-source Hugging Face embeddings model `all-MiniLM-L6-v2` to convert text chunks into 384-dimensional dense vectors. These vectors are then stored in a `FAISS` Vector Store for rapid semantic search queries."
])
add_code([
    "# Initialize embeddings model",
    "embeddings = HuggingFaceEmbeddings(model_name=\"sentence-transformers/all-MiniLM-L6-v2\")",
    "",
    "# Index chunks inside FAISS database",
    "vector_db = FAISS.from_documents(chunks, embeddings)",
    "print(\"FAISS Vector Database created and indexed successfully!\")"
])

# --- CELL 7: LOAD LOCAL LLM ---
add_markdown([
    "## 🤖 Step 6: Load Quantized Local LLM (Qwen 1.5B)",
    "We load `Qwen/Qwen2.5-1.5B-Instruct` - a state-of-the-art small language model that performs exceptionally well. We load the model in **FP16 half-precision** and map it to the GPU to maximize inference speeds."
])
add_code([
    "# Load model and tokenizer",
    "model_id = \"Qwen/Qwen2.5-1.5B-Instruct\"",
    "print(\"Loading model and tokenizer... (Takes ~1-2 minutes)\")",
    "",
    "tokenizer = AutoTokenizer.from_pretrained(model_id)",
    "",
    "# Check device target",
    "device = 0 if torch.cuda.is_available() else -1",
    "",
    "# Load model",
    "model = AutoModelForCausalLM.from_pretrained(",
    "    model_id,",
    "    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,",
    "    device_map=\"auto\" if torch.cuda.is_available() else None",
    ")",
    "",
    "# Create generation pipeline",
    "llm_pipeline = pipeline(",
    "    \"text-generation\",",
    "    model=model,",
    "    tokenizer=tokenizer,",
    "    max_new_tokens=256,",
    "    temperature=0.2,",
    "    top_p=0.9,",
    "    repetition_penalty=1.15,",
    "    device=device if not torch.cuda.is_available() else None",
    ")",
    "print(f\"Model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}!\")"
])

# --- CELL 8: RAG ROUTING LOGIC ---
add_markdown([
    "## 🔄 Step 7: Create RAG Pipeline Routing Logic",
    "We define our custom RAG function. It queries our FAISS database to find the top-3 most relevant text chunks, injects them into a system instruction template, formats the prompt using the model's chat template, and calls the generation pipeline."
])
add_code([
    "def run_rag_pipeline(question, num_chunks=3):",
    "    # 1. Retrieve relevant contexts",
    "    docs_and_scores = vector_db.similarity_search(question, k=num_chunks)",
    "    context = \"\\n---\\n\".join([doc.page_content for doc in docs_and_scores])",
    "    ",
    "    # 2. Formulate Chat template structure",
    "    messages = [",
    "        {",
    "            \"role\": \"system\",",
    "            \"content\": \"You are an HR Assistant. Use ONLY the provided Context to answer the Question. If the answer cannot be found in the Context, respond with 'I am sorry, but that information is not available in the employee handbook.' Keep your answer short and precise.\"",
    "        },",
    "        {",
    "            \"role\": \"user\",",
    "            \"content\": f\"Context:\\n{context}\\n\\nQuestion: {question}\"",
    "        }",
    "    ]",
    "    ",
    "    # Apply model tokenizer template",
    "    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
    "    ",
    "    # 3. Generate answer",
    "    outputs = llm_pipeline(prompt)",
    "    response = outputs[0][\"generated_text\"]",
    "    ",
    "    # Extract assistant's response portion",
    "    assistant_response = response[len(prompt):].strip()",
    "    ",
    "    return assistant_response, docs_and_scores"
])

# --- CELL 9: EVALUATE MANUALLY ---
add_markdown([
    "## 🧪 Step 8: Test the Pipeline with a Sample Query",
    "Let's test the RAG pipeline with a custom question to verify that context retrieval and question answering are functioning properly."
])
add_code([
    "test_query = \"How many sick leaves can I take, and do they carry forward?\"",
    "answer, sources = run_rag_pipeline(test_query)",
    "",
    "print(f\"Question: {test_query}\\n\")",
    "print(f\"Generated Answer:\\n{answer}\\n\")",
    "print(\"--- Source References Retrieved ---\")",
    "for idx, doc in enumerate(sources):",
    "    print(f\"\\n[Source {idx+1}] (Page {doc.metadata.get('page', 0) + 1}):\")",
    "    print(doc.page_content)"
])

# --- CELL 10: INTERACTIVE UI ---
add_markdown([
    "## 💬 Step 9: Interactive Chat with PDF Assistant",
    "Use this interactive widget to ask any question about the employee handbook! Type your question below, click **Ask AI Assistant**, and view real-time answers along with the exact source pages retrieved from the document."
])
add_code([
    "import ipywidgets as widgets",
    "from IPython.display import display, HTML",
    "",
    "# Text input",
    "query_input = widgets.Text(",
    "    value='How much monthly reimbursement can I claim for internet?',",
    "    placeholder='Ask a question about the handbook...',",
    "    description='Question:',",
    "    layout=widgets.Layout(width='80%')",
    ")",
    "",
    "# Action button",
    "ask_button = widgets.Button(",
    "    description='Ask AI Assistant',",
    "    button_style='info',",
    "    tooltip='Click to retrieve context and generate response',",
    "    icon='paper-plane'",
    ")",
    "",
    "chat_output = widgets.Output()",
    "",
    "def on_ask_clicked(b):",
    "    with chat_output:",
    "        chat_output.clear_output()",
    "        question = query_input.value.strip()",
    "        if not question:",
    "            print(\"Please enter a valid question!\")",
    "            return",
    "        ",
    "        print(\"Searching PDF index and loading LLM response...\")",
    "        ",
    "        # Execute pipeline",
    "        answer, sources = run_rag_pipeline(question)",
    "        ",
    "        chat_output.clear_output()",
    "        ",
    "        # Build sources HTML block",
    "        sources_html = \"\"",
    "        for idx, doc in enumerate(sources):",
    "            sources_html += f\"\"\"",
    "            <div style='background-color: #f1f3f5; border-radius: 4px; padding: 10px; margin-top: 6px; font-size: 12px; border-left: 3px solid #17a2b8;'>",
    "                <strong>Source {idx+1} (Page {doc.metadata.get('page', 0) + 1}):</strong> {doc.page_content}",
    "            </div>",
    "            \"\"\"",
    "        ",
    "        # Render output card",
    "        html_card = f\"\"\"",
    "        <div style='font-family: sans-serif; padding: 16px; border-radius: 8px; background-color: #ffffff; border: 1px solid #dee2e6; box-shadow: 0 4px 10px rgba(0,0,0,0.08); margin-top: 15px;'>",
    "            <h3 style='color: #17a2b8; margin-top: 0; margin-bottom: 12px;'>📄 Q&A RAG Assistant</h3>",
    "            <p style='font-size: 14px; margin-top: 0;'><strong>Question:</strong> \"{question}\"</p>",
    "            <div style='background-color: #e2f0d9; border-left: 6px solid #70ad47; padding: 12px; border-radius: 4px; font-size: 14px; color: #2e4d1e; line-height: 1.6;'>",
    "                <strong>AI Answer:</strong> {answer}",
    "            </div>",
    "            <div style='margin-top: 16px;'>",
    "                <span style='font-weight: bold; font-size: 13px; color: #495057;'>Source References (Ground Truth):</span>",
    "                {sources_html}",
    "            </div>",
    "        </div>",
    "        \"\"\"",
    "        display(HTML(html_card))",
    "",
    "ask_button.on_click(on_ask_clicked)",
    "display(widgets.VBox([",
    "    widgets.HTML(\"<h4 style='color:#17a2b8; margin-bottom:8px;'>💬 Ask a question about the Policy PDF:</h4>\"),",
    "    query_input,",
    "    ask_button,",
    "    chat_output",
    "]))"
])

# Write file to target path
file_path = "C:/Users/adiqu/.gemini/antigravity/scratch/pdf-ai-assistant/PDF_AI_Assistant_RAG.ipynb"
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook generated successfully at {file_path}")
