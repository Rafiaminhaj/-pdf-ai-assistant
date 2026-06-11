# 📄 PDF AI Assistant (Retrieval-Augmented Generation - RAG Chatbot)

This repository contains a self-contained, production-grade Google Colab notebook for building an **Interactive PDF Q&A Chatbot** using local **Retrieval-Augmented Generation (RAG)** pipelines.

The system allows users to upload any PDF document, break it down into semantic chunks, index it in a high-performance vector store, and query it using natural language. The local Large Language Model answers questions based **only** on the retrieved context, eliminating hallucinations.

---

## 🌟 Key Features

1. **Local Quantized LLM**:
   - Uses `Qwen/Qwen2.5-1.5B-Instruct` loaded on Colab's GPU using half-precision (`float16`) for extremely fast and high-quality responses.
2. **Offline Vector Search**:
   - Uses Hugging Face's `all-MiniLM-L6-v2` embeddings (384-dimensional dense vectors) to capture semantic meanings of text.
   - Employs **FAISS (Facebook AI Similarity Search)** as a fast, in-memory vector store for chunk retrieval.
3. **Out-of-the-box Execution**:
   - Generates a synthetic 3-page policy handbook (`indian_company_policy.pdf`) using Python's `reportlab` inside the notebook, allowing you to run and test it immediately without needing to upload external files.
4. **Interactive Chat Console**:
   - Renders a styled HTML interface inside the Colab cell output using `ipywidgets`. It displays AI responses alongside **numbered source page references** retrieved from the PDF.
5. **No-Cost Setup**:
   - Runs 100% locally on Colab's T4 GPU. No OpenAI/Anthropic API keys or paid tokens are required!

---

## 🚀 How to Run in Google Colab

1. **Download** the notebook file: [`PDF_AI_Assistant_RAG.ipynb`](PDF_AI_Assistant_RAG.ipynb).
2. **Open** [Google Colab](https://colab.research.google.com/).
3. **Upload** the notebook (`File > Upload notebook`).
4. **Enable GPU**:
   - Click `Runtime > Change runtime type` in the top menu.
   - Under *Hardware accelerator*, select **T4 GPU**.
5. **Execute all cells** (`Runtime > Run all`).

---

## 📊 RAG Pipeline Architecture

```text
  [ PDF Document ]
         │
         ▼ (pypdf Loader)
   [ Full Text ]
         │
         ▼ (Recursive Character Splitter)
  [ Text Chunks ] ──► (sentence-transformers Embeddings) ──► [ FAISS Vector Store ]
                                                                     ▲
                                                                     │ (Query Retrieval)
  [ User Question ] ─────────────────────────────────────────────────┼
         │                                                           │
         ▼                                                           ▼
 [ Prompt Template ] ◄──────────────────────────────────────── [ Top-k Context ]
         │
         ▼
[ Quantized LLM (Qwen) ] ──► [ Verified Answer + Sources ]
```

---

## 📁 Repository Structure

```text
├── PDF_AI_Assistant_RAG.ipynb  # Main Google Colab Jupyter Notebook
├── generate_notebook.py        # Python builder script to compile the notebook
├── README.md                   # Project documentation
└── .gitignore                  # Git exclusion rules
```

---

## 🛠️ Local Installation (Optional)

To run the builder script locally:
```bash
pip install -r requirements.txt
```
*(Dependencies within the notebook cells are installed automatically during Colab execution).*

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).
