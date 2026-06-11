# 🚀 Hugging Face Spaces Deployment Guide

Deploying your PDF AI Assistant (RAG Chatbot) to **Hugging Face Spaces** allows anyone to access and interact with your web application live on the internet, completely for free!

Follow this step-by-step guide to deploy your app in less than 2 minutes.

---

## 📋 Prerequisites
1. Create a free account at **[huggingface.co](https://huggingface.co/)**.
2. Generate a free API Access Token:
   - Go to your profile settings: **[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)**.
   - Click **Create new token**, name it (e.g. `Colab-Token`), select role `Read`, and click **Generate**.
   - Copy this token (you will paste it into your app sidebar to run the Large Language Model).

---

## ⚡ Step-by-Step Deployment

### Step 1: Create a new Space
1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)**.
2. Click the **Create new Space** button in the top right.
3. Fill in the following details:
   - **Space Name:** `pdf-ai-assistant` (or any name you prefer)
   - **License:** `mit`
   - **Select the Space SDK:** **Streamlit** (select the Python option)
   - **Space Hardware:** Select **Cpu Basic (Free)**
   - **Space Visibility:** Public
4. Click **Create Space** at the bottom.

---

### Step 2: Upload Files
Once the space is created, click on the **Files** tab at the top and upload the following two files:

#### 1. Upload `requirements.txt`
- Click **Add file > Create a new file** (or Upload file).
- Name it exactly: `requirements.txt`.
- Copy-paste the content of your local `requirements.txt`:
  ```text
  streamlit
  langchain
  langchain-community
  sentence-transformers
  faiss-cpu
  pypdf
  huggingface_hub
  ```
- Click **Commit new file to main** at the bottom.

#### 2. Upload `app.py`
- Click **Add file > Upload files** (or Create new file).
- Name it exactly: `app.py` *(Note: Hugging Face Spaces requires the main file to be named `app.py`, so you will copy the code from your local `app_rag.py` into this file).*
- Copy-paste the contents of your local **`app_rag.py`** into this file.
- Click **Commit new file to main**.

---

### Step 3: View your Live App!
Hugging Face will automatically detect the changes, install the requirements, and build your Streamlit container. 

1. Go to the **App** tab at the top.
2. You will see a status message saying **Building** and then **Running**.
3. Once it says **Running**, your app is live! 
4. Paste your **Hugging Face API Token** into the sidebar, upload any PDF, and start chatting!

---

## 🛠️ Running Locally (Streamlit)
If you want to run this app locally on your PC:
1. Open your terminal in the `pdf-ai-assistant` folder.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Streamlit:
   ```bash
   streamlit run app_rag.py
   ```
4. Open the displayed local URL (usually `http://localhost:8501`) in Chrome!
