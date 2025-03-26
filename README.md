# 📄 Research Paper Summarizer using Langchain + LLM

This is a simple PDF summarization app built with **Streamlit**, **Langchain**, and **LLMs**. It allows you to upload a PDF and get a concise, research-oriented summary based on semantic chunking and embedding-based similarity search.

This tool is for:

- Quickly summarizing academic papers
- Extracting high-level ideas for literature reviews
- Condensing long documents into digestible summaries

## ✨ Features

- Upload and read PDF documents
- Extracts text from all pages
- Chunks text intelligently using Langchain
- Embeds text using `sentence-transformers/all-MiniLM-L6-v2`
- Performs similarity-based retrieval using FAISS
- Generates a concise, research-focused summary using `llama3.1` via ChatOllama

## 🧠 LLM Configuration

- Uses `ChatOllama` with `llama3.1:latest` as the model
- Controlled temperature and max tokens for coherent, concise summaries
- Focuses on extracting **main ideas**, **key points**, and **numerical data**

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pdf-summarizer.git
   cd pdf-summarizer
   ```

2. **Set up a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```


3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

> 💡 Make sure you have Ollama installed and running locally to use the `llama3.1` model: https://ollama.com/


## Upcoming:
- Multimodal PDF parsing



**Abul Al Arabi**

---

🧠 *Powered by Langchain, HuggingFace Embeddings, and LLaMA3 via Ollama*
