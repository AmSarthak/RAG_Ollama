
# 📄 Homeopathic medicine finder by syptoms using RAG and Ollama.

A Retrieval-Augmented Generation (RAG) system that allows you to upload any PDF and interactively ask questions about its content. This project leverages the power of [Ollama](https://ollama.com/) to run local LLaMA 3 models for secure and efficient inference, and uses embeddings and vector search for semantic retrieval from documents.

Data used is a homeopathic REPERTORY of the Homeopathic Materia Medica by J. T. KENT, M. D. (Professor of Materia Medica and Homeopathics in Philadelphia PostGraduate School of Homoeopathies). 
Open source PDF data.
---

## 🚀 Features

* 🔍 **Semantic PDF Question Answering**
  Ask natural language questions about the content of your PDF.

* 🧠 **Powered by LLaMA 3 via Ollama**
  Uses Meta's LLaMA 3 model served locally using the Ollama runtime.

* 📚 **Retrieval-Augmented Generation (RAG)**
  Combines document retrieval with LLM generation for accurate and grounded answers.

* 🗃️ **Vector Embedding-based Context Retrieval**
  Chunks and embeds PDF content for efficient similarity search.

* 🖥️ **Runs Fully Locally (No Cloud Required)**
  All components including the language model run on your local machine.

---

## 🛠️ Tech Stack

* **Language Model:** LLaMA 3 via [Ollama](https://ollama.com/)
* **Frameworks:** Python, LangChain
* **Embedding Model:** OllamaEmbeddings
* **Vector Store:** ChromaDB
* **PDF Parsing:** PyPDFLoader

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-pdf-qa.git
cd rag-pdf-qa
```

### 2. Install and run Ollama

Install Ollama from [https://ollama.com/download](https://ollama.com/download), then run:

```bash
ollama pull llama3
ollama serve
```

This will pull and start the LLaMA 3 model.

---

## 🧠 How It Works

1. **Chunking & Embedding**: The PDF is split into manageable chunks and embedded using a semantic embedding model.
2. **Vector Search**: When a question is asked, the system finds the top-k most relevant chunks using ChromaDB.
3. **Context + LLM**: The retrieved chunks are passed as context to the LLaMA 3 model, which then generates a natural language answer.

---

## 🧱 Future Improvements

* Add a web UI using Streamlit or Gradio
* Support for multiple documents
* OCR for scanned PDFs
* Use local embeddings via Ollama (`llama3:instruct-embed` when available)
* Long-context support with LLaMA 3 70B

---

## 🧑‍💻 Author

**Sarthak Chakraborty**
Senior Software Engineer | Applied AI Enthusiast
