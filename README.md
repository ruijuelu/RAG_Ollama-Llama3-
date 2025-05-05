# 🔍 RAG Investigation Bot (Offline AI Assistant)

A fully local Retrieval-Augmented Generation (RAG) AI assistant built with:
- Python + LangChain (community modules)
- Ollama + LLaMA3 for LLM inference
- HuggingFace sentence-transformers for local embeddings
- Your own `.txt` documents as the searchable knowledge base

Use case: Supporting investigative work (e.g. scam detection, SOP queries, AI project planning) entirely offline.

---

## 📁 Project Structure

```
rag-investigation-bot/
├── docs/                       # Text documents used as knowledge base
│   ├── scam_detection_overview.txt
│   ├── investigation_protocols.txt
├── faiss_index/               # Auto-generated local vector index
├── main_huggingface.py        # Core RAG script
├── .env                       # Optional env vars (not used for Ollama setup)
├── README.md                  # Project summary
```

---

## 🚀 How to Run (Step-by-Step)

### 1. Install system dependencies
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv curl jq
```

### 2. Set up your project
```bash
mkdir ~/Desktop/rag_investigation_bot
cd ~/Desktop/rag_investigation_bot
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python packages
```bash
pip install --upgrade pip
pip install langchain langchain-community langchain-huggingface langchain-ollama sentence-transformers faiss-cpu python-dotenv
```

### 4. Install and run Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3
```

### 5. Add `.txt` documents
Put your text files in the `docs/` folder. Example:
```
Scam detection uses NLP, rule-based systems, and behavioral analytics to flag potential fraud...
```

### 6. Run the assistant
```bash
python3 main_huggingface.py
```

---

## 💡 Example Prompts

- "What is scam detection?"
- "Explain how RAG is used in enforcement."
- "What is the SOP for phishing scam investigations?"

---

## 🔁 Updating the Knowledge Base

1. Add or edit `.txt` files in the `docs/` directory
2. Delete the old FAISS index:
   ```bash
   rm -rf faiss_index
   ```
3. Re-run the script to auto-rebuild the vectorstore

---

## 📦 Requirements

- Python 3.10+
- Ollama (with LLaMA3 or any supported local model)
- Pip packages:
  - langchain
  - langchain-community
  - langchain-huggingface
  - langchain-ollama
  - sentence-transformers
  - faiss-cpu

---

## 🔐 Offline & Secure

This project runs fully offline and does not require any OpenAI API or cloud connection. Perfect for investigative environments where privacy and control are essential.

