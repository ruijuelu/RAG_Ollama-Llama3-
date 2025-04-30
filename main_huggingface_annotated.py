# main_huggingface.py
# ============================================
# 🧠 RAG AI Assistant (Fully Offline)
# This script loads your text files from the `docs/` folder, indexes them using HuggingFace embeddings,
# and answers questions using a local LLM (LLaMA3 via Ollama) — all offline.

from langchain_community.document_loaders import DirectoryLoader  # Load all .txt files from a folder
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split documents into manageable chunks
from langchain_community.vectorstores import FAISS  # FAISS is a fast vector database to store/search embeddings
from langchain_huggingface import HuggingFaceEmbeddings  # Local text embeddings (no OpenAI needed)
from langchain_ollama import ChatOllama  # Interface with Ollama (runs LLaMA3 locally)
from langchain.chains import RetrievalQA  # Combine retrieval with question answering
from dotenv import load_dotenv  # (Optional) Load environment variables from .env
import os

# Load environment variables from .env file (optional)
load_dotenv()

# STEP 1: Load all text files from the docs/ folder
loader = DirectoryLoader("docs", glob="*.txt")  # Only loads files ending with .txt
documents = loader.load()  # Returns a list of LangChain Document objects

# STEP 2: Split long documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # Overlapping improves context quality
chunks = splitter.split_documents(documents)

# STEP 3: Create (or reuse) a FAISS vectorstore with HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Fast, lightweight local embedding model
index_path = "faiss_index"  # Where to store the local FAISS vector index

# Check if an index already exists
if not os.path.exists(index_path):
    # First-time setup: Create new FAISS index from document chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
else:
    # Load existing FAISS index (must allow deserialization)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# STEP 4: Create a Retrieval-QA chain using the local LLM from Ollama
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # Return top 4 matching document chunks
llm = ChatOllama(model="llama3")  # Connect to locally running Ollama with llama3 model
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# STEP 5: Interactive CLI loop
while True:
    query = input("Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa_chain.invoke(query)  # Ask the question using the RAG pipeline
    print("\nAnswer:", result)
