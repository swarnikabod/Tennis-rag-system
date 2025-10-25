My RAG System 

Topic: Tennis 
I chose 'Tennis' as my topic due to my strong interest in the sport's rich history, diverse strategies, and the compelling athleticism of its players. 

What I changed
Documents: Replaced the default PDFs with 5 files related to tennis rules, history, and player mindset. 

Chunking: I used a chunk_size of 400 and chunk_overlap of 50, as this worked well for capturing specific rules and concepts. 

Retrieval: I modified the retriever to use k=5 and fetch_k=10 to get a wider set of relevant chunks for the LLM to use.

Here is a more formal, recruiter-friendly version of your README.md file, with the requested explanation for your model choices.

Tennis-Domain RAG System (Offline & CPU-Enabled)
Project Overview
This project implements an end-to-end Retrieval-Augmented Generation (RAG) system specialized for the domain of tennis.

It operates entirely offline by leveraging a local, curated knowledge base of PDF documents. The system uses a lightweight, open-source Large Language Model (TinyLlama-1.1B) that runs efficiently on a CPU, demonstrating a complete, private, and cost-free inference pipeline without reliance on external APIs.

Features
Offline Capability: Functions without any external API calls or internet access after the initial model download.

CPU-Based Inference: Designed to run on standard consumer hardware (CPU) without requiring a dedicated GPU.

Domain-Specific Knowledge: Ingests a curated set of PDF documents (e.g., rulebooks, history) to provide specialized answers.

Vector-Based Retrieval: Employs all-MiniLM-L6-v2 for efficient and accurate document retrieval from a local FAISS vector store.

Architecture and Model Rationale
This project was built to demonstrate a self-contained AI system that prioritizes privacy, accessibility, and zero operational cost.

Embedding Model: all-MiniLM-L6-v2
Why: For the "Retrieval" component of RAG, text documents must be converted into numerical vectors (embeddings).

Rationale: all-MiniLM-L6-v2 is a high-performance sentence-transformer model that is both lightweight and fast. It runs entirely locally, producing high-quality embeddings that allow the system to quickly find the most relevant document chunks to answer a user's question.

Language Model (LLM): TinyLlama/TinyLlama-1.1B-Chat-v1.0
Why: For the "Generation" component, a model was needed to synthesize an answer based on the retrieved documents.

Rationale: To ensure the system runs offline on standard hardware, API-based models (like GPT-4) were intentionally avoided. TinyLlama-1.1B was selected for its exceptional balance of size and performance. As a 1.1 billion parameter model, it is small enough to be loaded and run on a CPU while still being highly effective at chat and instruction-following, making it the ideal choice for a resource-constrained, local-first application.

Project Structure
.
├── Data/                   # Contains all Tennis-related PDF files
├── main.py                 # The main Python script to run the RAG pipeline
├── README.md               # This project documentation file
├── requirements.txt           # Folder for dependencies
└── venv/                   # Python virtual environment (ignored by .gitignore)

Setup and Execution

1.Clone the repository: git clone <your-repo-url>
cd <project-folder>

2.Create and activate a virtual environment:
# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
 source venv/bin/activate 

3.Install dependencies:
pip install -r requirements.txt

4.Run the application:
python main.py
