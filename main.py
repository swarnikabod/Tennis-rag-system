import os
import torch # Import torch for device mapping

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# HuggingFace embeddings and local LLM
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- Part 1: Document Loading ---
print("Loading documents...")
DATA_PATH = "Data/"
documents = []

for item in os.listdir(DATA_PATH):
    if item.endswith('.pdf'):
        pdf_path = os.path.join(DATA_PATH, item)
        loader = PyPDFLoader(pdf_path)
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")

if not documents:
    print("No PDF documents found or loaded. Please check the 'Data' folder.")
    exit()

print(f"Loaded {len(documents)} document pages.")

# --- Part 2: Chunking ---
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", "! ", "? ", ",", " ", ""]
)
docs = text_splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")

# --- Part 3: Embeddings and Vector Store ---
print("Creating vector store with HuggingFace embeddings...")
# This embedding model runs efficiently on CPU
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
print("✅ Vector store created.")

# --- Part 4: Retriever ---
print("Creating retriever...")
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
)
print("Retriever created.")

# --- Part 5: LLM using a very small local HuggingFace model ---
print("Setting up local LLM...")

# *** CHANGE MADE HERE ***
# Using TinyLlama-1.1B-Chat-v1.0 - a much smaller model suitable for CPU.
# This model will be downloaded automatically by transformers the first time it runs.
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# --- OR try "google/gemma-2b-it" if TinyLlama is not sufficient ---
# model_id = "google/gemma-2b-it"
# For Gemma, you might need to authenticate with HuggingFace (hf_token)
# For a college project, TinyLlama might be easier as it often doesn't require authentication.

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Ensure pad_token is set for batch generation, if not already.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model directly to CPU (device_map="cpu" ensures this)
# For very small models like TinyLlama, device_map="cpu" is good.
# For slightly larger ones like Gemma-2B, it's explicitly clear.
# For some models, if you have very limited RAM, you might consider loading in 8-bit or 4-bit even on CPU,
# but for TinyLlama 1.1B, full precision on CPU should be fine for a demo.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu", # Explicitly load to CPU
    torch_dtype=torch.float32 # Ensure it uses float32 for CPU if not loading in lower precision
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.0,
    max_new_tokens=256, # Generate up to 256 new tokens for the answer. Adjust if answers are too short/long.
    do_sample=False, # Set to False for deterministic output with temperature=0.0
    return_full_text=False, # Only return the generated part, not the whole prompt
    truncation=True, # Allow the pipeline to truncate input if it's too long
)
llm = HuggingFacePipeline(pipeline=pipe)
print("✅ Local LLM ready.")

# --- Part 6: RAG Chain ---
print("Creating RAG chain...")
template = """
You are a helpful assistant for answering questions about tennis.
Use only the following context to answer the question.
If you don't know the answer from the context, just say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("RAG chain ready.")

# --- Part 7: Test Questions ---
print("\n--- Testing RAG System ---")
questions = [
    "What is a 'let' in tennis?",
    "What was the score of the longest match in tennis history?",
    "According to 'The Inner Game of Tennis', what is the biggest obstacle for a player?"
]

for q in questions:
    print(f"\nQ: {q}")
    a = rag_chain.invoke(q)
    print(f"A: {a}")

print("\n--- Test complete ---")