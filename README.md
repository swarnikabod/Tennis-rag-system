# Tennis RAG System

This project is a RAG (Retrieval-Augmented Generation) system designed to answer questions about Tennis. The system uses a custom set of documents, including tennis rules, history, and player mindset, as its external knowledge base.

## 📜 Project Overview

I chose Tennis as the topic for this project due to my strong personal interest in the sport. I am fascinated by its rich history, the diverse strategies employed by players, and the compelling athleticism it demands. This project allows me to combine my passion for data science with my passion for the sport.

### What It Does

This application acts as a "Tennis expert" that you can ask questions to. Instead of just using a generic AI model, it retrieves relevant information directly from the provided text files (in the `/Data` folder) and then augments its answer with that specific knowledge.

This ensures the answers are based on the documents you provide, not just the model's pre-trained (and potentially outdated or generic) information.

## 🤖 Technology & Model Choice

### Why RAG (Retrieval-Augmented Generation)?

A standard Large Language Model (LLM) only knows what it was trained on. If you want to ask questions about specific, private, or very recent information (like the documents in my `/Data` folder), the LLM won't know the answers.

**RAG is the perfect solution for this problem.** It works in two steps:

1.  **Retrieval:** When you ask a question (e.g., "What is the history of Wimbledon?"), the system first searches the custom documents (`/Data` folder) to find the most relevant text snippets.
2.  **Generation:** It then takes your question *and* the relevant snippets it found and feeds them to the LLM. It essentially says, "Based on these specific facts, answer this question."

This approach makes the system:
* **Accurate:** Answers are grounded in your specific documents.
* **Knowledgeable:** It can "talk about" information it was never trained on.
* **Trustworthy:** It reduces the chance of the AI "hallucinating" or making up facts.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

* Python 3.8 or higher
* Git

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/swarnikabod/Tennis-rag-system.git](https://github.com/swarnikabod/Tennis-rag-system.git)
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd Tennis-rag-system
    ```

3.  **Create and activate a virtual environment:**

    * **On Windows:**
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

    * **On macOS / Linux:**
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    *(Note: You should see `(venv)` appear in your terminal prefix.)*

4.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

---

## 🏃‍♀️ Running the Application

Once everything is installed, you can run the main script:

```sh
python main.py
