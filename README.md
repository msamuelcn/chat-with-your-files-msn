# 📄 Modular RAG Chatbot

## 🌟 Project Overview

This project is a **Retrieval-Augmented Generation (RAG)** application that allows users to upload a document (PDF, TXT, or MD), index its content, and engage in a conversational Q&A chat based *strictly* on the document's information.

The application is built using **Streamlit** for the UI, **LangChain** for the RAG pipeline, and **FAISS** for high-performance vector storage.

---

## ✨ Key Technical Highlights & Architecture

The codebase is structured with a strong emphasis on **modularity, maintainability, and code consistency**, which was a core focus for this project.

### 1. **Clean Modular Architecture (Separation of Concerns)**
The project follows a clean architecture, with distinct responsibilities for each file:

* **`config.py`**: Centralized management of all environment variables, model names, and system constants (e.g., chunk sizes, cleanup timers). All constants use `UPPER_SNAKE_CASE`.
* **`utils.py`**: Handles foundational logic like directory paths (`INDEX_DIRECTORY`, `UPLOAD_DIRECTORY`) and file system setup.
* **`upload.py`**: Utility functions for file handling, including saving the uploaded file and extracting text from various formats (PDF/TXT/MD).
* **`indexer.py`**: The core data preparation logic:
    * Manages **text chunking** and LLM-based title suggestion.
    * Creates and persists the **FAISS vector store**.
    * Handles **metadata storage and automated file cleanup**.
* **`chat.py`**: The LLM orchestration layer. It loads the vector store and implements the LangChain **Agent** with a retrieval tool for conversational RAG.
* **`streamlit_app.py`**: Manages the application state, user interface, and connects the indexing/chat pipelines.

### 2. **Consistent Code Style**
* **Naming Conventions**: All variables, functions, and parameters in the logic files adhere to **`snake_case`** for consistency and readability.
* **Configuration**: All configuration constants are explicitly set in `config.py` using **`UPPER_SNAKE_CASE`** to distinguish them from local variables.
* **Dependencies**: Unused imports and variables have been meticulously removed.

### 3. **RAG Pipeline & LLM Details**
* **Vector Store**: Uses **FAISS** (Facebook AI Similarity Search) for efficient nearest-neighbor search, ensuring fast document retrieval.
* **Embeddings Flexibility**: Supports two swappable embedding backends (`OpenAIEmbeddings` or `HuggingFaceEmbeddings`) via a single configuration switch.
* **Persistence & Cleanup**: Indexed files and chat history are saved to disk (`data/`) and a background task automatically **deletes old indices and files after 12 hours** to manage storage efficiently.
* **Streaming**: Implemented LLM response streaming in the Streamlit UI for a responsive user experience.

---

## ⚙️ Setup and Installation

### Prerequisites

* Python 3.8+
* An OpenAI API Key (required for the default configuration)

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repo Link]
    cd modular-rag-chatbot
    ```
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(**Note**: Ensure your `requirements.txt` includes all necessary packages like `streamlit`, `langchain`, `openai`, `faiss-cpu`, `pypdf2`, `python-dotenv`, etc.)*
4.  **Configure Environment Variables:**
    Create a file named **`.env`** in the root directory and add your OpenAI key:
    ```ini
    OPENAI_API_KEY="sk-***********************************"
    ```
5.  **Run the App:**
    ```bash
    streamlit run streamlit_app.py
    ```

## 🛠️ Configuration

The application is easily configurable by editing **`config.py`**:

| Constant | Description | Default |
| :--- | :--- | :--- |
| `EMBEDDINGS_BACKEND` | Switch between `"openai"` or `"huggingface"`. | `"openai"` |
| `LLM_MODEL_NAME` | The language model to use for chat. | `"gpt-4o-mini"` |
| `DEFAULT_CHUNK_SIZE` | Size of text chunks for indexing. | `1000` |
| `HALF_DAY_IN_SECONDS` | File retention time before cleanup (in seconds). | `43200` (12 hours) |