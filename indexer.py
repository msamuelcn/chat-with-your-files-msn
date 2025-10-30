# indexer.py

import os
import uuid
import pickle
import time
import shutil
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Local imports
from utils import INDEX_DIRECTORY
import config  # Using centralized config

from pydantic import BaseModel, Field
from upload import extract_text
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import (
    ChatPromptTemplate,
)  # Kept, but its use is a bit isolated

# Configure logging to provide better feedback for cleanup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Configuration Constants (imported from config, standardized to snake_case for local use) ---
openai_api_key = config.OPENAI_API_KEY
embeddings_backend = config.EMBEDDINGS_BACKEND
openai_embedding_model = config.OPENAI_EMBEDDING_MODEL
llm_model_name = config.LLM_MODEL_NAME
half_day_in_seconds = config.HALF_DAY_IN_SECONDS
# --- Configuration Constants ---


# --- Chat History Persistence Functions (File-based fix) ---
def get_history_file_path(chat_id: str) -> Path:
    """Returns the path to the chat history JSON file within the index directory."""
    return INDEX_DIRECTORY / chat_id / "history.json"


def save_chat_history(chat_id: str, chat_history: List):
    """Saves the current chat thread for a given chat ID to a JSON file."""
    history_path = get_history_file_path(chat_id)
    try:
        with open(history_path, "w", encoding="utf-8") as f:
            # chat_history is a List[Tuple[str, str]], which is JSON serializable
            json.dump(chat_history, f)
    except IOError as e:
        logging.error(f"Failed to save chat history for {chat_id}: {e}")


def load_chat_history(chat_id: str) -> List:
    """Loads the chat thread for a specific chat ID from the JSON file."""
    history_path = get_history_file_path(chat_id)
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load chat history for {chat_id}: {e}")
            return []  # Return empty list on error
    return []


# --- End Chat History Persistence Functions ---


def list_indices() -> List[Dict[str, Any]]:
    """Lists metadata for all available chat indices."""
    index_metadata = []

    # We now simply list indices that have a meta.pkl file on disk
    for directory in INDEX_DIRECTORY.iterdir():
        if directory.is_dir():
            meta_path = directory / "meta.pkl"
            if meta_path.exists():
                try:
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                    index_metadata.append(meta)
                except Exception as e:
                    logging.warning(
                        f"Could not load metadata for index {directory.name}: {e}"
                    )

    return index_metadata


def start_clear_files():
    """
    Cleans up old uploaded files and FAISS indices.
    Improved error handling and logging (Fix: Incomplete Error Handling).
    """
    logging.info("Starting file cleanup for old indices.")

    for directory in INDEX_DIRECTORY.iterdir():
        if directory.is_dir():
            meta_path = directory / "meta.pkl"

            if meta_path.exists():
                try:
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)

                    timestamp = meta["uploaded_date"]
                    current_time = time.time()

                    if (
                        current_time - timestamp > half_day_in_seconds
                    ):  # Use consistent snake_case
                        file_path = Path(meta["source_path"])

                        # 1. Delete the original uploaded file
                        if file_path.exists():
                            os.remove(file_path)
                            logging.info(f"Deleted file: {file_path}")

                        # 2. Delete the index directory
                        shutil.rmtree(directory)
                        logging.info(f"Deleted index directory: {directory}")

                except FileNotFoundError:
                    logging.warning(
                        f"Meta file or source file not found for index {directory.name}. Deleting index directory."
                    )
                    try:
                        shutil.rmtree(directory)
                    except OSError as e:
                        logging.error(
                            f"Failed to delete index directory {directory.name}: {e}"
                        )
                except (IOError, pickle.UnpicklingError) as e:
                    logging.error(
                        f"Error reading or unpickling meta file in {directory.name}: {e}"
                    )
                except OSError as e:
                    logging.error(f"OS Error during cleanup (permissions?): {e}")

    logging.info("File cleanup complete.")


def get_embeddings(backend: str = embeddings_backend):  # Use consistent snake_case
    """Initializes and returns the appropriate embeddings model and dimension."""
    if backend == "openai":
        embeddings_model = OpenAIEmbeddings(
            model=openai_embedding_model,  # Use consistent snake_case
            api_key=openai_api_key,  # Use consistent snake_case
        )
    elif backend == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings_model = HuggingFaceEmbeddings(
            model_name=config.HUGGINGFACE_EMBEDDING_MODEL  # Use direct config value
        )
    else:
        raise NotImplementedError(f"Embedding backend '{backend}' not supported.")

    # Calculate dimension from a sample query
    sample = "test"
    vector = embeddings_model.embed_query(sample)
    embedding_dim = len(vector)

    return embeddings_model, embedding_dim


class TextTitleAnalysis(BaseModel):  # Renamed for clarity: only suggests a title
    """Pydantic model for structured output from LLM analysis (title only)."""

    suggested_title: str = Field(
        description="Concise, informative title for the text excerpt."
    )


def analyze_text_sample_with_llm(
    text_sample: str,
    llm: ChatOpenAI,
) -> str:
    """
    Asks the LLM to suggest a title for the document (Latentcy fix: chunking logic removed).
    """
    if llm is None:
        return "Untitled Document"

    # Simplified prompt: only ask for the title
    template = """
You are an expert in text summarization.

Suggest a **concise and meaningful title** (under 10 words) for the following document excerpt:

Document excerpt:
----------------
{excerpt}
----------------
"""
    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(excerpt=text_sample)

    llm_with_structure = llm.with_structured_output(
        TextTitleAnalysis
    )  # Use new class name

    try:
        resp = llm_with_structure.invoke(messages)
        return resp.suggested_title.strip()
    except Exception as e:
        logging.error(
            f"Error parsing LLM response for title: {e}. Falling back to default."
        )
        return "Untitled Document"


def chunk_text(text: str, chunk_size: int, chunk_overlap: int = 200) -> List[Document]:
    """Splits the text into LangChain Documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    texts = splitter.split_text(text)
    docs = [
        Document(page_content=t, metadata={"chunk": i}) for i, t in enumerate(texts)
    ]
    return docs


def save_vector_store_to_disk(faiss_index: FAISS, destination_path: Path):
    """Saves the FAISS index to the specified directory."""
    destination_path.mkdir(parents=True, exist_ok=True)
    faiss_index.save_local(str(destination_path))


def process_and_index(file_path: Path, file_name: str) -> Dict[str, Any]:
    """Extracts text, chunks it, and creates a FAISS vector store."""
    logging.info(f"Starting process and index for {file_name}")

    text_content = extract_text(file_path)
    sample_text = text_content[:2000]

    # Initialize LLM and Embeddings
    llm = ChatOpenAI(
        api_key=openai_api_key,  # Use consistent snake_case
        model=llm_model_name,  # Use consistent snake_case
        streaming=False,
        temperature=0.0,
    )
    embeddings_model, _ = get_embeddings(
        embeddings_backend
    )  # Use consistent snake_case

    # Use default chunking parameters for speed (Fix: Latency During Indexing)
    recommended_chunk_size = config.DEFAULT_CHUNK_SIZE
    recommended_overlap_chunk_size = config.DEFAULT_CHUNK_OVERLAP

    # Analyze sample to determine title (fast LLM call)
    suggested_title = analyze_text_sample_with_llm(sample_text, llm=llm)

    # Chunk the entire document
    documents = chunk_text(
        text_content,
        chunk_size=recommended_chunk_size,
        chunk_overlap=recommended_overlap_chunk_size,
    )

    # Create and save the FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings_model)

    chat_id = str(uuid.uuid4())
    destination_dir = INDEX_DIRECTORY / chat_id
    save_vector_store_to_disk(vector_store, destination_dir)
    logging.info(f"Saved FAISS index to {destination_dir}")

    # Save metadata
    metadata = {
        "uid": chat_id,
        "name": file_name,
        "title": suggested_title,
        "source_path": str(file_path),
        "n_chunks": len(documents),
        "chunk_size": recommended_chunk_size,
        "embeddings_backend": embeddings_backend,  # Use consistent snake_case
        "uploaded_date": time.time(),
    }
    try:
        with open(destination_dir / "meta.pkl", "wb") as f:
            pickle.dump(metadata, f)
        logging.info(f"Saved metadata for {chat_id}")
    except IOError as e:
        logging.error(f"Failed to save metadata for {chat_id}: {e}")

    # Initialize the chat history file (Fix: Fragile Chat History Persistence)
    save_chat_history(chat_id, [])

    return metadata
