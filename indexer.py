import os
import uuid
import pickle
import time
import shutil
import json

from typing import List
from dotenv import load_dotenv
from pathlib import Path
from utils import INDEX_DIR

from pydantic import BaseModel, Field
from upload import extract_text
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit_cookies_controller import CookieController
from datetime import datetime, timedelta


# Initialize cookie manager
controller = CookieController()

try:
    chat_uid = controller.get("chat_uid_list")
    chat_uid_list = json.loads(chat_uid)

    chat_threads_cookies = controller.get("chat_threads")
    chat_threads_list = json.loads(chat_threads_cookies)
except:
    chat_uid_list = []
    expiration = datetime.now() + timedelta(days=0.5)
    controller.set("chat_uid_list", json.dumps(chat_uid_list), expires=expiration)

    chat_threads_list = {}
    expiration = datetime.now() + timedelta(days=0.5)
    controller.set("chat_threads", json.dumps(chat_threads_list), expires=expiration)


# Load environment variables from .env
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_embedding_name = "text-embedding-3-small"
llm_model_name = "gpt-4o-mini"


def list_indices():
    out = []
    try:
        temp_chat_uid = controller.get("chat_uid_list")
        temp_chat_uid_list = json.loads(temp_chat_uid)

        for d in INDEX_DIR.iterdir():
            if d.is_dir():
                p = d / "meta.pkl"
                if p.exists():
                    import pickle
                with open(p, "rb") as f:
                    meta = pickle.load(f)

                if meta["uid"] in temp_chat_uid_list:
                    out.append(meta)

    except RuntimeError as e:
        return []
    return out


def start_clear_files():
    print("start_clear_files")
    try:
        for d in INDEX_DIR.iterdir():
            if d.is_dir():
                p = d / "meta.pkl"
                if p.exists():
                    import pickle
                with open(p, "rb") as f:
                    meta = pickle.load(f)

                timestamp = meta["uploaded_date"]
                current_time = time.time()
                half_day_in_seconds = 0.5 * 24 * 60 * 60

                if current_time - timestamp > half_day_in_seconds:
                    file_path = meta["source_path"]
                    os.remove(file_path)
                    print("deleted file: ", file_path)

                    index_path = "data\\indices\\" + meta["uid"]
                    if os.path.isdir(index_path):
                        shutil.rmtree(index_path)
                    print("deleted index: ", index_path)

    except RuntimeError as e:
        print("An error occured", e)


def heuristic_chunk_size(text: str) -> int:
    n_chars = len(text)
    if n_chars < 5_000:
        return 1000
    if n_chars < 20_000:
        return 1200
    if n_chars < 100_000:
        return 1500
    return 2000


def get_embeddings(backend: str = "openai"):
    if backend == "openai":
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # or "text-embedding-3-large"
            api_key=OPENAI_API_KEY,
        )

        sample = "test"
        vector = embeddings.embed_query(sample)
        embedding_dim = len(vector)

        return embeddings, embedding_dim


def analyze_text_sample_with_llm(
    text_sample: str,
    embedding_dim: int = 384,
    llm=None,
):
    """
    Ask the LLM to:
    1. Recommend an optimal chunk size for embeddings based on the model's embedding dimension.
    2. Recommend an optimal overlap size for smooth semantic continuity.
    3. Suggest the best title for the text excerpt.

    Args:
        text_sample (str): Example document content.
        embedding_dim (int): The number of dimensions of the embedding model.
        llm: The language model instance to query.

    Returns:
        dict: {
            "recommended_chunk_size": int,
            "recommended_overlap_chunk_size": int,
            "suggested_title": str
        }
    """

    if llm is None:
        raise ValueError(
            "LLM model is not initialized. Please check your configuration or API key."
        )

    # --- Prompt Template ---
    template = """
You are an expert in text processing and retrieval optimization.

Given the following document excerpt and the embedding model dimension of {dimensions}:

1. Recommend the **optimal chunk size** in characters for text splitting that balances semantic coherence and retrieval accuracy.
2. Recommend the **optimal overlap size** in characters to ensure smooth context continuity between chunks.
3. Suggest a **concise and meaningful title** for the excerpt (under 10 words).

Return the results in structured form, using integers for numeric values.

Document excerpt:
----------------
{excerpt}
----------------
"""

    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(excerpt=text_sample, dimensions=embedding_dim)

    # --- Structured Output Definition ---
    class TextSampleAnalysis(BaseModel):
        recommended_chunk_size: int = Field(
            description="Optimal chunk size in characters for text splitting."
        )
        recommended_overlap_chunk_size: int = Field(
            description="Optimal overlap size in characters to maintain context."
        )
        suggested_title: str = Field(
            description="Concise, informative title for the text excerpt."
        )

    llm_with_structure = llm.with_structured_output(TextSampleAnalysis)
    resp = llm_with_structure.invoke(messages)

    # --- Safe Parsing ---
    try:
        return {
            "recommended_chunk_size": int(resp.recommended_chunk_size),
            "recommended_overlap_chunk_size": int(resp.recommended_overlap_chunk_size),
            "suggested_title": resp.suggested_title.strip(),
        }
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return {
            "recommended_chunk_size": 1000,
            "recommended_overlap_chunk_size": 200,
            "suggested_title": "Untitled Excerpt",
        }


def chunk_text(text: str, chunk_size: int, chunk_overlap: int = 200) -> List[Document]:
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


def save_vectorstore(faiss_index: FAISS, dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)
    faiss_index.save_local(str(dirpath))


def store_indices_cookies(uid):
    chat_uid_list.append(uid)
    expiration = datetime.now() + timedelta(days=0.5)
    controller.set("chat_uid_list", json.dumps(chat_uid_list), expires=expiration)

    chat_threads_cookies = controller.get("chat_threads")
    chat_threads_list = json.loads(chat_threads_cookies)
    chat_threads_list[uid] = []
    controller.set("chat_threads", json.dumps(chat_threads_list), expires=expiration)


def store_indices_chat_thread_cookies(uid, chat_threads: List):
    chat_threads_list[uid] = chat_threads
    expiration = datetime.now() + timedelta(days=0.5)
    controller.set("chat_threads", json.dumps(chat_threads_list), expires=expiration)


def load_chat_threads(uid):
    chat_threads_cookies = controller.get("chat_threads")
    chat_threads_list = json.loads(chat_threads_cookies)

    chat_threads_list = chat_threads_list[uid]

    return chat_threads_list


def process_and_index(file_path: Path, name: str, embeddings_backend="openai") -> str:
    text = extract_text(file_path)
    # choose chunk size heuristically or using LLM guidance (optional)

    sample = text[:2000]
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=llm_model_name,
        streaming=False,
        temperature=0.0,
    )
    embeddings, embeddings_dimension = get_embeddings()

    analysis_result = analyze_text_sample_with_llm(
        sample, embedding_dim=embeddings_dimension, llm=llm
    )
    recommended_chunk_size = analysis_result["recommended_chunk_size"]
    recommended_overlap_chunk_size = analysis_result["recommended_overlap_chunk_size"]
    suggested_title = analysis_result["suggested_title"]

    docs = chunk_text(
        text,
        chunk_size=recommended_chunk_size,
        chunk_overlap=recommended_overlap_chunk_size,
    )

    store = FAISS.from_documents(docs, embeddings)

    # Save FAISS index + docstore metadata
    store.save_local("vectorstores/my_index")

    uid = str(uuid.uuid4())
    dest = INDEX_DIR / uid
    save_vectorstore(store, dest)

    timestamp = time.time()

    store_indices_cookies(uid)

    meta = {
        "uid": uid,
        "name": name,
        "title": suggested_title,
        "source_path": str(file_path),
        "n_chunks": len(docs),
        "chunk_size": recommended_chunk_size,
        "embeddings_backend": embeddings_backend,
        "uploaded_date": timestamp,
    }
    with open(dest / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    return meta
