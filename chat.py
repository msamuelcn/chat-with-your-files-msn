# chat.py

import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import (
    ConversationalRetrievalChain,
)  # Unused but common in RAG, keeping for reference
from langchain_core.callbacks import BaseCallbackHandler
from langchain_classic.tools.retriever import create_retriever_tool
from langchain.agents import create_agent  # Updated import path for agent
from langchain_core.messages import HumanMessage, AIMessageChunk

# Removed: from langchain.tools import tool (unused)
# Removed: from langchain_core.prompts import ChatPromptTemplate (unused, prompt is f-string)

# Local imports
from utils import INDEX_DIRECTORY
import config  # Using centralized config

# Load environment variables from .env
load_dotenv(override=True)

# --- Configuration Constants (imported from config, standardized to snake_case for local use) ---
# NOTE: The values are still sourced from config.py's CAPITALIZED names, but assigned to lowercase locals.
openai_api_key = config.OPENAI_API_KEY
embeddings_backend: str = config.EMBEDDINGS_BACKEND
openai_embedding_model = config.OPENAI_EMBEDDING_MODEL
huggingface_embedding_model = config.HUGGINGFACE_EMBEDDING_MODEL
llm_model_name = config.LLM_MODEL_NAME
# --- Configuration Constants ---


class StreamlitTokenStreamer(BaseCallbackHandler):
    """A simple callback handler that collects tokens."""

    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token


def get_embeddings_model(backend: str):
    """Initializes and returns the appropriate embedding model."""
    if backend == "openai":
        return OpenAIEmbeddings(
            model=openai_embedding_model, openai_api_key=openai_api_key
        )
    elif backend == "huggingface":
        return HuggingFaceEmbeddings(model_name=huggingface_embedding_model)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")


def load_vector_store(index_id: str, embeddings):
    """Loads the FAISS vector store from disk."""
    index_path = INDEX_DIRECTORY / index_id
    if not index_path.exists():
        raise RuntimeError(f"Index ID '{index_id}' does not exist at {index_path}")

    # Use str(index_path) for compatibility with FAISS.load_local
    return FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )


def get_streaming_chain(
    index_id: str,
    document_title: str,
    user_query: str = None,
    chat_history=[],
):
    """
    Returns a streaming agent that uses retrieval tool.
    """
    # 1. Load Embeddings and Vector Store
    embeddings = get_embeddings_model(config.EMBEDDINGS_BACKEND)
    vector_store = load_vector_store(index_id, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 2. Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_documents",
        f"Search and retrieve information from {document_title}. "
        "Use this tool when you need to find specific information from the document.",
    )

    # 3. Format chat history for system prompt
    history_string = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])

    # 4. Create system prompt
    system_prompt = f"""
You are an expert, helpful, and courteous document analysis assistant.
Your primary goal is to provide **accurate, concise, and complete** answers based *only* on the provided document context.

**Current Document Title**: {document_title}

**Instructions**:
1. **Use the retrieve_documents tool** to search for relevant information in the document.
2. **Strictly adhere to the context** retrieved from the tool.
3. If the answer cannot be found in the retrieved context, clearly state: "I apologize, but I could not find a relevant answer in the uploaded document." Do not try to guess or use general knowledge.
4. Maintain a polite and professional tone.
5. If the user's query is purely conversational (e.g., "Hi," "Thanks"), respond appropriately without needing the retrieval tool.
6. Base your response on the current chat history for continuity.

**Chat History**:
{history_string}
"""
    # 5. Initialize LLM
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model=llm_model_name,
        streaming=True,
        temperature=0.0,
    )

    # 6. Create agent with retriever tool
    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=system_prompt,
    )

    # Removed unused 'response' variable initialization

    # 7. Stream the response
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=user_query)]}, stream_mode="messages"
    ):
        # Yield content from AI messages
        ai_message_chunk, _ = chunk
        if isinstance(ai_message_chunk, AIMessageChunk) and ai_message_chunk.content:
            yield ai_message_chunk.content

    # Removed unused print statements
