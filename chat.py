import os
import streamlit
import time
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from utils import INDEX_DIR


# Load environment variables from .env
load_dotenv(override=True)

# Access the key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model_embedding_name = "text-embedding-3-small"
llm_model_name = "gpt-4o-mini"
embeddings_backend: str = "openai"


class StreamlitTokenStreamer(BaseCallbackHandler):
    """A simple callback handler that collects tokens and provides them via a
    buffer.
    The get_output() method returns the accumulated text.
    """

    def __init__(self):
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # This method is invoked by LangChain for each new token when
        streaming = True
        self.text += token


def _get_embeddings(backend: str):
    if backend == "openai":
        return OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY
        )
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/allMiniLM-L6-v2")


def _load_store(uid: str, embeddings):
    p = INDEX_DIR / uid
    if not p.exists():
        raise RuntimeError("Index does not exist")
    return FAISS.load_local(str(p), embeddings, allow_dangerous_deserialization=True)


def get_streaming_chain(
    uid: str,
    query: str = None,
    parent_container: streamlit.delta_generator.DeltaGenerator = None,
    chat_thread=[],
):
    """
    Returns a generator that yields incremental text chunks as the LLM streams tokens.
    Usage:
        for chunk in get_streaming_chain(...):
            container.markdown(chunk)
    """
    embeddings = _get_embeddings(embeddings_backend)
    store = _load_store(uid, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 5})

    # Streaming setup
    # token_streamer = StreamlitCallbackHandler(parent_container=parent_container)

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=llm_model_name,
        streaming=True,
        # callbacks=[token_streamer],
        temperature=0.0,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    # Run the chain - note: with streaming enabled, the LLM will call our callback for tokens
    # We'll run the chain synchronously, and yield the token buffer intermittently.
    # Because we cannot yield tokens directly from the callback, we poll the token_streamer buffer.
    # Start chain in blocking call
    # Run the chain

    result = chain.invoke({"question": query, "chat_history": chat_thread})
    # After chain completes, yield the full accumulated text
    # time.sleep(0.05)
    yield result["answer"]

    # Optionally provide sources as markdown
    # sources = result.get("source_documents", [])
    # if sources:
    #     src_md = "\n\n**Sources:**\n"
    #     for s in sources:
    #         src_md += f"- chunk {s.metadata.get('chunk', '?')}\n"
    #     yield src_md
