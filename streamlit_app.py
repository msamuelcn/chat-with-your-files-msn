# streamlit_app.py

import os
import uuid
import streamlit as st
from dotenv import load_dotenv

# Local imports
from upload import save_uploaded_file
from indexer import (
    list_indices,
    process_and_index,
    start_clear_files,
    save_chat_history,
    load_chat_history,
)
from utils import UPLOAD_DIRECTORY
from chat import get_streaming_chain
import config  # Using centralized config

# Load environment variables from .env
load_dotenv(override=True)

# Run cleanup on app start
try:
    start_clear_files()
except Exception:
    # This will catch serious errors that prevent cleanup but still let the app run
    st.warning("Automatic file cleanup encountered a serious issue. Check server logs.")

# --- Streamlit Session State Initialization ---
# Consistent naming: Using snake_case for session state keys
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = (
        ""  # The unique ID of the current active index/chat
    )

if "active_chat_metadata" not in st.session_state:
    st.session_state.active_chat_metadata = {}  # Metadata for the current active chat

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # Messages for the chat UI

if "chat_history_list" not in st.session_state:
    st.session_state.chat_history_list = (
        []
    )  # List of (user_prompt, ai_response) tuples for LangChain
# --- Streamlit Session State Initialization ---


st.set_page_config(page_title="RAG Chat Modular", layout="wide")
st.title("📚 Chat with your files — Modular Streamlit + LangChain + FAISS")


# --- Sidebar: Chat History and New Chat Button ---
with st.sidebar.container():

    if st.sidebar.button(":heavy_plus_sign: New chat for your file"):
        # Reset all session state variables for a new chat
        st.session_state.active_chat_id = ""
        st.session_state.active_chat_metadata = {}
        st.session_state.display_messages = []
        st.session_state.chat_history_list = []
        st.rerun()

    st.sidebar.header("Chat History")
    chat_history_indices = list_indices()

    # Display buttons for past chat indices
    for chat_meta in chat_history_indices:
        # Use metadata title/name for the button text
        button_label = (
            chat_meta["title"]
            if chat_meta["title"] != "Untitled Document"
            else chat_meta["name"]
        )
        is_topic_selected = st.sidebar.button(
            button_label, key=chat_meta["uid"]
        )  # Renamed variable

        if is_topic_selected:  # Use renamed variable
            # Set active chat and clear existing messages to trigger history load below
            st.session_state.active_chat_id = chat_meta["uid"]
            st.session_state.active_chat_metadata = chat_meta
            st.session_state.display_messages = []
            st.session_state.chat_history_list = []

            st.markdown(
                f"""
                    <style>
                    .st-key-{chat_meta["uid"]} button {{
                        background-color: rgb(85 88 103 / 95%); /* Green */
                        color: white; /* Text color */
                    }}
                    </style>
                    """,
                unsafe_allow_html=True,
            )

            # st.rerun()
# --- End Sidebar ---


st.markdown("---")  # Separator


if st.session_state.active_chat_id == "":

    # --- File Upload and Indexing Area ---
    st.header("1) Upload a PDF or TXT")
    uploaded_file = st.file_uploader(
        "Upload PDF, TXT, or MD", type=["pdf", "txt", "md"], accept_multiple_files=False
    )

    # --- Important Usage Notes (Drafted for Simplicity) ---
    st.markdown("---")
    st.subheader("💡 Important Usage Notes")
    st.info(
        """
        **1. Privacy & Cleanup (Do's)**
        * **Do** treat this application as a short-term tool.
        * **Privacy:** Your uploaded files and chat history are saved **locally** on this machine's hard drive (`/data` directory).
        * **Cleanup:** All files and chats are **automatically deleted after 12 hours** to manage storage space.

        **2. Accuracy & Limitations (Don'ts)**
        * **Don't** expect answers from general web knowledge.
        * **Accuracy:** The chat is a **Retrieval-Augmented Generation (RAG)** system, meaning it will **only** answer based on the text found in your document.
        * **Limitation:** If the answer is not in the document, the model is instructed to say, "I apologize, but I could not find a relevant answer in the uploaded document." It will not guess.

        **3. Performance (Do's)**
        * **Do** be patient during the 'Processing' step.
        * **Indexing Time:** Creating the index (embeddings) for large files can take a few minutes, depending on the file size.
        * **Current Settings:** Indexing uses a default chunk size of **1000 tokens** to balance speed and accuracy.
    """
    )
    st.markdown("---")
    # --- End Usage Notes ---

    # Display a warning about the default chunking settings (Fix: Latency During Indexing)
    st.info(
        f"Indexing will use default chunk size of **{config.DEFAULT_CHUNK_SIZE}** and overlap of **{config.DEFAULT_CHUNK_OVERLAP}** for faster processing."
    )

    process_button = st.button("Process & Index Upload")  # Renamed variable

    if uploaded_file and process_button:  # Use renamed variable
        # File saving logic
        file_id = str(uuid.uuid4())
        filename = uploaded_file.name
        destination_path = UPLOAD_DIRECTORY / f"{file_id}_{filename}"
        save_uploaded_file(uploaded_file, destination_path)
        st.success(f"The file has been saved.")

        with st.spinner("Processing file, generating embeddings, and indexing..."):
            try:
                # Process the file, chunk the text, and create the FAISS index
                metadata = process_and_index(destination_path, file_name=filename)

                st.success(f"Successfully indexed file.")

                # Set the newly created index as the active chat
                st.session_state.active_chat_id = metadata["uid"]
                st.session_state.active_chat_metadata = metadata
                st.session_state.display_messages = []
                st.session_state.chat_history_list = []
                st.rerun()
                st.markdown(
                    f"""
                        <style>
                        .st-key-{metadata["uid"]} button {{
                            background-color: rgb(85 88 103 / 95%); /* Green */
                            color: white; /* Text color */
                        }}
                        </style>
                        """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Failed to process file: {e}")
                # Clean up the file if processing failed
                try:
                    os.remove(destination_path)
                except Exception:
                    pass  # Ignore if cleanup fails
    # --- End Upload Area ---


def handle_user_prompt_submission(chat_id: str, document_title: str, prompt: str):
    """
    Submits a user prompt, streams the response, updates the chat history, and saves history to file.
    """

    # 1. Add user message to display state
    st.session_state.display_messages.append({"role": "user", "content": prompt})

    # 2. Display the user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Stream the AI response
    with st.chat_message("assistant"):
        stream_generator = get_streaming_chain(
            chat_id,
            document_title,
            user_query=prompt,
            chat_history=st.session_state.chat_history_list,
        )
        response = st.write_stream(stream_generator)  # Captures the full response text

    # 4. Update the LangChain-compatible chat history list
    new_history_pair = (prompt, response)
    st.session_state.chat_history_list.append(new_history_pair)

    # 5. Save the updated chat history to file
    save_chat_history(chat_id, st.session_state.chat_history_list)

    # 6. Add assistant message to display state
    st.session_state.display_messages.append({"role": "assistant", "content": response})

    # 7. Rerun to ensure all UI elements reflect the new state
    st.rerun()


# --- Chat Interface ---
if st.session_state.active_chat_id != "":

    metadata = st.session_state.active_chat_metadata
    title = metadata.get("title", "Untitled Document")
    file_name = metadata.get("name", "Unknown File")
    n_chunks = metadata.get("n_chunks", 0)
    chunk_size = metadata.get("chunk_size", config.DEFAULT_CHUNK_SIZE)

    st.markdown(
        f"""
            <style>
            .st-key-{st.session_state.active_chat_id} button {{
                background-color: rgb(85 88 103 / 95%); /* Green */
                color: white; /* Text color */
            }}
            </style>
            """,
        unsafe_allow_html=True,
    )

    # Load existing history from file on first entry to a chat
    if not st.session_state.chat_history_list and not st.session_state.display_messages:
        raw_chat_history = load_chat_history(
            st.session_state.active_chat_id
        )  # Renamed variable

        # chat_history_list is for LangChain (List[Tuple[str, str]])
        # Assuming load_chat_history returns the correct List[Tuple[str, str]] format
        st.session_state.chat_history_list = raw_chat_history

        # display_messages is for Streamlit UI (List[Dict[str, str]])
        st.session_state.display_messages = [
            {"role": role, "content": content}
            for pair in raw_chat_history  # Use renamed variable
            for role, content in zip(["user", "assistant"], pair)
        ]

    st.subheader(f"Title: :newspaper: **{title}**")
    st.markdown(f"Interacting with your uploaded file: **{file_name}**")
    st.markdown(f"Chunks: **{n_chunks}** | Chunk Size: **{chunk_size}**")

    if st.button("Summarize this document for me."):
        # Call the submission handler with a pre-defined prompt
        handle_user_prompt_submission(
            st.session_state.active_chat_id, title, "Summarize this document for me."
        )

    # Display all previous messages in the chat history
    for message in st.session_state.display_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input box
    if prompt := st.chat_input("Ask a question about your document..."):
        handle_user_prompt_submission(st.session_state.active_chat_id, title, prompt)
