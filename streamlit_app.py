"""
streamlit_app.py
Main Streamlit app. Uses the modular components in upload.py, indexer.py, chat.py, utils.py.
"""

import os
import uuid
import pickle
import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from upload import save_uploaded_file

from indexer import (
    list_indices,
    process_and_index,
    start_clear_files,
    store_indices_chat_thread_cookies,
    load_chat_threads,
)
from utils import DATA_DIR, INDEX_DIR, UPLOAD_DIR
from chat import get_streaming_chain
from streamlit_cookies_controller import CookieController

# Load environment variables from .env
load_dotenv(override=True)

start_clear_files()

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = ""

if "active_chat_attb" not in st.session_state:
    st.session_state.active_chat_attb = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_thread" not in st.session_state:
    st.session_state.chat_thread = []


st.set_page_config(page_title="RAG Chat Modular", layout="wide")
st.title("📚 Chat with your files — Modular Streamlit + LangChain + FAISS")


with st.sidebar.container():

    if st.sidebar.button(":heavy_plus_sign: New chat for your file"):
        st.session_state.active_chat_id = ""
        st.session_state.active_chat_attb = {}

    st.sidebar.header("Chat History")
    chat_history = list_indices()
    choices = {f"{m['name']} ({m['n_chunks']} chunks)": m["uid"] for m in chat_history}

    for ch in chat_history:
        selected_topic = st.sidebar.button(ch["title"], key=ch["uid"])
        if selected_topic:
            st.session_state.active_chat_id = ch["uid"]
            st.session_state.active_chat_attb = ch
            st.session_state.messages = []
            st.session_state.chat_thread = []

            st.markdown(
                f"""
                    <style>
                    .st-key-{ch["uid"]} button {{
                        background-color: rgb(85 88 103 / 95%); /* Green */
                        color: white; /* Text color */
                    }}
                    </style>
                    """,
                unsafe_allow_html=True,
            )

st.markdown("---")  # Separator

if st.session_state.active_chat_id == "":

    # Upload area
    st.header("1) Upload a PDF or TXT")
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT", type=["pdf", "txt", "md"], accept_multiple_files=False
    )

    process_btn = st.button("Process & Index Upload")

    # Process and index upload
    if uploaded_file and process_btn:
        file_id = str(uuid.uuid4())
        filename = uploaded_file.name
        dest = UPLOAD_DIR / f"{file_id}_{filename}"
        save_uploaded_file(uploaded_file, dest)
        st.success(f"Saved upload to {dest}")

        try:
            meta = process_and_index(
                dest,
                name=filename,
            )
            st.success(f"Indexed file. Index UID: {meta['uid']}")

            st.session_state.active_chat_id = meta["uid"]
            st.session_state.active_chat_attb = meta
            st.session_state.messages = []
            st.session_state.chat_thread = []

            st.markdown(
                f"""
                    <style>
                    .st-key-{meta["uid"]} button {{
                        background-color: rgb(85 88 103 / 95%); /* Green */
                        color: white; /* Text color */
                    }}
                    </style>
                    """,
                unsafe_allow_html=True,
            )
            st.rerun()

        except Exception as e:
            st.error(f"Failed to process file: {e}")


def submit_a_prompt(active_chat_id: str, prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # print(st.session_state.active_chat_id)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream_generator = get_streaming_chain(
            active_chat_id, query=prompt, chat_thread=st.session_state.chat_thread
        )
        response = st.write_stream(stream_generator)

    insert_thread = (prompt, response)
    st.session_state.chat_thread.append(insert_thread)

    store_indices_chat_thread_cookies(active_chat_id, st.session_state.chat_thread)

    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()


if "active_chat_id" in st.session_state and st.session_state.active_chat_id != "":

    title = st.session_state.active_chat_attb["title"]
    file_name = st.session_state.active_chat_attb["name"]
    n_chunks = st.session_state.active_chat_attb["n_chunks"]
    chunk_size = st.session_state.active_chat_attb["chunk_size"]

    chat_threads_list = load_chat_threads(st.session_state.active_chat_id)

    st.session_state.chat_thread = [(c[0], c[1]) for c in chat_threads_list]

    chat_threads_list = [
        {"role": role, "content": content}
        for pair in chat_threads_list
        for role, content in zip(["user", "assistant"], pair)
    ]
    st.session_state.messages = chat_threads_list

    st.subheader("Title: :newspaper:  " + title)
    st.markdown("You'll be interacting with your uploaded file: " + file_name)
    st.markdown(
        "Number of chunks: " + str(n_chunks) + "\t\t chunk size: " + str(chunk_size)
    )

    if st.button("Summarize this document for me."):
        submit_a_prompt(
            st.session_state.active_chat_id, "Summarize this document for me."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        submit_a_prompt(st.session_state.active_chat_id, prompt)
