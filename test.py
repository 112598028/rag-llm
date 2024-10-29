import streamlit as st
import ollama
import chromadb
from chromadb.config import Settings
import pandas as pd
import os


# @st.cache_resource
def create_chromadb_client():
    db_path = "/home/aclab/peggyyu/chroma_db"
    os.makedirs(db_path, exist_ok=True)

    # settings = Settings(
    #     chroma_db_impl="sqlite",
    #     persist_directory=db_path
    # )
    # return chromadb.Client(settings)
    # return chromadb.EphemeralClient()
    return chromadb.PersistentClient(db_path)


def initialize():
    if "already_executed" not in st.session_state:
        st.session_state.already_executed = False

    if not st.session_state.already_executed:
        setup_database()


def setup_database():
    client = create_chromadb_client()
    file_path = "/home/aclab/peggyyu/QA50.xlsx"
    documents = pd.read_excel(file_path, header=None)

    collection = client.get_or_create_collection(name="demodocs")

    for index, content in documents.iterrows():
        response = ollama.embeddings(model="mxbai-embed-large", prompt=content[0])
        collection.add(
            ids=[str(index)], embeddings=[response["embedding"]], documents=[content[0]]
        )

    st.session_state.already_executed = True
    st.session_state.collection = collection


def main():
    initialize()
    st.title("first test of rag-llm")
    user_input = st.text_area("input your question?", "")

    if st.button("go"):
        if user_input:
            handle_user_input(user_input, st.session_state.collection)
        else:
            st.warning("you forget to input your question!")


def handle_user_input(user_input, collection):
    response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")
    results = collection.query(query_embeddings=[response["embedding"]], n_results=3)
    data = results["documents"][0]

    output = ollama.generate(
        model="ycchen/breeze-7b-instruct-v1_0",
        prompt=f"Using this data: {data}. Respond to this prompt and use chinese: {user_input}",
    )

    st.text("answer: ")
    st.write(output["response"])


if __name__ == "__main__":
    main()


# open the ollama service: ollama service
# streamlit test.py
