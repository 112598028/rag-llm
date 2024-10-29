"""
make sure to install first: pip install langchain openai chromadb streamlit pandas
if reading pdf, install: pip install pypdf
if reading excel, install: pip install --upgrade --quiet langchain-community unstructured openpyxl

"""

import streamlit as st

# from langchain_community.embeddings import OllamaEmbeddings
## old-one
from langchain_ollama import OllamaEmbeddings

# from langchain.embeddings import HuggingFaceEmbeddings
## another embeddings model

from langchain_community.document_loaders import UnstructuredExcelLoader

# from langchain.vectorstores import Chroma
## old-one
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import CharacterTextSplitter

from langchain_ollama.llms import OllamaLLM

from langchain.retrievers import Retriever

from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.prompts import PromptTemplate


import pandas as pd
import os

# setup DB


# read file and put into DB
def setup_database():
    file_path = "./QA50.xlsx"
    loader = UnstructuredExcelLoader(file_path, mode="elements")
    documents = loader.load()
    filtered_documents = filter_complex_metadata(documents)

    text_splitter = CharacterTextSplitter(chunk_size=15, chunk_overlap=5)
    docs = text_splitter.split_documents(filtered_documents)
    # embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    db = Chroma.from_documents(
        documents=docs,
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
        persist_directory="./lang_db",
    )

    return db.as_retriever()


def user_input(user_input, retriever: Retriever):
    return None


# ans = db.similarity_search("交通違規", k=3)
# print(ans)
# https://www.youtube.com/watch?v=abMwFViFFhI
