import streamlit as st
import hashlib

from langchain_ollama import OllamaEmbeddings

from langchain_community.document_loaders.excel import UnstructuredExcelLoader

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFacePipeline  
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import torch
import os

def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for doc in iter(lambda: f.read(4096), b""):
            hash_md5.update(doc)
    return hash_md5.hexdigest()



def excel_to_documents(file_path):
    loader = UnstructuredExcelLoader(file_path, mode="elements")
    documents = loader.load()

    filtered_documents = filter_complex_metadata(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)

    docs = text_splitter.split_documents(filtered_documents)
    
    return docs


def docs_to_db(docs):
    db = Chroma.from_documents(
        documents=docs,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./lang_db",
    )
    return db


def retrieve_db(vectorstore, query):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    results = retriever.invoke(query)
    return results


def rag_response(vectorstore, question):
    retrieve_data = retrieve_db(vectorstore, question)

    prompt_template = PromptTemplate(
        input_variables=["question", "retrieve_data"],
        template="Using this data: {retrieve_data}. Respond to this prompt in Chinese: {question}",
    )
     
    model_id = "MediaTek-Research/Breeze-7B-Instruct-v1_0"  
    tokenizer = AutoTokenizer.from_pretrained(model_id)  
    model = AutoModelForCausalLM.from_pretrained(model_id)  
    
    pipe = pipeline(  
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=10, 
        # device=0  
    )  
    hf = HuggingFacePipeline(pipeline=pipe)

    # llm = ChatOpenAI(
    #     model_name="ycchen/breeze-7b-instruct-v1_0",
    #     # api_key="EMPTY",
    #     temperature=0.0,
    # )

    rag_chain = prompt_template | hf | StrOutputParser()
    response = rag_chain.invoke({"question": question, "retrieve_data": retrieve_data})

    return response


if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    file_path = "./QA50.xlsx"
    doc = excel_to_documents(file_path)
    vector_store = docs_to_db(doc)
    question = "喪失國籍怎麼處理?"
    response = rag_response(vector_store, question)
    print(response)

    torch.cuda.empty_cache()