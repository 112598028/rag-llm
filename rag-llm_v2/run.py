from backend.document_processor import DocumentProcessor
from backend.database_handler import ChromaDBHandler
from backend.rag import RAG

import streamlit as st


def main():

    file_pahth = "./data/QA50.xlsx"

    # Initialize the components
    processor = DocumentProcessor(file_pahth)
    db_handler = ChromaDBHandler()
    rag = RAG(db_handler)

    # Process documents and add them to the database
    docs = processor.documents_processor()
    db_handler.update_documents(docs)

    st.title("RAG問答系統")

    query = st.text_input("請輸入你的查詢：")

    if st.button("提交"):
        if query:
            # Query the database using RAG
            response = rag.get_response(query)
            st.write("回答：", response)
        else:
            st.warning("請先輸入查詢內容")


if __name__ == "__main__":
    main()
