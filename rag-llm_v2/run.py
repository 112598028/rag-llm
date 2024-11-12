from backend.document_processor import DocumentProcessor
from backend.database_handler import ChromaDBHandler
from backend.rag import RAG


def main():

    file_pahth = "./data/QA50.xlsx"

    # Initialize the components
    processor = DocumentProcessor(file_pahth)
    db_handler = ChromaDBHandler()
    rag = RAG(db_handler)

    # Process documents and add them to the database
    docs = processor.excel_documents_processor()
    db_handler.update_documents(docs)

    # Query the database using RAG
    query = "喪失國籍怎麼處理?"
    response = rag.get_response(query)
    print(response)


if __name__ == "__main__":
    main()
