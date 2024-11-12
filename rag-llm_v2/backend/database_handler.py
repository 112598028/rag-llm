from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# from document_processor import DocumentProcessor

embedding_model = "nomic-embed-text"


class ChromaDBHandler:

    def __init__(self, model=embedding_model):
        self.persist_directory = "./lang_db"
        self.embedding_function = OllamaEmbeddings(model=model)
        self.chroma_db = self._initialize_chromadb()

    def _initialize_chromadb(self):
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
        )
    
    def check_documents_existed(self, source_id):
        metadata = self.chroma_db.get(include=["metadatas"])
        existing_ids = set()

        for data in metadata["metadatas"]:
            if "source_id" in data:
                existing_ids.add(data["source_id"])

        return source_id in existing_ids
    
    def update_documents(self, docs):
        source_id = docs[0].metadata.get("source_id")

        if self.check_documents_existed(source_id):
            print("Data already exists in the database.")
            print("Query directly.")
        else:
            print("Adding new documents to the database.")
            self.chroma_db.add_documents(docs)
            print("Successfully added new data.")

    def get_chroma_db(self):
        return self.chroma_db
