"""
processor = ExcelDocumentProcessor("your_excel_file.xlsx")
documents = processor.process_documents()
"""

import hashlib

# basic tool
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter

# excel
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


class DocumentProcessor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.source_id = self._get_file_hash()

    def _get_file_hash(self):
        hash_md5 = hashlib.md5()
        with open(self.file_path, "rb") as f:
            for doc in iter(lambda: f.read(4096), b""):
                hash_md5.update(doc)
        return hash_md5.hexdigest()

    def excel_documents_processor(self):
        loader = UnstructuredExcelLoader(self.file_path, mode="elements")
        documents = loader.load()

        filtered_documents = filter_complex_metadata(documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=20
        )

        docs = text_splitter.split_documents(filtered_documents)

        for doc in docs:
            doc.metadata["source_id"] = self.source_id

        return docs
