"""
processor = ExcelDocumentProcessor("your_excel_file.xlsx")
documents = processor.process_documents()
"""

import hashlib
import re

# basic tool
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter

# excel
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

# word
from langchain_community.document_loaders.word_document import (
    UnstructuredWordDocumentLoader,
)


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

    def _select_loader(self):
        if self.file_path.endswith(".xlsx") or self.file_path.endswith(".xls"):
            return UnstructuredExcelLoader(self.file_path, mode="elements")
        elif self.file_path.endswith(".docx"):
            return UnstructuredWordDocumentLoader(self.file_path, mode="elements")
        else:
            raise ValueError(
                "Unsupported file type. Please use .xlsx, .xls or .docx files."
            )

    def _clean_text(self, text):
        cleaned_text = re.sub(r"(\n+|\s+)", " ", text)
        return cleaned_text.strip()

    def documents_processor(self):
        loader = self._select_loader()
        documents = loader.load()
        for docs in documents:
            docs.page_content = self._clean_text(docs.page_content)
        filtered_documents = filter_complex_metadata(documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
        docs = text_splitter.split_documents(filtered_documents)
        for doc in docs:
            doc.metadata["source_id"] = self.source_id
        return docs
