import hashlib

from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter


file_path = "./QA50.xlsx"

hash_md5 = hashlib.md5()
with open(file_path, "rb") as f:
    for doc in iter(lambda: f.read(4096), b""):
        hash_md5.update(doc)

print("MD5:", hash_md5.hexdigest())


loader = UnstructuredExcelLoader(file_path, mode="elements")
documents = loader.load()

filtered_documents = filter_complex_metadata(documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)

docs = text_splitter.split_documents(filtered_documents)

for doc in docs:
    doc.metadata["source_id"] = hash_md5.hexdigest()


print("doc: ", docs[4])
# print(docs[0].metadata['element_id'])
