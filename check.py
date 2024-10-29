import chromadb

client = chromadb.Client()
version = client.get_version()
print(f"ChromaDB version: {version}")
