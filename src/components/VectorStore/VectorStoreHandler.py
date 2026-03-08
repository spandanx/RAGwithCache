from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
import logging

class DocumentLoader:
    def __init__(self, url):
        self.url = url

    def document_reader(self) -> List[Document]:
        loader = WebBaseLoader(self.url)
        docs = loader.load()
        return docs

    def split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        return splitter.split_documents(documents)

class VectorStore:
    def __init__(self, embedding_model, collection_name, qdrant_url, qdrant_key, vector_dimension):
        self.embedding_model = embedding_model
        # self.vector_db_client = QdrantClient(":memory:")
        self.vector_db_client = QdrantClient(
                url = qdrant_url,
                api_key = qdrant_key
            )
        self.vector_store = None
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        logging.info("Called VectorStore constructor()")

    def create_vectorstore(self):
        # self.vector_db_client = QdrantClient(":memory:")
        self.vector_db_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_dimension, distance=Distance.COSINE),
        )
        logging.info("Called create_vectorstore()")

    def load_vector_store(self):
        self.vector_store = QdrantVectorStore(
            client=self.vector_db_client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
        )
        logging.info("Called load_vector_store()")


    def ingest_data(self, documents):
        self.vector_store.add_documents(documents)
        logging.info("Called ingest_data()")

    def check_if_collection_exists(self):
        logging.info("Called check_if_collection_exists()")
        return self.vector_db_client.collection_exists(self.collection_name)


    def get_retriever(self):
        logging.info("Called get_retriever()")
        return self.vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k": 10})