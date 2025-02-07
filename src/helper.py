from langchain_huggingface import HuggingFaceEmbeddings
import time
import os
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def download_hugging_face_embeddings():
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding
    

def initialize_pinecone():
    pc=Pinecone()
    index_name = "medchatbot3"  # change if desired

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="enabled",  # Defaults to "disabled"
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    return index

def get_chunks(filepath):
    loader=DirectoryLoader(filepath,glob="*.pdf",
    loader_cls=PyPDFLoader
    )
    docs=loader.load()
    docs=docs[14:]
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    
    chunks=text_splitter.split_documents(docs)
    return chunks