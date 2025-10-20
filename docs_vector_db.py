from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup, SoupStrainer
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

def create_docs_vector_store(vector_store_path: str = "./chroma_docs_db", base_url: str ="https://docs.manim.community/en/stable/"):

    strainer = SoupStrainer(["article", "main", "div"], attrs={"class": ["content", "document", "body"]})

    loader = RecursiveUrlLoader(
        url=base_url,
        max_depth=2,
        extractor=lambda html: BeautifulSoup(html, "lxml", parse_only=strainer).get_text(),
        prevent_outside=True,
        use_async=True,
        timeout=30,
        check_response_status=True
    )

    documents = loader.load()
    print(f"Loaded {len(documents)} document pages")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )

    all_splits = text_splitter.split_documents(documents)

    print(f"split the documents into {len(all_splits)} sub documents")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory=vector_store_path
    )

    batch_size = 100
    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i:i+batch_size]
        print(f"Adding batch {i//batch_size + 1}/{(len(all_splits)+batch_size-1)//batch_size}....")
        vector_store.add_documents(batch)

    print(f"Vector store made at {vector_store_path}")

    return vector_store

if __name__ == "__main__":
    create_docs_vector_store()