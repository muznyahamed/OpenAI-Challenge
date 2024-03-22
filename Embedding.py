import bs4
from langchain import hub
import ray

from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from langchain_text_splitters import RecursiveCharacterTextSplitter
# Load, chunk and index the contents of the blog.
from langchain_community.document_loaders import TextLoader

bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
embeddings = FastEmbedEmbeddings()


pdf_folder_path = "book"
documents = []
for file in os.listdir(pdf_folder_path):
    if file.endswith('.txt'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = TextLoader(pdf_path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

print("embedding started")
vectordb = FAISS.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory="Store"
    )
print("embedding ended")
