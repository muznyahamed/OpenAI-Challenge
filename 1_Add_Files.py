import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader
import docx
import tempfile   # temporary file
from langchain.document_loaders.csv_loader import CSVLoader  # using CSV loaders


from langchain.text_splitter import CharacterTextSplitter

import os

st.set_page_config(page_title="Add Files Here", page_icon="")

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_csv_text(file):
    return "a"

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    return knowledge_base
st.markdown("# Upload Files")
uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
uploaded_csv = st.file_uploader("Upload File", type="csv")
if uploaded_csv:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_csv.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(data, embeddings)

process = st.button("Process")

if process:
    files_text = get_files_text(uploaded_files)
    text_chunks = get_text_chunks(files_text)
    vetorestore2 = get_vectorstore(text_chunks)
    vetorestore2.save_local("NewFiles")
    st.write("ready")
    if uploaded_csv:
        db.save_local("NewFiles")
        st.write("ready")


