import os
import re
from langchain_community.document_loaders import WikipediaLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Function to filter out non-English words
def filter_english_words(text):
    # Use regular expression to match English words
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    # Join the English words into a single string
    filtered_text = ' '.join(english_words)
    return filtered_text

# List of topics to retrieve data from
topics = ["HUNTER X HUNTER", "Harry Potter", "Pettai"]

# Path to the directory for saving text files
output_folder = "web_text"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize an empty string to store concatenated text content
all_text_content = ""

# Iterate over each topic
for topic in topics:
    # Load documents from Wikipedia for the current topic
    docs = WikipediaLoader(query=topic, load_max_docs=1).load()

    # Check if any documents were retrieved
    if docs:
        # Extract page content and append to the accumulated text
        page_content = docs[0].page_content.strip()
        # Filter out non-English words
        english_content = filter_english_words(page_content)
        all_text_content += english_content + "\n\n"

# Define the full file path for the output file
output_file_path = os.path.join(output_folder, "all_topics.txt")

# Write the concatenated text content to the output file
with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(all_text_content)

print(f"All topics data saved in '{output_file_path}'")

pdf_folder_path = "web_text"
documents = []

# Iterate through text files in the folder
for file in os.listdir(pdf_folder_path):
    if file.endswith('.txt'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = TextLoader(pdf_path)
        documents.extend(loader.load())

# Instantiate a text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

# Split documents into chunks
chunked_documents = text_splitter.split_documents(documents)

# Instantiate OpenAI embeddings
embeddings = OpenAIEmbeddings()

print("Embedding started")

# Create FAISS index from documents
db = FAISS.from_documents(chunked_documents, embeddings)

# Save the FAISS index locally
db.save_local("Faiss_WEB")

print("Embedding done")
