from openai import OpenAI
import os
client = OpenAI()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription.text.strip()  # Trim whitespace

audio_folder = "Audio_Files"

all_transcriptions = []

for filename in os.listdir(audio_folder):
    if filename.endswith(".mp3"):
        file_path = os.path.join(audio_folder, filename)
        # Transcribe audio and append the result to the list
        transcription_text = transcribe_audio(file_path)
        all_transcriptions.append(transcription_text)

complete_text = "\n".join(all_transcriptions)


text_folder = "audio_text"

if not os.path.exists(text_folder):
    os.makedirs(text_folder)

text_file_path = os.path.join(text_folder, "complete_text.txt")

with open(text_file_path, "w") as text_file:
    text_file.write(complete_text)

pdf_folder_path = "audio_text"
documents = []
for file in os.listdir(pdf_folder_path):
    if file.endswith('.txt'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = TextLoader(pdf_path)
        documents.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

print("embedding started")
db = FAISS.from_documents(chunked_documents, embeddings)

db.save_local("Faiss_Audio")

print("embedding done")