import streamlit as st
from langchain.llms import OpenAI
import pickle
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from langchain.memory import ConversationBufferWindowMemory
st.set_page_config(
    page_title="ChatGPT Clone",
    layout="wide"
)

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are a very kindl and friendly AI assistant. You are
    currently having a conversation with a human. Answer the questions
    in a kind and friendly tone with some sense of humor.

    chat_history: {chat_history},
    Human: {question}
    AI:"""
)

from langchain import PromptTemplate, HuggingFaceHub, LLMChain


def add_model(model_name, model_repo):
    new_model_info = f"{model_name},{model_repo}\n"
    with open("models.txt", "a") as text_file:
        text_file.write(new_model_info)
models = {
    'GPT4': ChatOpenAI(model_name="gpt-4", temperature=0),
    'GPT3.5': ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0),
    'Google Flan': HuggingFaceHub(repo_id="google/flan-t5-large",
                                  model_kwargs={"temperature": 0.9, "max_length": 1000})
}

# Load existing models from a text file if it exists
if os.path.exists("models.txt"):
    with open("models.txt", "r") as text_file:
        for line in text_file:
            model_name, model_repo = line.strip().split(",")
            models[model_name] = HuggingFaceHub(repo_id=model_repo,
                                  model_kwargs={"temperature": 0.9, "max_length": 1000})

# Initialize the selected model
# llm = ChatOpenAI()

# Form to add a new model
with st.sidebar.form("add_model"):
    st.write("Add New Model")
    new_model_name = st.text_input('Model Name', 'E.g., Huggingface')
    new_model_repo = st.text_input('Model ID', 'E.g., Huggingface')
    submitted = st.form_submit_button("Add Model")
    if submitted:
        add_model(new_model_name, new_model_repo)
        st.success("New model added successfully!")

# Selectbox to choose the model
Model = st.sidebar.selectbox(
    'Select your Model',
    list(models.keys())
)

# Get the selected model
llm = models.get(Model)

# Check if the selected model exists
if llm is None:
    st.error("No model selected. Please select a model from the list.")

llm_summary= HuggingFaceHub(repo_id="facebook/bart-large-cnn", model_kwargs={"temperature": 0.9, "max_length": 1000})

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4)
llm_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)

embedding_function = OpenAIEmbeddings()

encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity



new_db0 = FAISS.load_local("Faiss_Audio_PDF_TXT_WEB", embedding_function, allow_dangerous_deserialization=True)
if os.path.exists("NewFiles"):
    new_db1 = FAISS.load_local("NewFiles", embedding_function, allow_dangerous_deserialization=True)
    new_db0.merge_from(new_db1)
retriever = new_db0.as_retriever()
template = """
    {context}

    Question: {question}
    in the question have added the previous chat history along with the last chat make a better reply using the context and history chat
    
    
    """


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template(template)
conversation = ""

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser()
)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there"}
    ]

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)
multi_llm = OpenAI(n=4, best_of=4)



conversation = ""
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            # ai_response = llm_chain.predict(question=user_prompt)
            conversation = ""

            # Iterate through the list of messages
            for message in st.session_state.messages:
                # Concatenate the role and content of each message
                conversation += message['role'] + ": " + message['content'] + "\n"

            user_prompt = user_prompt + "past conversation history" + conversation

            ai_response = rag_chain.invoke(user_prompt)
            st.write(ai_response)

            print(st.session_state.messages)

    new_ai_message = {"role": "assistant", "content": ai_response}

    st.session_state.messages.append(new_ai_message)