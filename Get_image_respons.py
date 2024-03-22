import streamlit as st
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from openai import OpenAI
client = OpenAI()
def user_message(message):
    st.markdown(f'<div class="user-message" style="display: flex; justify-content: flex-end; padding: 5px;">'
                f'<div style="background-color: #196b1c; color: white; padding: 10px; border-radius: 10px; font-size:18px; margin-bottom:10px; margin-left:20px;">{message}</div>'
                f'</div>', unsafe_allow_html=True)


def bot_message(message):
    st.markdown(f'<div class="bot-message" style="display: flex; padding: 5px;">'
                f'<div style="background-color: #074c85; color: white; padding: 10px; border-radius: 10px; font-size:18px; margin-bottom:10px; margin-right:20px;">{message}</div>'
                f'</div>', unsafe_allow_html=True)


def main(i):
    st.title("Cosmos AI")
    embedding_function = OpenAIEmbeddings()

    new_db0 = FAISS.load_local("Faiss_Audio_PDF_TXT_WEB", embedding_function)
    if os.path.exists("NewFiles"):
        new_db1 = FAISS.load_local("NewFiles", embedding_function)
        new_db0.merge_from(new_db1)
    retriever = new_db0.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    # Initialize chat history using session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input field for user to enter a message
    user_input = st.chat_input("Your Message:")
    # Button to send the user's message
    if user_input:
        # Display previous chat messages
        for message, is_bot_response in st.session_state.chat_history:
            if is_bot_response:
                bot_message(message)
            else:
                user_message(message)
        # Add the user's message to the chat history
        st.session_state.chat_history.append((user_input, False))

        # Display the user's message
        user_message(user_input)

        # Bot's static response (you can replace this with a dynamic response generator)

        bot_response = rag_chain.invoke(user_input)
        response = client.images.generate(
            model="dall-e-3",
            prompt="generatea an image for "+ bot_response,
            size="1024x1024",
            quality="hd",
            n=1,
        )

        image_url = response.data[0].url
        print(image_url)

        # Add the bot's response to the chat history
        st.session_state.chat_history.append((bot_response, True))
        image= st.image(image_url)

        # Display the bot's response
        bot_message(image)


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


# Run the app
if __name__ == "__main__":
    main(0)