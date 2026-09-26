import os
from dotenv import load_dotenv
import streamlit as st
from langchain_mistralai import ChatMistralAI

load_dotenv()

st.set_page_config(page_title="CineSage Chat", page_icon="🎬")
st.title("🎬 CineSage Assistant")

api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.warning("Please add your MISTRAL_API_KEY in the .env file before running the app.")
    st.stop()

model = ChatMistralAI(
    model="mistral-large-latest",
    api_key=api_key,
    temperature=0.2,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about movies, stories, or coding."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Type your message here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})