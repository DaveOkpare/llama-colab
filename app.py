import os

import streamlit as st
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from completion import get_completion

load_dotenv()

os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HUGGING_FACE_HUB_TOKEN")

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

st.title("Chat with Llama")

with st.spinner("Initializing the llama..."):
    if "model" not in st.session_state or "tokenizer" not in st.session_state:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            load_in_4bit=True,
            rope_scaling={"type": "dynamic", "factor": 2.0},
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

        st.session_state["model"] = model
        st.session_state["tokenizer"] = tokenizer

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_completion(prompt=prompt, tokenizer=tokenizer, model=model)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
