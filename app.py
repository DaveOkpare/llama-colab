import streamlit as st

from inference import init_checkpoints, get_completion

st.title("Chat with Llama")

if 'model' not in st.session_state or 'tokenizer' not in st.session_state :
    with st.spinner("Initializing the llama..."):
        st.session_state['model'], st.session_state['tokenizer'] = init_checkpoints()

model, tokenizer = st.session_state['model'], st.session_state['tokenizer']

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
