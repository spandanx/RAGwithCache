import logging

import streamlit as st
# from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

from main import RAGApplication, ChatHistoryHandler
from datetime import datetime

rag_app = RAGApplication()
chatHistoryHandler = ChatHistoryHandler()

st.session_state['user_id'] = "user11"

# -------------------- Thread management start --------------------
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def new_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

# def load_conversation(thread_id):
#     state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
#     # Check if messages key exists in state values, return empty list if not
#     return state.values.get('messages', [])

# -------------------- Thread management end --------------------

# -------------------- Session config start --------------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# add_thread(st.session_state['thread_id'])

# -------------------- Session config end --------------------

# -------------------- Chat History --------------------
# st.session_state -> dict ->
# CONFIG = {'configurable': {'thread_id': 'thread-1'}}
#
# if 'message_history' not in st.session_state:
#     st.session_state['message_history'] = []
#
# # loading the conversation history
# for message in st.session_state['message_history']:
#     with st.chat_message(message['role']):
#         st.text(message['content'])

# -------------------- Chat History --------------------

st.sidebar.title('RAG Chatbot')
st.sidebar.header('My Sessions')

if st.sidebar.button('New Chat'):
    new_chat()

# for thread_id in st.session_state['chat_threads'][::-1]:
#     if st.sidebar.button(str(thread_id)):
#         st.session_state['thread_id'] = thread_id

################ Load conversations ##################
def load_conversation(thread_id):
    # state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    # return state.values.get('messages', [])
    return []


for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        logging.info("session_id is set to", thread_id)
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages

################ Load conversations ##################

user_input = st.chat_input('Type here')

if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    ## ------- Insert user message ---------
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%dT%H:%M:%S")
    chatHistoryHandler.insert_chat_record(message = user_input,
                                          username = st.session_state['user_id'],
                                          session_id = str(st.session_state['thread_id']),
                                          timestamp = time_string,
                                          role = "user")
    ai_message = rag_app.answer_question(user_input)

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)
    ## ------- Insert assistant message ---------
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%dT%H:%M:%S")
    chatHistoryHandler.insert_chat_record(message=ai_message,
                                          username=st.session_state['user_id'],
                                          session_id=str(st.session_state['thread_id']),
                                          timestamp=time_string,
                                          role="assistant")