import logging
import asyncio
import streamlit as st
# from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

from main import RAGApplication, ChatHistoryHandler, ChatSessionListHandler
from datetime import datetime

rag_app = RAGApplication()
chatHistoryHandler = ChatHistoryHandler()
chatSessionListHandler = ChatSessionListHandler()

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
    chatSessionListHandler.insert_chat_record(
        session_id=str(thread_id),
        username = st.session_state['user_id'],
        description = '',
        timestamp = datetime.now()
    )
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

# def load_conversation(thread_id):
#     state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
#     # Check if messages key exists in state values, return empty list if not
#     return state.values.get('messages', [])

def load_all_sessions(username):
    st.session_state['chat_threads'] = []
    chat_sessions = chatSessionListHandler.retrive_chat(username)
    for session in chat_sessions:
        st.session_state['chat_threads'].append(session["session_id"])

# -------------------- Thread management end --------------------

# -------------------- Session config start --------------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# add_thread(st.session_state['thread_id'])
load_all_sessions(st.session_state['user_id'])
# -------------------- Session config end --------------------

st.sidebar.title('RAG Chatbot')
st.sidebar.header('My Sessions')

if st.sidebar.button('New Chat'):
    new_chat()

# -------------------- Chat History --------------------
# st.session_state -> dict ->
# CONFIG = {'configurable': {'thread_id': 'thread-1'}}
#
# if 'message_history' not in st.session_state:
#     st.session_state['message_history'] = []

# loading the conversation history
# for message in st.session_state['message_history']:
#     with st.chat_message(message['role']):
#         st.text(message['content'])

# -------------------- Chat History --------------------

################ Load conversations ##################
def load_conversation(username, session_id):
    logging.info("Called load_conversation()")
    messages = chatHistoryHandler.retrive_chat(username=username, session_id=session_id)
    # for msg in messages:
    #     logging.info(msg)
    # state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    # return state.values.get('messages', [])
    return messages


for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id), key=thread_id):
        st.session_state['thread_id'] = thread_id
        logging.info("session_id is set to", str(thread_id))

        messages = load_conversation(username=st.session_state['user_id'], session_id=st.session_state['thread_id'])

        temp_messages = []
        st.session_state['message_history'] = []
        logging.info("Iterating over messages")
        for msg in messages:
            logging.info("MESSAGE")
            logging.info(msg)
            if isinstance(msg, HumanMessage):
                role='user'
                logging.info("MESSAGE - user")
            else:
                role='assistant'
                logging.info("MESSAGE - assistant")
            temp_messages.append({'role': msg["role"], 'content': msg["data"]})
            # st.session_state['message_history'].append({'role': 'user', 'content': msg["data"]})

        st.session_state['message_history'] = temp_messages

################ Load conversations ##################
logging.info("Loading conversations, .............")
for msg in st.session_state['message_history']:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg["content"])

user_input = st.chat_input('Type here')

if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_input)

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
    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_message)
    ## ------- Insert assistant message ---------
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%dT%H:%M:%S")
    chatHistoryHandler.insert_chat_record(message=ai_message,
                                          username=st.session_state['user_id'],
                                          session_id=str(st.session_state['thread_id']),
                                          timestamp=time_string,
                                          role="assistant")