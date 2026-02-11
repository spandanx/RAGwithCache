import logging
import asyncio
import streamlit as st
from streamlit_extras.mention import mention
from streamlit_extras.stylable_container import stylable_container
# from streamlit_extras.bottom_container import bottom
# from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

from main import RAGApplication, ChatHistoryHandler, ChatSessionListHandler
from datetime import datetime
import re

rag_app = RAGApplication()
if rag_app.rag_chain is None:
    rag_app.load_store()
chatHistoryHandler = ChatHistoryHandler()
chatSessionListHandler = ChatSessionListHandler()

# st.session_state['user_id'] = "user11"

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
    logging.info("Loading sessions")
    for session in chat_sessions:
        logging.info(session)
        st.session_state['chat_threads'].append(session["session_id"])

# -------------------- Thread management end --------------------

# -------------------- Session config start --------------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

################ Load conversations ##################
def load_conversation(username, session_id):
    logging.info("Called load_conversation()")
    messages = chatHistoryHandler.retrive_chat(username=username, session_id=session_id, record_limit = 5)
    # for msg in messages:
    #     logging.info(msg)
    # state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    # return state.values.get('messages', [])
    return messages

def format_chat_history():
    logging.info("format_chat_history()")
    chat_history = ""
    for msg in st.session_state['message_history']:
        if msg["role"] == "user":
            chat_history += "User: " + msg["content"] + "\n"
        else:
            if isinstance(msg["content"], dict) and "answer" in msg["content"]:
                chat_history += "Assistant: " + msg["content"]["answer"] + "\n"
            else:
                chat_history += "Assistant: " + msg["content"] + "\n"
    return chat_history

def show_ai_chat(ai_message):
    logging.info("Called show_ai_chat()")
    logging.info(ai_message)
    with st.chat_message("assistant", avatar="🤖"):
        # if isinstance(ai_message, dict):
        if "content" in ai_message:
            if "answer" in ai_message["content"]:
                st.write(ai_message["content"]["answer"])

            if "source" in ai_message["content"] and len(ai_message["content"]["source"])>0:
                # st.markdown("**Sources:**")
                st.write("**Sources:**")
                # for i, (source_type, index) in enumerate(citations, 1):
                for source_url in ai_message["content"]["source"]:
                    # Replace with actual URLs from your data sources (e.g., session_state or msg["sources"])
                    # source_label = f"{source_type.upper()}{index}"
                    # logging.info("source_url: " + source_url)
                    # logging.info("source_label: " + source_label)
                    mention(label=source_url, icon="🔗", url=source_url)
        else:
            st.write(ai_message)

def login_page():
    # Define a placeholder for the login form
    login_placeholder = st.empty()

    # Check if the user is already logged in
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        with login_placeholder.form("login_form"):
            st.markdown("### Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                # response = asyncio.run(chatSessionListHandler.authenticate(username=username, password=password))
                response = chatSessionListHandler.authenticate(username=username, password=password)
                logging.info("Response - ")
                logging.info(response)

                if response is not None and response is not False:
                    logging.info("Logging In")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.token = response
                    login_placeholder.empty()  # Clear the login form
                    # st.success("Login successful!")
                    logging.info("Setting cache done")
                    st.toast("Login successful!", icon="✅")
                    st.rerun()
                else:
                    logging.info("Failed to login")
                    st.error("Login failed. Please check your username and password.")
                    # st.toast("Login failed. Incorrect username or password.", icon="❌")

# async def stream_response():
#     logging.info("Starting stream_response()")
#     selected_key = ""
#     current_key = ""
#     key_already_selected = False
#
#     val = ""
#     pattern = r'[^a-zA-Z0-9\n\s"\']'
#     quotation_crawled = 0
#     async for chunk in rag_app.answer_question(user_input, ""):
#         # Apply custom transformations
#         # processed = chunk.strip().upper()
#         cleaned_chunk = re.sub(r'[\n\r]+', '', re.sub(r'[^a-zA-Z0-9,\']', '', chunk))
#         # logging.info(chunk + " -> regex -> " + re.sub(r'[\n\r]+', ' ', re.sub(r'[^a-zA-Z0-9"\']', '', chunk)) + "|")
#         quotation_crawled += chunk.count('"')
#         if quotation_crawled % 4 == 1:
#             current_key += cleaned_chunk
#         elif quotation_crawled % 4 == 2:
#             key_already_selected = True
#             selected_key = current_key
#             current_key = ""
#         elif quotation_crawled % 4 == 3:
#             # if val == "":
#             val += cleaned_chunk
#             # logging.info(
#             #     chunk + " -> cleaned -> " + cleaned_chunk + " current_key -> " + current_key + " -> " + selected_key + " -> quotation_crawled -> " + str(quotation_crawled))
#             yield cleaned_chunk
#         elif quotation_crawled % 4 == 4:
#             selected_key = ""
#             current_key = ""
#             # else:
#             #     val += chunk
#             #     yield chunk
#         logging.info(
#             chunk + " -> cleaned -> " + cleaned_chunk + " current_key -> " + current_key + " -> " + selected_key + " -> quotation_crawled -> " + str(quotation_crawled))
# st.markdown("""
# <style>
# .fixed-bottom {
#     position: fixed;
#     bottom: 0;
#     left: 0;
#     width: 100%;
#     background-color: white; /* Match your app background */
#     padding: 10px;
#     border-top: 1px solid #ddd; /* Optional: adds a line above the button */
#     display: flex;
#     justify-content: center; /* Center the button horizontally */
#     align-items: center;
#     z-index: 9999;
# }
# </style>
# """, unsafe_allow_html=True)

# dummy_tab_1, dummy_tab_2, logout_tab = st.columns([1, 1, 0.2])

# with logout_tab:
#     if ('logged_in' in st.session_state) and st.session_state.logged_in:
#         if st.button("Logout", key="logout_button_small"):
#             st.session_state.logged_in = False
#             st.session_state.username = ""
#             # st.experimental_rerun() # Rerun to show login page again
#             st.rerun()

if ('logged_in' not in st.session_state) or (('logged_in' in st.session_state) and (not st.session_state.logged_in)):
    login_page()

if ('logged_in' in st.session_state) and st.session_state.logged_in:
    # add_thread(st.session_state['thread_id'])
    load_all_sessions(st.session_state['username'])
    # -------------------- Session config end --------------------

    with st.sidebar:
        # st.sidebar.title('RAG Chatbot')
        # st.sidebar.header('My Sessions')
        #
        # if st.sidebar.button('New Chat'):
        #     new_chat()
        st.title('RAG Chatbot')
        st.header('My Sessions')

        if st.button('New Chat'):
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

        for thread_id in st.session_state['chat_threads'][::-1]:
            if st.button(str(thread_id), key=thread_id):
                st.session_state['thread_id'] = thread_id
                logging.info("session_id is set to", str(thread_id))

                messages = load_conversation(username=st.session_state['username'], session_id=st.session_state['thread_id'])

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

        if ('logged_in' in st.session_state) and st.session_state.logged_in:
            with stylable_container(
                    "green",
                    css_styles="""
                button {
                    background-color: #cc6c6c;
                    color: black;
                }""",
            ):
                # button1_clicked = st.button("Logout", key="button1")
                if st.button("Logout", key="button1"):
                    st.session_state.logged_in = False
                    st.session_state.username = ""
                    # st.experimental_rerun() # Rerun to show login page again
                    st.rerun()


    ################ Load conversations ##################
    logging.info("Loading conversations, .............")
    for msg in st.session_state['message_history']:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(msg["content"])
        else:
            # with st.chat_message("assistant", avatar="🤖"):
            #     st.write(msg["content"])
            show_ai_chat(msg)

    user_input = st.chat_input('Type here')

    if user_input:

        st.session_state['message_history'].append({'role': 'user', 'content': user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(user_input)

        ## ------- Insert user message ---------
        now = datetime.now()
        time_string = now.strftime("%Y-%m-%dT%H:%M:%S")
        chatHistoryHandler.insert_chat_record(message = user_input,
                                              username = st.session_state['username'],
                                              session_id = str(st.session_state['thread_id']),
                                              timestamp = time_string,
                                              role = "user")
        chat_history = format_chat_history() # Modify this
        ai_message = rag_app.answer_question(user_input, chat_history)

        # first add the message to message_history
        # st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
        # with st.chat_message("assistant", avatar="🤖"):
        #     st.write(ai_message)
        show_ai_chat({"content": ai_message})
        # full_response = st.write_stream(rag_app.answer_question(user_input, chat_history))
        # placeholder = st.empty()
        #--------------------
        # with st.chat_message("user", avatar="🧑‍💻"):
        #     st.write_stream(stream_response())
        # full_response = st.write_stream(rag_app.answer_question(user_input, ""))
        # logging.info("Printing the full response")
        # for message in full_response:
        #     logging.info(message)
        # --------------------
        # full_response.clear()
        # if st.button("Clear Output"):
        #     placeholder.empty()
        # st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
        ## ------- Insert assistant message ---------
        now = datetime.now()
        time_string = now.strftime("%Y-%m-%dT%H:%M:%S")
        chatHistoryHandler.insert_chat_record(message=ai_message,
                                              username=st.session_state['username'],
                                              session_id=str(st.session_state['thread_id']),
                                              timestamp=time_string,
                                              role="assistant")