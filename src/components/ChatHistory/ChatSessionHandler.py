import logging
import datetime
from src.security.Auth import authenticate_user, create_access_token
from src.components.ChatHistory.ChatSessionDB import MySQLDB
from src.components.ChatHistory.ChatHistoryDB import ChatHistoryDB

###------------- Config Parser
from configparser import ConfigParser

parser = ConfigParser()
config_file_path = 'config.properties'

with open(config_file_path) as f:
    file_content = f.read()

parser.read_string(file_content)
###------------- Config Parser


class ChatSessionListHandler:
    def __init__(self):
        self.chatSessionMySQL = MySQLDB(host = parser["MYSQL"]["mysql_hostname"],
                                        port = parser["MYSQL"]["mysql_port"],
                                        username = parser["MYSQL"]["mysql_username"],
                                        password = parser["MYSQL"]["mysql_password"],
                                        database = parser["MYSQL"]["mysql_database"]
                                        )

    def retrive_chat(self, username):
        results = self.chatSessionMySQL.get_chat_sessions_by_username(username=username)
        return results

    def insert_chat_record(self, session_id, username, description, timestamp):
        self.chatSessionMySQL.insert_new_session(
            session_id = session_id,
            username = username,
            description = description,
            timestamp = timestamp
        )

    '''
    Authenticate user
    '''
    def authenticate(self, username, password):
        response = authenticate_user(username, password, self.chatSessionMySQL)
        if response:
            logging.info("main - authenticate()")
            logging.info(response)
            data = {"sub": username}
            token = create_access_token(data, parser['ENCRYPTION']['SECRET_KEY'], parser['ENCRYPTION']['ALGORITHM'],
                                        int(parser['ENCRYPTION']['ACCESS_TOKEN_EXPIRE_MINUTES']))
            return token
        return response

    def update_stock_token_controller(self, token, username):
        logging.info("Calling update_stock_token_controller()")
        logging.info(token)
        current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        self.chatSessionMySQL.update_stock_token(token, current_time, username)
        return {"message": "Successfully Updated the stock token"}

    def get_stock_token_controller(self, username):
        return self.chatSessionMySQL.get_basic_user_info(username)

class ChatHistoryHandler:
    def __init__(self):
        self.chatHistoryMongoDB = ChatHistoryDB(username = parser['MONGODB']['mongodb_username'],
                                          password = parser['MONGODB']['mongodb_password'],
                                          hostname = parser['MONGODB']['mongodb_hostname'],
                                          database = parser['MONGODB']['mongodb_database'],
                                          keyspace = parser['MONGODB']['mongodb_keyspace'],
                                          port = parser['MONGODB']['mongodb_port']
                                          )

    def retrive_chat(self, username, session_id, record_limit):
        results = self.chatHistoryMongoDB.get_record(username=username, session_id=session_id, record_limit=record_limit)
        return results

    def insert_chat_record(self, message, username, session_id, timestamp, role):
        data = {
            "key": username + "_" + session_id + "_" + timestamp,
            "data": message,
            "username": username,
            "session_id": session_id,
            "timestamp": timestamp,
            "role": role
        }
        self.chatHistoryMongoDB.insert_record(data)