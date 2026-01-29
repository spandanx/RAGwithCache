import logging

from pymongo import MongoClient

class ChatHistoryDB:
    def __init__(self, username, password, hostname, database, keyspace, port):
        self.database = self.get_database(username, password, hostname, database, keyspace, port)

    def get_database(self, username, password, hostname, database, keyspace, port):

        CONNECTION_STRING = f"mongodb://{username}:{password}@{hostname}:{port}"
        client = MongoClient(CONNECTION_STRING)

        return client[database][keyspace]

    def get_record(self, session_id, username):
        # item_details = self.database.find_one({"key": key})
        # logging.info("get_record()")
        chat_history = self.database.find({"session_id":session_id,"username":username}).sort("timestamp", 1)
        # logging.info(chat_history)
        return chat_history

    def insert_record(self, data):
        response = self.database.insert_one(data)
        return response