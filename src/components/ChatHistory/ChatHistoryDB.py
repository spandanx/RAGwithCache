from pymongo import MongoClient

class ChatHistoryDB:
    def __init__(self, username, password, hostname, database, keyspace, port):
        self.database = self.get_database(username, password, hostname, database, keyspace, port)

    def get_database(self, username, password, hostname, database, keyspace, port):

        CONNECTION_STRING = f"mongodb://{username}:{password}@{hostname}:{port}"
        client = MongoClient(CONNECTION_STRING)

        return client[database][keyspace]

    def get_record(self, key):
        item_details = self.database.find_one({"key": key})
        return item_details

    def insert_record(self, data):
        response = self.database.insert_one(data)
        return response