import mysql.connector

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatSessionMySQL:

    def __init__(self, host, port, username, password, database):
        self.cnx = mysql.connector.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            database=database
        )

        self.cur = self.cnx.cursor()
        logging.info("MYSQL connection established.")

    def reestablish_connection(self):
        print("Called MysqlDB.start_connection()")
        self.cnx.reconnect()

    def enrich_user_result(self, columns, result_array):
        result = []
        for row in result_array:
            row_dict = dict()
            for i in range(len(columns)):
                # if type(row[i]) is datetime:
                #     row_dict[columns[i]] = row[i].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                # else:
                row_dict[columns[i]] = row[i]
            result.append(row_dict)
        return result

    def get_user_by_username(self, username):
        print("Calling MysqlDB.get_user_by_username()")
        if self.cnx.is_connected():
            print("MySQL Connection is active")
        else:
            # self.start_connection()
            self.reestablish_connection()
            print("MySQL Connection is not active")
        # self.cur.execute("SELECT username, session_id, session_description, DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s') AS created_at FROM chat_session where username = %s", (username, ))
        self.cur.execute(
            "SELECT username, session_id, session_description, created_at FROM chat_session where username = %s",
            (username,))
        desc = self.cur.description
        columns = [col[0] for col in desc]
        row = self.cur.fetchall()
        result = self.enrich_user_result(columns=columns, result_array=row)
        return result

    def insert_new_session(self, session_id, username, description, timestamp):
        print("Calling MysqlDB.update_stock_token()")
        print(session_id, username)
        if self.cnx.is_connected():
            print("MySQL Connection is active")
        else:
            self.reestablish_connection()
            print("MySQL Connection is not active")
        sql_insert_query = "INSERT INTO chat_session (session_id, username, session_description, created_at) values (%s, %s, %s, %s)"
        self.cur.execute(sql_insert_query, (session_id, username, description, timestamp))
        self.cnx.commit()

    def close_connection(self):
        print("Called MysqlDB.close_connection()")
        self.cnx.close()


if __name__ == "__main__":

    ###------------- Config Parser
    from configparser import ConfigParser

    parser = ConfigParser()
    config_file_path = '../../../config.properties'

    with open(config_file_path) as f:
        file_content = f.read()

    parser.read_string(file_content)
    ###------------- Config Parser

    chatSessionMySQL = ChatSessionMySQL(host = parser["MYSQL"]["mysql_hostname"],
                                        port = parser["MYSQL"]["mysql_port"],
                                        username = parser["MYSQL"]["mysql_username"],
                                        password = parser["MYSQL"]["mysql_password"],
                                        database = parser["MYSQL"]["mysql_database"]
                                        )
    import uuid
    from datetime import datetime
    # response = chatSessionMySQL.insert_new_session(session_id=str(uuid.uuid4()),
    #                                     username = 'user11',
    #                                     description = '',
    #                                     timestamp = datetime.now()
    #                                     )
    response = chatSessionMySQL.get_user_by_username('user11')
    print(response)

