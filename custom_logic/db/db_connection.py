import mysql.connector
from custom_logic.db.credentials import ConnectionConstants


def get_db_connection():
    return MySQLConnection(
        ConnectionConstants.HOST,
        ConnectionConstants.USERNAME,
        ConnectionConstants.PASSWORD,
        ConnectionConstants.DATABASE)

class MySQLConnection:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def __enter__(self):
        # Called when entering the 'with' block
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )

            # if self.connection.is_connected():
            #     print(f"Connected to MySQL server (Host: {self.host}, Database: {self.database})")

            return self.connection

        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None

    def __exit__(self, exc_type, exc_value, traceback):
        if traceback is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
