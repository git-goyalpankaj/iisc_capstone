import mysql.connector
import time

from config.settings import DB_CONFIG

def connect_to_mysql(attempts=3, delay=2):
    attempt = 1
    # Implement a reconnection routine
    while attempt < attempts + 1:
        try:
            return mysql.connector.connect(**DB_CONFIG)
        except (mysql.connector.Error, IOError) as err:
            if attempts is attempt:
                # Attempts to reconnect failed; returning None
                print("Failed to connect, exiting without a connection: %s", err)
                return None
            print("Connection failed: %s. Retrying (%d/%d)...",
                err,
                attempt,
                attempts-1,
            )
            # progressive reconnect delay
            time.sleep(delay ** attempt)
            attempt += 1
    return None

def is_db_connected(cnx):

    if cnx and cnx.is_connected():
        with cnx.cursor() as cursor:
            result = cursor.execute("SELECT * FROM user_data LIMIT 5")
            rows = cursor.fetchall()
            if len(rows) > 0:
                return True
    else:
        return False

def select_record(query):
    cnx = connect_to_mysql()

    with cnx.cursor(buffered=True) as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        if not rows:
            print("No records found.")
        else:
            for row in rows:
                print(row)

        return rows

def delete_record(query):
    cnx = connect_to_mysql()

    if cnx is None:
        print("No database connection.")
        return False

    try:
        with cnx.cursor() as cursor:
            cursor.execute(query)
            cnx.commit()
            return True
    except Exception as e:
        print("Failed to delete record:", e)
        return False
    finally:
        cnx.close()