import pymysql
import time
from config import DB_CONFIG

def connect_to_mysql(attempts=3, delay=2):
    for attempt in range(1, attempts + 1):
        try:
            connection = pymysql.connect(**DB_CONFIG)
            print("Connected successfully on attempt", attempt)
            return connection
        except pymysql.MySQLError as err:
            if attempt == attempts:
                print(f"Failed to connect after {attempts} attempts. Error: {err}")
                return None
            print(f"Connection failed: {err}. Retrying ({attempt}/{attempts})...")
            time.sleep(delay ** attempt)  # Exponential backoff
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
    if cnx is None:
        print("Database connection could not be established.")
        return []

    try:
        with cnx.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            if not rows:
                print("No records found.")
            else:
                for row in rows:
                    print(row)
            return rows
    except Exception as e:
        print(f"An error occurred while executing the query: {e}")
        return []
    finally:
        cnx.close()


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