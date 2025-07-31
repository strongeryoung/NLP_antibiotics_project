import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.oracle.db_connect import connect_to_db

def main():
    conn = connect_to_db(db_user='EDU116', db_name='KDT')

    # DB 이름 출력 (예시 쿼리)
    cursor = conn.cursor()
    cursor.execute("SELECT SYS_CONTEXT('USERENV', 'DB_NAME') FROM dual")
    print("DB 1:", cursor.fetchone()[0])
    cursor.close()

    conn.close()

if __name__ == "__main__":
    main()