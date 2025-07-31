# /source_code/common/oracle/db_connect.py

import oracledb
from common.config.db_config import get_db_config, get_dsn, INSTANT_CLIENT_DIR

def connect_to_db(db_user=None, db_name=None):
    # Thick 모드 초기화
    oracledb.init_oracle_client(lib_dir=INSTANT_CLIENT_DIR)

    # DB 접속 정보 가져오기
    config = get_db_config(db_user, db_name)
    dsn = get_dsn(config)

    connection = oracledb.connect(
        user=config["user"],
        password=config["password"],
        dsn=dsn
    )
    print(f"✅ Successfully connected to DB (Thick mode)")
    return connection