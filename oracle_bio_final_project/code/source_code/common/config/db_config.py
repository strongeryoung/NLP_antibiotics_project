# /source_code/common/config/db_config.py

import os
from dotenv import load_dotenv

load_dotenv()

INSTANT_CLIENT_DIR = os.getenv("ORACLE_INSTANT_CLIENT_DIR")

def get_db_config(db_user=None, db_name=None):
    user = os.getenv(f"ORACLE_{db_user}_USER")
    password = os.getenv(f"ORACLE_{db_user}_PASSWORD")
    host = os.getenv(f"ORACLE_{db_name}_HOST")
    port = os.getenv(f"ORACLE_{db_name}_PORT")
    service_name = os.getenv(f"ORACLE_{db_name}_SERVICE_NAME")

    if not all([user, password, host, port, service_name]):
        raise ValueError("A required environment variable is missing. Verify your .env file configuration.")

    return {
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "service_name": service_name
    }

def get_dsn(config):
    return f"{config['host']}:{config['port']}/{config['service_name']}"