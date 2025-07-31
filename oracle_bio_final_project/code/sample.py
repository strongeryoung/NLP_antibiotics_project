import oracledb

username = "TEAM04"
password = "oracle_4U"
dsn = "138.2.63.245:1521/srvinv.sub03250142080.kdtvcn.oraclevcn.com"

try:
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    print("✅ Oracle DB 연결 성공!")
    conn.close()
except Exception as e:
    print("❌ 연결 실패:", e)