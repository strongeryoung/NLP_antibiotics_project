import pandas as pd
from sqlalchemy import create_engine, text

# 1. Oracle DB 연결
username = 'TEAM04'
password = 'oracle_4U'
host = '138.2.63.245'
port = 1521
service_name = 'srvinv.sub03250142080.kdtvcn.oraclevcn.com'
oracle_url = f"oracle+cx_oracle://{username}:{password}@{host}:{port}/?service_name={service_name}"
engine = create_engine(oracle_url)

# 2. PARSED_RESULTS_MERGED에서 데이터 로드
query = """
SELECT 
    test_id,
    "균명",
    "균수치",
    "항생제명",
    "mic결과",
    "감수성결과",
    "결과보고시간",
    visit_no,
    patient_no
FROM PARSED_RESULTS_MERGED
ORDER BY test_id
"""
df = pd.read_sql(query, con=engine)
df.columns = [col.lower() for col in df.columns]
print(f"✅ 불러온 데이터 수: {len(df)}")

# 3. 고유한 unique_id 생성 (prefix + 순번 방식)
df['prefix'] = 'UI_' + df['visit_no'].astype(str).str[-3:] + '_' + df['patient_no'].astype(str).str[-3:]
df['row_num'] = df.groupby('prefix').cumcount().astype(str).str.zfill(3)
df['unique_id'] = df['prefix'] + '_' + df['row_num']

# 4. 임시 테이블용 DataFrame 생성
temp_df = df[['test_id', 'unique_id']].copy()

# 5. TEMP_UNIQUE_IDS 테이블 삭제 (존재 시)
with engine.begin() as conn:
    conn.execute(text("""
        BEGIN
            EXECUTE IMMEDIATE 'DROP TABLE TEMP_UNIQUE_IDS PURGE';
        EXCEPTION
            WHEN OTHERS THEN NULL;
        END;
    """))
    print("🔄 기존 TEMP_UNIQUE_IDS 테이블 삭제 (있으면)")

# 6. TEMP_UNIQUE_IDS 테이블 업로드
temp_df.to_sql('TEMP_UNIQUE_IDS', con=engine, index=False, if_exists='append')
print("✅ TEMP_UNIQUE_IDS 테이블 업로드 완료")

# 7. Oracle MERGE 실행 (test_id 기준)
with engine.begin() as conn:
    merge_sql = """
    MERGE INTO PARSED_RESULTS_MERGED tgt
    USING TEMP_UNIQUE_IDS src
    ON (tgt.test_id = src.test_id)
    WHEN MATCHED THEN
      UPDATE SET tgt.UNIQUE_ID = src.unique_id
    """
    conn.execute(text(merge_sql))
print("✅ Oracle MERGE UPDATE 완료!")

# 8. CSV 저장
df.drop(columns=['prefix', 'row_num'], inplace=True)
df.to_csv("parsed_results_with_unique_id.csv", index=False, encoding='utf-8-sig')
print("📁 CSV 저장 완료")

# 9. 중복 검사
dup = df['unique_id'].duplicated().sum()
print(f"🧪 중복된 UNIQUE_ID 수: {dup}")
if dup == 0:
    print("✅ UNIQUE_ID는 모든 행에서 고유합니다.")
