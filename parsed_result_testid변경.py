import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import types

# 1. CSV 파일 불러오기 (순서 보존)
# df = pd.read_csv("parsed_results_final.csv", dtype=str, keep_default_na=False)
#
# # 2. 컬럼명을 소문자로 통일 (예외 방지)
# df.columns = df.columns.str.lower()
#
# # 3. test_id 뒤에 붙은 -숫자 제거 (예: TI268636-1 → TI268636)
# df["test_id"] = df["test_id"].str.replace(r"-\d+$", "", regex=True)
#
# # 4. 결과 저장
# df.to_csv("parsed_results_final_cleaned.csv", index=False, encoding="utf-8-sig")
# print("✅ TEST_ID에서 균 번호 제거 완료 → 'parsed_results_final_cleaned.csv'로 저장됨")


## DB insert
# 1. DB 연결
username = 'TEAM04'
password = 'oracle_4U'
host = '138.2.63.245'
port = 1521
service_name = 'srvinv.sub03250142080.kdtvcn.oraclevcn.com'
oracle_url = f"oracle+cx_oracle://{username}:{password}@{host}:{port}/?service_name={service_name}"
engine = create_engine(oracle_url)


# 2. CSV 불러오기
parsed_df = pd.read_csv("parsed_results_final_cleaned.csv", dtype=str)

# 3. 컬럼 소문자 통일
parsed_df.columns = parsed_df.columns.str.lower()

# 4. DB 기준 테이블 불러오기
query = "SELECT test_id, visit_no, patient_no FROM MICROBIOLOGY_RESULTS_TEST"
ref_df = pd.read_sql(query, con=engine)

# 5. 컬럼 소문자 통일
ref_df.columns = ref_df.columns.str.lower()

# 6. 병합 (test_id 기준)
merged_df = pd.merge(parsed_df, ref_df, on="test_id", how="left")

# 7. INSERT할 컬럼 정의
# 필요한 컬럼만 선택 (예: test_id, org_name, abx_name, mic_value 등 + visit_no, patient_no 등)
final_df = merged_df[[
    "test_id", "균명", "균수치", "항생제명", "mic결과", "감수성결과",
    "결과보고시간", "visit_no", "patient_no"
]]

# 8. INSERT 실행
final_df.to_sql(
    "PARSED_RESULTS_MERGED",  # 👉 원하는 테이블명
    con=engine,
    if_exists="replace",      # 이미 있으면 덮어쓰기 ("append" 하면 추가)
    index=False,
    dtype={
        'test_id': types.VARCHAR(20),
        '균명': types.VARCHAR(100),
        '균수치': types.VARCHAR(50),
        '항생제명': types.VARCHAR(100),
        'mic결과': types.VARCHAR(50),
        '감수성결과': types.VARCHAR(10),
        '결과보고시간': types.VARCHAR(50),
        'visit_no': types.VARCHAR(30),
        'patient_no': types.VARCHAR(20)
    }
)

print("✅ INSERT 완료 → 테이블명: PARSED_RESULTS_MERGED")