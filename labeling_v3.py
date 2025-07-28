import pandas as pd
import json
import os
from tqdm import tqdm
import re

# 경로 설정
csv_path = "../BioBERT/parsed_results.csv"
output_jsonl_path = "doccano_ner_sample_TESTID_grouped_final.jsonl"
균코드_path = "/parsed/균_코드_완성.xlsx"
항생제코드_path = "/parsed/항생제_코드_완성.xlsx"

# 데이터 불러오기
df = pd.read_csv(csv_path)
균_df = pd.read_excel(균코드_path)
항생제_df = pd.read_excel(항생제코드_path)

df["항생제명"] = df["항생제명"].astype(str).str.strip()
df["균명"] = df["균명"].astype(str).str.strip()
df["균수치"] = df["균수치"].astype(str).str.strip()

grouped = df.groupby("TEST_ID")

def match_result_pattern(text, target):
    target = re.escape(target)
    patterns = [
        rf"\(\s*{target}\s*\)",
        rf"\[\s*{target}\s*\]",
        rf"[^\w]{target}[^\w]",
        rf"^{target}[^\w]",
        rf"[^\w]{target}$",
        rf"\b{target}\b"
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.start(), m.end()
    return None, None

def make_labels(text, group):
    labels = []
    text_lower = text.lower()

    row0 = group.iloc[0]
    균명 = str(row0["균명"]).strip()
    균수치 = str(row0["균수치"]).strip()

    if 균명 and 균명 in text:
        start = text.find(균명)
        end = start + len(균명)
        labels.append([start, end, "균명"])
    if 균수치 and 균수치 in text:
        start = text.find(균수치)
        end = start + len(균수치)
        labels.append([start, end, "균수치"])

    for _, row in group.iterrows():
        항 = str(row.get("항생제명", "")).strip()
        mic = str(row.get("MIC결과", "")).strip()
        res = str(row.get("감수성결과", "")).strip().upper()

        if not (항 and mic and res):
            continue

        if 항 in text:
            s = text.find(항)
            labels.append([s, s + len(항), "항생제명"])

        if mic in text:
            s = text.find(mic)
            labels.append([s, s + len(mic), "MIC결과"])

        s, e = match_result_pattern(text, res)
        if s is not None:
            labels.append([s, e, "감수성결과"])

    return labels

results = []
for test_id, group in tqdm(grouped, desc="TEST_ID별 라벨링"):
    text = str(group.iloc[0]["원본문"])
    labels = make_labels(text, group)
    results.append({"text": text, "labels": labels})

with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 저장 완료: {output_jsonl_path}")
