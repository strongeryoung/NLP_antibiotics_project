import json, os, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset
from collections import defaultdict
from transformers import Trainer
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score
import random



# 🔧 경로 설정
doccano_file = "/final_biobert/ner_training_data_biobert.jsonl"
base_dir = "C:/Users/KDT_35/PycharmProjects/Oracle_Final_Project/DB-BioBERT pipeline/ner_biobert_model_epoch_2_kfold_3_sampling"
ENTITY_TYPES = ["균명", "균수치", "항생제명", "MIC결과", "감수성결과"]

# ✅ 1. Doccano 데이터 로드
def get_bio_labels(text, labels):
    tags = ["O"] * len(text)
    for start, end, label in labels:
        tags[start] = f"B-{label}"
        for i in range(start + 1, end):
            tags[i] = f"I-{label}"
    return tags

examples = []
with open(doccano_file, "r", encoding="utf-8-sig") as f:
    for line in f:
        item = json.loads(line)
        text = item["text"]
        bio_labels = get_bio_labels(text, item["labels"])
        examples.append({"tokens": list(text), "ner_tags": bio_labels})

# 전체 데이터에서 20만개 무작위 샘플링
SAMPLE_SIZE = 200_000  # 원하는 샘플 수
random.seed(42)  # 재현성

if len(examples) > SAMPLE_SIZE:
    examples = random.sample(examples, SAMPLE_SIZE)
else:
    print(f"데이터가 {SAMPLE_SIZE}보다 적으므로 전체 {len(examples)}개를 그대로 사용합니다.")

# ✅ 2. train/val/test 분리 (90:10)
trainval_examples, test_examples = train_test_split(examples, test_size=0.1, random_state=42)

# ✅ 3. tokenizer 및 라벨 설정
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
label_list = ['O'] + [f"{p}-{l}" for l in ENTITY_TYPES for p in ['B', 'I']]
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

def tokenize_and_align_labels(example):
    tokenized_input = tokenizer(example["tokens"], is_split_into_words=True, truncation=True, max_length=512, padding='max_length')
    word_ids = tokenized_input.word_ids()
    aligned_labels = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != prev_word_idx:
            aligned_labels.append(label_to_id.get(example["ner_tags"][word_idx], 0))
        else:
            aligned_labels.append(-100)
        prev_word_idx = word_idx
    tokenized_input["labels"] = aligned_labels
    return tokenized_input

# ✅ 4. Dataset 생성 및 토큰화
dataset_trainval = Dataset.from_list(trainval_examples).map(tokenize_and_align_labels)
dataset_test = Dataset.from_list(test_examples).map(tokenize_and_align_labels)

# ✅ 5. 3-Fold 교차검증 학습
kf = KFold(n_splits=3, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_trainval)):
    print(f"\n===== 🌀 Fold {fold+1} / 3 시작 =====")
    train_dataset = dataset_trainval.select(train_idx)
    eval_dataset = dataset_trainval.select(val_idx)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id
    )

    fold_dir = os.path.join(base_dir, f"fold_{fold+1}")
    os.makedirs(fold_dir, exist_ok=True)

    # ✅ fold_dir 정의 이후 추가
    final_model_dir = os.path.join(fold_dir, "final_model")  # ✅ 수정됨: 하위에 final_model 폴더 생성
    os.makedirs(final_model_dir, exist_ok=True)

    def compute_metrics(p):
        predictions, labels = p
        preds = np.argmax(predictions, axis=2)

        true_labels = []
        true_preds = []

        for pred, label in zip(preds, labels):
            temp_pred = []
            temp_label = []
            for p_, l_ in zip(pred, label):
                if l_ != -100:
                    temp_pred.append(id_to_label[p_])
                    temp_label.append(id_to_label[l_])
            true_preds.append(temp_pred)
            true_labels.append(temp_label)

        return {
            "f1": f1_score(true_labels, true_preds),
            "accuracy": accuracy_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall": recall_score(true_labels, true_preds)
        }

    training_args = TrainingArguments(
        output_dir=fold_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=os.path.join(fold_dir, "logs"),
        logging_steps=10,
        logging_strategy="steps",  # ✅ 꼭 필요
        overwrite_output_dir=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        compute_metrics=compute_metrics
    )

    trainer.train()
    # ✅ 수정됨: 실제로 사용할 모델은 final_model 디렉토리에 저장
    trainer.model.save_pretrained(final_model_dir)  # ✅ best model 저장
    trainer.tokenizer.save_pretrained(final_model_dir)  # ✅ tokenizer 저장

    # fold 평가 (eval_dataset 기준)
    predictions, labels, _ = trainer.predict(eval_dataset)
    preds = np.argmax(predictions, axis=2)

    true_labels, true_preds = [], []
    for pred, label in zip(preds, labels):
        sent_true, sent_pred = [], []
        for p_, l_ in zip(pred, label):
            if l_ != -100:
                sent_true.append(id_to_label[l_])
                sent_pred.append(id_to_label[p_])
        true_labels.append(sent_true)
        true_preds.append(sent_pred)

    # 📊 엔티티별 성능 분석
    report = seq_classification_report(true_labels, true_preds, output_dict=True)
    entity_scores = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})

    def extract_entity(bio):
        return bio.split("-")[1] if "-" in bio else "O"

    for label, metrics in seq_classification_report(true_labels, true_preds, output_dict=True).items():
      if label in ["accuracy", "macro avg", "weighted avg"]:
        continue
      entity = label.split("-")[1] if "-" in label else "O"
      entity_scores[entity]["precision"].append(metrics["precision"])
      entity_scores[entity]["recall"].append(metrics["recall"])
      entity_scores[entity]["f1"].append(metrics["f1-score"])

    entity_avg_report = {
        ent: {
            "precision": np.mean(scores["precision"]),
            "recall": np.mean(scores["recall"]),
            "f1": np.mean(scores["f1"])
        }
        for ent, scores in entity_scores.items()
    }

    # 저장
    with open(os.path.join(fold_dir, "entity_wise_report.json"), "w", encoding="utf-8") as f:
        json.dump(entity_avg_report, f, ensure_ascii=False, indent=2)


# ✅ 6. Fold 3의 Best Model 불러와서 test set 평가
print("\n🧪 Fold 3의 Best Model로 test set 평가 중...")

# fold_3 디렉토리에서 best model 불러오기
best_model_dir = os.path.join(base_dir, "fold_3", "final_model")
model = AutoModelForTokenClassification.from_pretrained(best_model_dir)
tokenizer = AutoTokenizer.from_pretrained(best_model_dir)

# 새 Trainer 생성 (평가만 수행)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

# 예측
predictions, labels, _ = trainer.predict(dataset_test)
preds = np.argmax(predictions, axis=2)

# 라벨 복원
# 🤖 id_to_label은 기존에 학습한 tokenizer.config 또는 label_list 기반으로 정의되어 있어야 함
true_labels, true_preds = [], []

# labels와 preds는 model.predict() 결과에서 얻은 logits 또는 prediction label list
for pred, label in zip(preds, labels):  # preds: List[List[int]], labels: List[List[int]]
    sent_true, sent_pred = [], []
    for p_, l_ in zip(pred, label):
        if l_ != -100:  # ignore special tokens
            sent_true.append(id_to_label[l_])  # 예: 'B-Bacteria', 'O', ...
            sent_pred.append(id_to_label[p_])
    true_labels.append(sent_true)
    true_preds.append(sent_pred)

# 엔티티별 성능 분석
report = seq_classification_report(true_labels, true_preds, output_dict=True)

# ✅ 7. 성능 저장
final_report = seq_classification_report(true_labels, true_preds, digits=4)
report_path = os.path.join(base_dir, "final_test_classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(final_report)

print("✅ 최종 test 성능 저장 완료!\n")
print(final_report)

# ✅ 8. test 엔티티별 성능 저장 (JSON)
test_entity_scores = defaultdict(lambda: {"precision": [], "recall": [], "f1": []})

for label, metrics in report.items():
    if label in ["accuracy", "macro avg", "weighted avg"]:
        continue
    entity = label.split("-")[1] if "-" in label else "O"
    test_entity_scores[entity]["precision"].append(metrics["precision"])
    test_entity_scores[entity]["recall"].append(metrics["recall"])
    test_entity_scores[entity]["f1"].append(metrics["f1-score"])

test_entity_avg = {
    entity: {
        "precision": np.mean(score["precision"]),
        "recall": np.mean(score["recall"]),
        "f1": np.mean(score["f1"])
    }
    for entity, score in test_entity_scores.items()
}

entity_json_path = os.path.join(base_dir, "final_test_entity_report.json")
with open(entity_json_path, "w", encoding="utf-8") as f:
    json.dump(test_entity_avg, f, ensure_ascii=False, indent=2)

print("\u2705 test 엔z티티별 성능 JSON 저장 완료")

# ✅ 9. test 예측 결과 csv 저장
flat_tokens, flat_true, flat_pred = [], [], []
for i, ex in enumerate(dataset_test):
    tokens = test_examples[i]["tokens"]
    token_idx = 0
    for j, word_id in enumerate(ex['labels']):
        if word_id != -100:  # ✅ 수정됨: 실제 토큰과 라벨 매핑
            if token_idx < len(tokens):
                flat_tokens.append(tokens[token_idx])
            else:
                flat_tokens.append("[UNK]")  # ✅ IndexError 방지

            flat_true.append(id_to_label[ex['labels'][j]])
            flat_pred.append(id_to_label[preds[i][j]])
            token_idx += 1
        # if word_id != -100:
        #     flat_tokens.append(tokens[token_idx])
        #     flat_true.append(id_to_label[ex['labels'][j]])
        #     flat_pred.append(id_to_label[preds[i][j]])
        #     token_idx += 1

df_test_results = pd.DataFrame({
    "token": flat_tokens,
    "true_label": flat_true,
    "pred_label": flat_pred
})
csv_path = os.path.join(base_dir, "final_test_predictions.csv")
df_test_results.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("✅ test 예측 결과 CSV 저장 완료:", csv_path)