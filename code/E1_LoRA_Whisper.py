# ============================================================
# E1: Whisper-large-v3 + LoRA Fine-Tuning on Korean Elderly Speech
# ============================================================
# Colab/학교 GPU 환경에서 실행. Whisper-large-v3에 LoRA(rank=8) 어댑터를 붙여
# 학습하고, E0(zero-shot)와 동일한 평가 파이프라인으로 CER을 비교한다.
#
# 사용법:
#   1. New_Sample.zip(또는 VOTE400)을 Drive/로컬 경로에 두기
#   2. 학교 GPU 또는 Colab A100/L4에 접속 (VRAM 16GB+)
#   3. 셀 단위로 실행
#
# 핵심 설계:
#   - 화자 단위 분할(Speaker-disjoint): 한 화자가 train/val에 동시 출현 금지
#   - 라벨 정제(analyze_cleaned.py와 동일 로직): (NO:)/(SP:)/(FP:)/(SN:) 태그 제거
#   - 평가 지표: 정제된 refs와의 CER (E0와 비교 가능)
#   - LoRA target: q_proj, v_proj (Whisper attention 블록의 query·value projection)
# ============================================================

# %% [markdown]
# # 🔧 E1: Whisper-large-v3 + LoRA 파인튜닝
#
# E0(zero-shot CER 4.25%) 대비 노인 음성에서의 CER 개선을 측정한다.

# %% --- Cell 1: 설치 ---
# !pip install -q transformers==4.44.2 accelerate peft datasets jiwer librosa soundfile

# %% --- Cell 2: (Colab 한정) Drive 마운트 + zip 해제 ---
# from google.colab import drive
# drive.mount('/content/drive')
#
# import os, zipfile
# ZIP_PATH = "/content/drive/MyDrive/New_Sample.zip"  # 또는 VOTE400 zip
# EXTRACT_DIR = "/content/data"
# if not os.path.exists(os.path.join(EXTRACT_DIR, "New_Sample")):
#     with zipfile.ZipFile(ZIP_PATH) as z:
#         z.extractall(EXTRACT_DIR)
# DATA_ROOT = os.path.join(EXTRACT_DIR, "New_Sample")

# %% --- Cell 3: 경로 + 데이터 로드 (E0와 동일 로직) ---
import os
import json
import glob
import re

DATA_ROOT = "/content/data/New_Sample"  # ★ 환경에 맞게 수정
WAV_DIR   = os.path.join(DATA_ROOT, "원천데이터",   "1.AI챗봇", "1.AI챗봇_1_자유대화(노인남여)_TRAINING")
LABEL_DIR = os.path.join(DATA_ROOT, "라벨링데이터", "1.AI챗봇", "1.AI챗봇_라벨링_자유대화(노인남여)_TRAINING")

# (NO:content) -> content  (analyze_cleaned.py와 동일)
TAG_RE = re.compile(r'\(([A-Z]+):([^)]*)\)')
def clean_text(t):
    out = TAG_RE.sub(lambda m: ' ' + m.group(2).strip() + ' ', t)
    return re.sub(r'\s+', ' ', out).strip()

def load_dataset(wav_dir, label_dir):
    dataset = []
    for spk in sorted(os.listdir(wav_dir)):
        wfolder = os.path.join(wav_dir, spk)
        jfolder = os.path.join(label_dir, spk)
        if not os.path.isdir(wfolder):
            continue
        parts = spk.split("_")
        gender, speaker_id, age, region = parts[2], parts[3], int(parts[4]), parts[5]
        for wav in sorted(glob.glob(os.path.join(wfolder, "*.wav"))):
            base = os.path.splitext(os.path.basename(wav))[0]
            jpath = os.path.join(jfolder, base + ".json")
            if not os.path.exists(jpath):
                continue
            with open(jpath, "r", encoding="utf-8") as f:
                label = json.load(f)
            raw = label["발화정보"]["stt"]
            cleaned = clean_text(raw)
            if not cleaned:
                continue
            dataset.append({
                "wav_path": wav,
                "transcript": cleaned,
                "duration": float(label["발화정보"].get("recrdTime", 0)),
                "speaker_id": speaker_id,
                "gender": gender,
                "age": age,
                "region": region,
            })
    return dataset

dataset = load_dataset(WAV_DIR, LABEL_DIR)
print(f"✅ 로드: {len(dataset)}개 발화")
speakers = sorted(set(d["speaker_id"] for d in dataset))
print(f"   화자 {len(speakers)}명: {speakers}")

# %% --- Cell 4: Speaker-disjoint Train / Val 분할 ---
# AI허브 샘플(5명)에서는 leave-one-speaker-out으로 1명을 val에 둠.
# VOTE400으로 갈아끼울 때는 화자 그룹(정상/구음장애/80+)을 보존하며 분할해야 함.

HELD_OUT_SPEAKER = "1527825984"  # ★ CER 가장 높았던 화자(F/71/충청). 어려운 케이스로 val.
                                  # VOTE400에서는 stratified split 별도 구현 필요.

train_data = [d for d in dataset if d["speaker_id"] != HELD_OUT_SPEAKER]
val_data   = [d for d in dataset if d["speaker_id"] == HELD_OUT_SPEAKER]

print(f"Train: {len(train_data)} (화자 {len(set(d['speaker_id'] for d in train_data))}명)")
print(f"Val:   {len(val_data)}  (화자 {HELD_OUT_SPEAKER})")

# %% --- Cell 5: Whisper 모델 + Processor 로드 ---
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0) if device=='cuda' else '-'}")

processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="ko", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# 학습 시에는 forced_decoder_ids 와 suppress_tokens 가 generate 동작을 강제하지 않도록 비움.
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = None

# 메모리 절약
model.gradient_checkpointing_enable()
model.config.use_cache = False  # gradient_checkpointing과 충돌 방지

# %% --- Cell 6: LoRA 어댑터 적용 (peft) ---
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

lora_config = LoraConfig(
    r=8,                       # PPT 명시: rank=8
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Whisper attention의 Q·V
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 기대 출력: trainable params: ~8M / total ~1,550M (≈0.5%)

# %% --- Cell 7: HF Dataset 빌드 + 전처리 ---
import librosa
from datasets import Dataset

def to_hf(item):
    return {
        "wav_path": item["wav_path"],
        "transcript": item["transcript"],
    }

train_ds = Dataset.from_list([to_hf(d) for d in train_data])
val_ds   = Dataset.from_list([to_hf(d) for d in val_data])

def prepare_example(batch):
    audio, _ = librosa.load(batch["wav_path"], sr=16000)
    inputs = processor.feature_extractor(audio, sampling_rate=16000)
    batch["input_features"] = inputs.input_features[0]
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
    return batch

# num_proc=1 권장: librosa+한국어 토크나이저는 멀티프로세싱 시 가끔 deadlock
train_ds = train_ds.map(prepare_example, remove_columns=["wav_path", "transcript"])
val_ds   = val_ds.map(prepare_example,   remove_columns=["wav_path", "transcript"])

# %% --- Cell 8: 데이터 콜레이터 ---
from dataclasses import dataclass
from typing import Any

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        # 오디오 (input_features) 패딩
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 라벨 (labels) 패딩 + 패딩 토큰을 -100 으로 마스킹
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # BOS가 모든 라벨에 붙어있으면 제거 (Whisper decoder_start_token이 BOS와 동일)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# %% --- Cell 9: 평가 함수 (E0와 동일하게 정제된 CER 사용) ---
from jiwer import cer, wer

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100 마스크를 pad_token_id로 복원하여 decode
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 라벨은 이미 학습 단계에서 정제됨. 그래도 안전하게 한 번 더 처리.
    label_str = [clean_text(s) for s in label_str]
    pred_str  = [s.strip() if s.strip() else " " for s in pred_str]

    return {
        "cer": cer(label_str, pred_str),
        "wer": wer(label_str, pred_str),
    }

# %% --- Cell 10: 학습 설정 + Trainer ---
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

OUT_DIR = "/content/whisper-lora-elderly"  # ★ 학교 GPU에서는 영구 저장 경로로 변경

training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,        # T4/L4는 4, A100 24GB+면 8~16
    gradient_accumulation_steps=4,         # 유효 batch 16
    learning_rate=1e-4,                    # PPT 명시
    warmup_steps=50,
    num_train_epochs=3,                    # AI허브 샘플(소량) 기준 3epoch. VOTE400은 1~2면 충분할 수 있음
    fp16=True,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=25,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=5,
    save_total_limit=2,
    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,
    report_to=[],                          # wandb 쓰면 ["wandb"]
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# %% --- Cell 11: 학습 ---
print("⏳ 학습 시작…")
trainer.train()
print("✅ 학습 완료")

# %% --- Cell 12: 최종 평가 + E0 비교 ---
metrics = trainer.evaluate()
print("=" * 50)
print(f"📊 E1 (LoRA) 결과 — held-out 화자: {HELD_OUT_SPEAKER}")
print(f"   CER: {metrics['eval_cer']*100:.2f}%")
print(f"   WER: {metrics['eval_wer']*100:.2f}%")
print("=" * 50)
print()
print("E0 (zero-shot, 정제 후) 동일 화자 CER: 5.14% (참고)")
print("→ 상대 개선율: "
      f"{(1 - metrics['eval_cer']/0.0514)*100:.1f}% "
      "(목표 30%+)")

# %% --- Cell 13: 어댑터 저장 ---
model.save_pretrained(os.path.join(OUT_DIR, "adapter_final"))
processor.save_pretrained(os.path.join(OUT_DIR, "adapter_final"))
print(f"✅ 어댑터 저장: {OUT_DIR}/adapter_final  (~30MB)")

# %% [markdown]
# ## 다음 단계
# 1. **VOTE400 도착 시**: Cell 3의 폴더 파싱 로직(`parts = spk.split("_")`)을 VOTE400 명명 규칙에 맞춰 수정.
# 2. **Stratified Split**: 정상/구음장애/80+ 그룹을 train/val에 비율 보존하여 분할. AI허브 5명 leave-one-out과는 다른 로직 필요.
# 3. **E2 (VAD 재튜닝)**: 본 스크립트에 inference 전 VAD 임계값 조정 단계 추가. 10초+ 발화의 Del% 감소를 측정.
