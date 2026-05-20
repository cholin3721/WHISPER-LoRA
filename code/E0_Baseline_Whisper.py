# ============================================================
# E0 Baseline: Whisper-large-v3 Zero-Shot Korean Elderly Speech
# ============================================================
# Colab에서 실행하세요.
# 사용법:
#   1. New_Sample 폴더를 Google Drive에 업로드
#   2. 이 파일을 Colab에 복사
#   3. 런타임 > 런타임 유형 변경 > GPU (T4) 선택
#   4. 셀 단위로 실행
# ============================================================

# %% [markdown]
# # 🎙️ E0 Baseline: Whisper Zero-Shot 노인 음성 CER 측정

# %% --- Cell 1: 설치 ---
# !pip install -q transformers accelerate jiwer librosa soundfile

# %% --- Cell 2: Google Drive 마운트 ---
# from google.colab import drive
# drive.mount('/content/drive')

# %% --- Cell 3: 경로 설정 ---
import os

# ★ 이 경로를 본인의 Drive 경로에 맞게 수정하세요
DATA_ROOT = "/content/drive/MyDrive/WHISPER/New_Sample"

WAV_DIR = os.path.join(DATA_ROOT, "원천데이터", "1.AI챗봇",
                       "1.AI챗봇_1_자유대화(노인남여)_TRAINING")
LABEL_DIR = os.path.join(DATA_ROOT, "라벨링데이터", "1.AI챗봇",
                         "1.AI챗봇_라벨링_자유대화(노인남여)_TRAINING")

# 경로 확인
print(f"WAV dir exists: {os.path.exists(WAV_DIR)}")
print(f"Label dir exists: {os.path.exists(LABEL_DIR)}")

# %% --- Cell 4: 데이터 로드 ---
import json
import glob

def load_dataset(wav_dir, label_dir):
    """WAV-JSON 쌍을 로드하여 리스트로 반환"""
    dataset = []
    for speaker_folder in sorted(os.listdir(wav_dir)):
        wav_folder = os.path.join(wav_dir, speaker_folder)
        json_folder = os.path.join(label_dir, speaker_folder)

        if not os.path.isdir(wav_folder):
            continue

        # 폴더명에서 메타데이터 파싱
        # 형식: 노인남여_노인대화07_F_1522434093_60_경상_실내
        parts = speaker_folder.split("_")
        gender = parts[2]           # F or M
        speaker_id = parts[3]       # 1522434093
        age = int(parts[4])         # 60
        region = parts[5]           # 경상

        for wav_file in sorted(glob.glob(os.path.join(wav_folder, "*.wav"))):
            basename = os.path.splitext(os.path.basename(wav_file))[0]
            json_file = os.path.join(json_folder, basename + ".json")

            if not os.path.exists(json_file):
                continue

            with open(json_file, "r", encoding="utf-8") as f:
                label = json.load(f)

            transcript = label["발화정보"]["stt"]
            duration = float(label["발화정보"].get("recrdTime", 0))

            dataset.append({
                "wav_path": wav_file,
                "transcript": transcript,
                "duration": duration,
                "speaker_id": speaker_id,
                "gender": gender,
                "age": age,
                "region": region,
            })

    return dataset

dataset = load_dataset(WAV_DIR, LABEL_DIR)
print(f"✅ 로드 완료: {len(dataset)}개 발화")
print(f"   총 길이: {sum(d['duration'] for d in dataset)/60:.1f}분")
print(f"   화자 수: {len(set(d['speaker_id'] for d in dataset))}명")

# 화자별 통계
from collections import Counter
for sid in sorted(set(d["speaker_id"] for d in dataset)):
    items = [d for d in dataset if d["speaker_id"] == sid]
    info = items[0]
    print(f"   [{info['gender']}/{info['age']}세/{info['region']}] "
          f"{len(items)}개, {sum(d['duration'] for d in items)/60:.1f}분")

# %% --- Cell 5: Whisper 모델 로드 ---
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("⏳ 모델 로딩 중... (약 2~3분)")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
model.eval()
print("✅ 모델 로드 완료")

# %% --- Cell 6: 추론 함수 ---
import librosa
import numpy as np
from tqdm import tqdm

def transcribe_batch(dataset, model, processor, device, batch_size=8):
    """Whisper로 배치 추론"""
    results = []
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ko", task="transcribe"
    )

    for i in tqdm(range(0, len(dataset), batch_size), desc="Transcribing"):
        batch = dataset[i:i+batch_size]

        # 오디오 로드
        audios = []
        for item in batch:
            audio, sr = librosa.load(item["wav_path"], sr=16000)
            audios.append(audio)

        # 전처리
        inputs = processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(device)
        if device == "cuda":
            input_features = input_features.half()

        # 추론
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=128,
                num_beams=5,
            )

        # 디코딩
        predictions = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        for item, pred in zip(batch, predictions):
            results.append({
                **item,
                "prediction": pred.strip(),
            })

    return results

# %% --- Cell 7: 추론 실행 ---
print("⏳ 추론 시작... (샘플 2930개, 약 10~20분 소요)")
results = transcribe_batch(dataset, model, processor, device, batch_size=4)
print(f"✅ 추론 완료: {len(results)}개")

# 예시 출력
print("\n--- 샘플 결과 (처음 5개) ---")
for r in results[:5]:
    print(f"  정답: {r['transcript']}")
    print(f"  예측: {r['prediction']}")
    print()

# %% --- Cell 8: CER/WER 계산 ---
from jiwer import cer, wer

def compute_metrics(results):
    """전체 CER, WER 계산"""
    references = [r["transcript"] for r in results]
    hypotheses = [r["prediction"] for r in results]

    # 빈 문자열 방어
    refs_clean = []
    hyps_clean = []
    for ref, hyp in zip(references, hypotheses):
        if ref.strip():
            refs_clean.append(ref.strip())
            hyps_clean.append(hyp.strip() if hyp.strip() else " ")

    overall_cer = cer(refs_clean, hyps_clean)
    overall_wer = wer(refs_clean, hyps_clean)

    return overall_cer, overall_wer, refs_clean, hyps_clean

total_cer, total_wer, refs, hyps = compute_metrics(results)

print("=" * 50)
print(f"📊 E0 Baseline 결과")
print(f"=" * 50)
print(f"  CER: {total_cer*100:.2f}%")
print(f"  WER: {total_wer*100:.2f}%")
print(f"  발화 수: {len(refs)}")
print(f"=" * 50)

# %% --- Cell 9: 오류 유형 분해 (Del/Ins/Sub) ---
from jiwer import process_words

def error_type_analysis(refs, hyps):
    """Deletion, Insertion, Substitution 비율 분석"""
    total_del = 0
    total_ins = 0
    total_sub = 0
    total_hits = 0
    total_ref_len = 0

    for ref, hyp in zip(refs, hyps):
        # 글자 단위로 분해
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))

        ref_str = " ".join(ref_chars)
        hyp_str = " ".join(hyp_chars)

        try:
            output = process_words(ref_str, hyp_str)
            total_hits += output.hits
            total_del += output.deletions
            total_ins += output.insertions
            total_sub += output.substitutions
            total_ref_len += len(ref_chars)
        except Exception:
            total_ref_len += len(ref_chars)

    total_errors = total_del + total_ins + total_sub
    print(f"\n📊 오류 유형 분해 (글자 단위)")
    print(f"  총 정답 글자: {total_ref_len}")
    print(f"  총 오류:     {total_errors}")
    print(f"  ├─ Deletion (누락):     {total_del} ({total_del/max(total_ref_len,1)*100:.1f}%)")
    print(f"  ├─ Insertion (삽입):    {total_ins} ({total_ins/max(total_ref_len,1)*100:.1f}%)")
    print(f"  └─ Substitution (대치): {total_sub} ({total_sub/max(total_ref_len,1)*100:.1f}%)")

    return {
        "deletions": total_del,
        "insertions": total_ins,
        "substitutions": total_sub,
        "ref_length": total_ref_len,
    }

error_stats = error_type_analysis(refs, hyps)

# %% --- Cell 10: 화자별 (Subgroup) 분석 ---
def subgroup_analysis(results):
    """화자별 CER 분석"""
    speakers = sorted(set(r["speaker_id"] for r in results))

    print(f"\n📊 화자별 CER (Subgroup Analysis)")
    print(f"{'화자ID':>12} {'성별':>4} {'나이':>4} {'지역':>6} {'발화수':>6} {'CER':>8}")
    print("-" * 50)

    speaker_cers = {}
    for sid in speakers:
        items = [r for r in results if r["speaker_id"] == sid]
        info = items[0]

        s_refs = [r["transcript"].strip() for r in items if r["transcript"].strip()]
        s_hyps = [r["prediction"].strip() if r["prediction"].strip() else " "
                  for r in items if r["transcript"].strip()]

        if s_refs:
            s_cer = cer(s_refs, s_hyps)
            speaker_cers[sid] = s_cer
            print(f"{sid:>12} {info['gender']:>4} {info['age']:>4} "
                  f"{info['region']:>6} {len(s_refs):>6} {s_cer*100:>7.2f}%")

    return speaker_cers

speaker_cers = subgroup_analysis(results)

# %% --- Cell 11: 결과 저장 ---
import csv
from datetime import datetime

output_file = os.path.join(DATA_ROOT, "..", "E0_baseline_results.csv")

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "wav_path", "speaker_id", "gender", "age", "region",
        "duration", "transcript", "prediction"
    ])
    writer.writeheader()
    for r in results:
        writer.writerow({
            "wav_path": os.path.basename(r["wav_path"]),
            "speaker_id": r["speaker_id"],
            "gender": r["gender"],
            "age": r["age"],
            "region": r["region"],
            "duration": r["duration"],
            "transcript": r["transcript"],
            "prediction": r["prediction"],
        })

print(f"\n✅ 결과 저장: {output_file}")

# %% --- Cell 12: 요약 출력 ---
print("\n" + "=" * 60)
print("🏁 E0 Baseline 최종 요약")
print("=" * 60)
print(f"  모델: {MODEL_NAME}")
print(f"  데이터: AI허브 자유대화 음성(노인남여) 샘플")
print(f"  발화 수: {len(refs)}개 / 화자 {len(speaker_cers)}명")
print(f"  디코딩: beam=5, language=ko")
print(f"")
print(f"  📈 CER: {total_cer*100:.2f}%")
print(f"  📈 WER: {total_wer*100:.2f}%")
print(f"")
print(f"  오류 분해:")
print(f"    Del: {error_stats['deletions']} "
      f"({error_stats['deletions']/max(error_stats['ref_length'],1)*100:.1f}%)")
print(f"    Ins: {error_stats['insertions']} "
      f"({error_stats['insertions']/max(error_stats['ref_length'],1)*100:.1f}%)")
print(f"    Sub: {error_stats['substitutions']} "
      f"({error_stats['substitutions']/max(error_stats['ref_length'],1)*100:.1f}%)")
print(f"")
print(f"  → 이 수치가 E1(LoRA 학습 후)과 비교할 Baseline입니다.")
print("=" * 60)
