import json

cells = []

def add_md(text):
    lines = text.split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source})

def add_code(text):
    lines = text.split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': source,
                  'outputs': [], 'execution_count': None})

# ===== CELLS =====

add_md("""# 🎙️ E0 Baseline: Whisper Zero-Shot 노인 음성 CER 측정

**목적:** Whisper-large-v3를 파인튜닝 없이 노인 음성에 돌려서 Baseline CER을 측정합니다.

### 사용법
1. `New_Sample.zip`을 Google Drive에 업로드
2. 런타임 > 런타임 유형 변경 > **GPU (T4)** 선택
3. 셀 순서대로 실행""")

add_md("## 1. 설치 및 환경 준비")

add_code("!pip install -q transformers accelerate jiwer librosa soundfile")

add_code("""from google.colab import drive
drive.mount('/content/drive')""")

add_md("""## 2. 데이터 압축 해제

Google Drive에 `New_Sample.zip`을 업로드한 후 아래 셀을 실행하세요.
경로가 다르면 수정해주세요.""")

add_code("""import os
import zipfile

# ★ Google Drive에 업로드한 zip 파일 경로
ZIP_PATH = "/content/drive/MyDrive/New_Sample.zip"
EXTRACT_DIR = "/content/data"

if not os.path.exists(os.path.join(EXTRACT_DIR, "New_Sample")):
    print("⏳ 압축 해제 중...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_DIR)
    print("✅ 압축 해제 완료")
else:
    print("✅ 이미 해제됨")

DATA_ROOT = os.path.join(EXTRACT_DIR, "New_Sample")

WAV_DIR = os.path.join(DATA_ROOT, "원천데이터", "1.AI챗봇",
                       "1.AI챗봇_1_자유대화(노인남여)_TRAINING")
LABEL_DIR = os.path.join(DATA_ROOT, "라벨링데이터", "1.AI챗봇",
                         "1.AI챗봇_라벨링_자유대화(노인남여)_TRAINING")

print(f"WAV dir exists: {os.path.exists(WAV_DIR)}")
print(f"Label dir exists: {os.path.exists(LABEL_DIR)}")""")

add_md("## 3. 데이터 로드")

add_code("""import json as json_lib
import glob
from collections import Counter

def load_dataset(wav_dir, label_dir):
    dataset = []
    for speaker_folder in sorted(os.listdir(wav_dir)):
        wav_folder = os.path.join(wav_dir, speaker_folder)
        json_folder = os.path.join(label_dir, speaker_folder)
        if not os.path.isdir(wav_folder):
            continue

        parts = speaker_folder.split("_")
        gender = parts[2]
        speaker_id = parts[3]
        age = int(parts[4])
        region = parts[5]

        for wav_file in sorted(glob.glob(os.path.join(wav_folder, "*.wav"))):
            basename = os.path.splitext(os.path.basename(wav_file))[0]
            json_file = os.path.join(json_folder, basename + ".json")
            if not os.path.exists(json_file):
                continue

            with open(json_file, "r", encoding="utf-8") as f:
                label = json_lib.load(f)

            dataset.append({
                "wav_path": wav_file,
                "transcript": label["발화정보"]["stt"],
                "duration": float(label["발화정보"].get("recrdTime", 0)),
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

for sid in sorted(set(d["speaker_id"] for d in dataset)):
    items = [d for d in dataset if d["speaker_id"] == sid]
    info = items[0]
    mins = sum(d['duration'] for d in items) / 60
    print(f"   [{info['gender']}/{info['age']}세/{info['region']}] {len(items)}개, {mins:.1f}분")""")

add_md("## 4. Whisper-large-v3 모델 로드")

add_code("""import torch
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
print("✅ 모델 로드 완료")""")

add_md("## 5. 추론 실행")

add_code("""import librosa
import numpy as np
from tqdm import tqdm

def transcribe_batch(data, model, processor, device, batch_size=4):
    results = []
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ko", task="transcribe"
    )
    for i in tqdm(range(0, len(data), batch_size), desc="Transcribing"):
        batch = data[i:i+batch_size]
        audios = [librosa.load(item["wav_path"], sr=16000)[0] for item in batch]

        inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        feats = inputs.input_features.to(device)
        if device == "cuda":
            feats = feats.half()

        with torch.no_grad():
            pred_ids = model.generate(
                feats,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=128,
                num_beams=5,
            )
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        for item, pred in zip(batch, preds):
            results.append({**item, "prediction": pred.strip()})
    return results

print("⏳ 추론 시작... (약 10~20분)")
results = transcribe_batch(dataset, model, processor, device)
print(f"✅ 추론 완료: {len(results)}개")""")

add_md("## 6. 샘플 결과 확인")

add_code("""for r in results[:10]:
    print(f"정답: {r['transcript']}")
    print(f"예측: {r['prediction']}")
    print()""")

add_md("## 7. CER / WER 계산")

add_code("""from jiwer import cer, wer

refs = [r["transcript"].strip() for r in results if r["transcript"].strip()]
hyps = [r["prediction"].strip() if r["prediction"].strip() else " "
        for r in results if r["transcript"].strip()]

total_cer = cer(refs, hyps)
total_wer = wer(refs, hyps)

print("=" * 50)
print("📊 E0 Baseline 결과")
print("=" * 50)
print(f"  CER: {total_cer*100:.2f}%")
print(f"  WER: {total_wer*100:.2f}%")
print(f"  발화 수: {len(refs)}")
print("=" * 50)""")

add_md("## 8. 오류 유형 분해 (Del / Ins / Sub)\n\nH2 가설 검증 핵심. Deletion(누락)이 노인 음성에서 특히 높을 것으로 예상됩니다.")

add_code("""from jiwer import process_words

td, ti, ts, tr = 0, 0, 0, 0
for ref, hyp in zip(refs, hyps):
    rc = list(ref.replace(" ", ""))
    hc = list(hyp.replace(" ", ""))
    try:
        out = process_words(" ".join(rc), " ".join(hc))
        td += out.deletions
        ti += out.insertions
        ts += out.substitutions
        tr += len(rc)
    except Exception:
        tr += len(rc)

n = max(tr, 1)
print("📊 오류 유형 분해 (글자 단위)")
print(f"  총 정답 글자: {tr}")
print(f"  Deletion (누락):     {td} ({td/n*100:.1f}%)")
print(f"  Insertion (삽입):    {ti} ({ti/n*100:.1f}%)")
print(f"  Substitution (대치): {ts} ({ts/n*100:.1f}%)")""")

add_md("## 9. 화자별 CER (Subgroup Analysis)")

add_code("""speaker_ids = sorted(set(r["speaker_id"] for r in results))
print(f"{'화자ID':>14} {'성별':>4} {'나이':>4} {'지역':>6} {'발화수':>6} {'CER':>8}")
print("-" * 50)

for sid in speaker_ids:
    items = [r for r in results if r["speaker_id"] == sid]
    info = items[0]
    sr = [r["transcript"].strip() for r in items if r["transcript"].strip()]
    sh = [r["prediction"].strip() if r["prediction"].strip() else " "
          for r in items if r["transcript"].strip()]
    if sr:
        sc = cer(sr, sh)
        print(f"  {sid:>12} {info['gender']:>4} {info['age']:>4} "
              f"{info['region']:>6} {len(sr):>6} {sc*100:>7.2f}%")""")

add_md("## 10. 결과 저장")

add_code("""import csv

output_file = "/content/drive/MyDrive/E0_baseline_results.csv"
with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "wav_path","speaker_id","gender","age","region",
        "duration","transcript","prediction"
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
print(f"✅ 결과 저장: {output_file}")
print()
print("→ 이 E0 CER이 E1(LoRA 학습 후)과 비교할 기준선입니다.")""")

# ===== BUILD =====
nb = {
    'nbformat': 4,
    'nbformat_minor': 0,
    'metadata': {
        'colab': {'provenance': [], 'gpuType': 'T4'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'},
        'accelerator': 'GPU'
    },
    'cells': cells
}

from pathlib import Path
path = Path(__file__).resolve().parent / 'E0_Baseline_Whisper.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'✅ Created: {path}')
