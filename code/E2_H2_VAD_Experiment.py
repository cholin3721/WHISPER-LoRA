# %% [markdown]
# # E2 (H2 Mini-Ablation): Whisper + 노인용 VAD 사전분할
#
# **가설(H2)**: VAD 임계값을 노인 음성 특성에 맞게 조정(2초+ 침묵 허용)하면 Deletion 오류가 감소한다.
#
# **데이터**: AI허브 자유대화(노인남여) 샘플 중 **10초 이상 발화 281개** — E0에서 Del% 2.34%로 다른 구간(0.40~0.65%)보다 4~5배 높았던 구간.
#
# **조건**:
# - **A (Baseline)**: E0와 동일 설정 (`beam=5, max_new_tokens=128, language=ko`). 풀 오디오 입력.
# - **B (Elderly VAD)**: Silero VAD로 `min_silence_duration_ms=2000` 기준 사전 분할 → 분할된 speech 청크만 합쳐서 Whisper에 입력 (긴 침묵은 제거되어 모델이 "끝났다"고 오인할 신호가 사라짐).
#
# **측정**: 전체 CER, Del/Sub/Ins, 화자별 CER. E0 cleaned baseline(5.14% on 충청 71세 등)과 비교.

# %% --- Cell 1: 설치 ---
# !pip install -q transformers accelerate jiwer librosa soundfile
# !pip install -q silero-vad

# %% --- Cell 2: (Colab) Drive 마운트 + 데이터 압축 해제 ---
from google.colab import drive
drive.mount('/content/drive')

import os
import zipfile

ZIP_PATH = "/content/drive/MyDrive/New_Sample.zip"  # ★ 본인 경로
EXTRACT_DIR = "/content/data"

if not os.path.exists(os.path.join(EXTRACT_DIR, "New_Sample")):
    print("⏳ 압축 해제 중...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR)
    print("✅ 완료")

DATA_ROOT = os.path.join(EXTRACT_DIR, "New_Sample")
WAV_DIR   = os.path.join(DATA_ROOT, "원천데이터",   "1.AI챗봇", "1.AI챗봇_1_자유대화(노인남여)_TRAINING")
LABEL_DIR = os.path.join(DATA_ROOT, "라벨링데이터", "1.AI챗봇", "1.AI챗봇_라벨링_자유대화(노인남여)_TRAINING")

# %% --- Cell 3: 데이터 로드 + 라벨 정제 + 10초+ 필터 ---
import json
import glob
import re

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
                "gender": gender, "age": age, "region": region,
            })
    return dataset

full_dataset = load_dataset(WAV_DIR, LABEL_DIR)
print(f"전체 발화: {len(full_dataset)}")

# 10초+ 만 필터
dataset = [d for d in full_dataset if d["duration"] >= 10.0]
print(f"10초+ 발화: {len(dataset)}  (기대값: 281 ± 약간)")
print(f"   평균 길이: {sum(d['duration'] for d in dataset)/len(dataset):.1f}초")
print(f"   화자 분포:")
from collections import Counter
for sid, n in Counter(d["speaker_id"] for d in dataset).most_common():
    print(f"     {sid}: {n}개")

# %% --- Cell 4: Whisper 모델 로드 ---
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_NAME = "openai/whisper-large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
model.eval()
print("✅ 모델 로드 완료")

# %% --- Cell 5: Silero VAD 로드 ---
import torch

vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False,
)
(get_speech_timestamps, _save_audio, read_audio, _VADIterator, collect_chunks) = vad_utils
print("✅ Silero VAD 로드 완료")

# %% --- Cell 6: 추론 헬퍼 ---
import librosa
import numpy as np
from tqdm import tqdm

def whisper_transcribe(audio_array, processor, model, device):
    """단일 오디오 → 텍스트. E0와 동일 설정."""
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
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
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def transcribe_condition_A(item):
    """조건 A: 풀 오디오 입력 (E0 베이스라인 재측정)."""
    audio, _ = librosa.load(item["wav_path"], sr=16000)
    return whisper_transcribe(audio, processor, model, device)


def transcribe_condition_B(item, min_silence_ms=2000):
    """조건 B: Silero VAD로 사전 분할 → 침묵 제거된 합본 → Whisper."""
    audio, _ = librosa.load(item["wav_path"], sr=16000)
    audio_t = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(
        audio_t,
        vad_model,
        sampling_rate=16000,
        min_silence_duration_ms=min_silence_ms,   # ★ 핵심: 노인 임계값 (default 100ms → 2000ms)
        min_speech_duration_ms=250,
    )
    if not timestamps:
        # 발화 자체를 못 잡으면 원본 그대로
        return whisper_transcribe(audio, processor, model, device)
    speech_only = collect_chunks(timestamps, audio_t).numpy()
    return whisper_transcribe(speech_only, processor, model, device)

# %% --- Cell 7: 조건 A — Baseline 추론 ---
print("⏳ 조건 A (Baseline) 추론 시작...")
for item in tqdm(dataset, desc="A: Baseline"):
    item["pred_A"] = transcribe_condition_A(item)
print(f"✅ 조건 A 완료: {len(dataset)}개")

# %% --- Cell 8: 조건 B — Elderly VAD 추론 ---
print("⏳ 조건 B (Elderly VAD, min_silence=2000ms) 추론 시작...")
for item in tqdm(dataset, desc="B: Elderly VAD"):
    item["pred_B"] = transcribe_condition_B(item, min_silence_ms=2000)
print(f"✅ 조건 B 완료: {len(dataset)}개")

# %% --- Cell 9: 메트릭 계산 + 비교 ---
from jiwer import cer, wer, process_words

def char_err(ref, hyp):
    rc = list(ref.replace(" ", ""))
    hc = list(hyp.replace(" ", ""))
    if not rc:
        return 0, 0, 0, 0
    try:
        o = process_words(" ".join(rc), " ".join(hc))
        return o.deletions, o.insertions, o.substitutions, len(rc)
    except Exception:
        return 0, 0, 0, len(rc)


def summarize(pred_key, label):
    refs = [d["transcript"] for d in dataset]
    hyps = [d[pred_key].strip() if d[pred_key].strip() else " " for d in dataset]
    c = cer(refs, hyps)
    w = wer(refs, hyps)
    td, ti, ts, tn = 0, 0, 0, 0
    for r, h in zip(refs, hyps):
        d, i, s, n = char_err(r, h)
        td += d; ti += i; ts += s; tn += n
    return {
        "label": label, "cer": c, "wer": w,
        "del_pct": td / max(tn, 1), "sub_pct": ts / max(tn, 1), "ins_pct": ti / max(tn, 1),
        "ref_chars": tn,
    }


summary_A = summarize("pred_A", "A: Baseline")
summary_B = summarize("pred_B", "B: Elderly VAD (silence=2s)")

print("=" * 60)
print(f"📊 H2 미니 ablation 결과 (10초+ 발화 {len(dataset)}개)")
print("=" * 60)
print(f"{'조건':<28} {'CER':>8} {'Del%':>8} {'Sub%':>8} {'Ins%':>8}")
for s in [summary_A, summary_B]:
    print(f"{s['label']:<28} {s['cer']*100:>7.2f}% "
          f"{s['del_pct']*100:>7.2f}% {s['sub_pct']*100:>7.2f}% {s['ins_pct']*100:>7.2f}%")
print()
delta_cer = (summary_B['cer'] - summary_A['cer']) * 100
delta_del = (summary_B['del_pct'] - summary_A['del_pct']) * 100
print(f"Δ CER: {delta_cer:+.2f}%p   (음수면 VAD가 도움)")
print(f"Δ Del: {delta_del:+.2f}%p   ← H2 가설의 핵심 지표")
print("=" * 60)

# %% --- Cell 10: 화자별 분해 ---
from collections import defaultdict

speakers = defaultdict(list)
for d in dataset:
    speakers[d["speaker_id"]].append(d)

print(f"\n화자별 CER (조건 A → B)")
print(f"{'화자ID':>14} {'성/나이/지역':>14} {'#발화':>6} {'CER A':>8} {'CER B':>8} {'Δ':>8}")
print("-" * 70)
for sid in sorted(speakers.keys()):
    items = speakers[sid]
    info = items[0]
    refs = [d["transcript"] for d in items]
    hA = [d["pred_A"].strip() if d["pred_A"].strip() else " " for d in items]
    hB = [d["pred_B"].strip() if d["pred_B"].strip() else " " for d in items]
    cA = cer(refs, hA)
    cB = cer(refs, hB)
    print(f"{sid:>14} {info['gender']}/{info['age']}/{info['region']:>6} "
          f"{len(items):>6} {cA*100:>7.2f}% {cB*100:>7.2f}% "
          f"{(cB-cA)*100:>+7.2f}%p")

# %% --- Cell 11: 정성 분석 — VAD가 가장 크게 도움/해친 사례 ---
diffs = []
for d in dataset:
    cA = cer([d["transcript"]], [d["pred_A"].strip() or " "])
    cB = cer([d["transcript"]], [d["pred_B"].strip() or " "])
    diffs.append((cA - cB, d, cA, cB))

diffs.sort(key=lambda x: x[0], reverse=True)
print("\n=== VAD가 가장 크게 개선한 사례 (top 5) ===")
for delta, d, cA, cB in diffs[:5]:
    print(f"[Δ {delta*100:+.1f}%p | {d['duration']:.1f}s | {d['speaker_id']}]")
    print(f"  정답: {d['transcript'][:80]}")
    print(f"  A   : {d['pred_A'][:80]}")
    print(f"  B   : {d['pred_B'][:80]}")
    print()

print("=== VAD가 가장 크게 해친 사례 (bottom 5) ===")
for delta, d, cA, cB in diffs[-5:]:
    print(f"[Δ {delta*100:+.1f}%p | {d['duration']:.1f}s | {d['speaker_id']}]")
    print(f"  정답: {d['transcript'][:80]}")
    print(f"  A   : {d['pred_A'][:80]}")
    print(f"  B   : {d['pred_B'][:80]}")
    print()

# %% --- Cell 12: 결과 저장 ---
import csv

OUT_CSV = "/content/drive/MyDrive/E2_h2_vad_results.csv"
with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "wav_path", "speaker_id", "gender", "age", "region", "duration",
        "transcript", "pred_A", "pred_B",
    ])
    writer.writeheader()
    for d in dataset:
        writer.writerow({
            "wav_path": os.path.basename(d["wav_path"]),
            "speaker_id": d["speaker_id"], "gender": d["gender"],
            "age": d["age"], "region": d["region"], "duration": d["duration"],
            "transcript": d["transcript"],
            "pred_A": d["pred_A"], "pred_B": d["pred_B"],
        })

print(f"\n✅ 결과 저장: {OUT_CSV}")

# %% [markdown]
# ## 결과 해석 가이드
#
# - **Δ CER < 0 (VAD 도움)**: H2 가설 지지. 발표 슬라이드에 "VAD 재튜닝만으로 X% 개선" 추가.
# - **Δ CER > 0 (VAD 해침)**: H2 단독으로는 부족 → "LoRA 파인튜닝(E1)이 필요한 이유" 논리로 활용.
# - **Δ Del 만 음수, CER은 거의 같음**: VAD가 Deletion은 줄였지만 Substitution을 만들어냄(분할 경계가 단어 중간을 잘랐을 가능성). 흥미로운 trade-off — 본 실험의 가장 중요한 발견 가능성.
#
# ## 다음 단계
# - 결과가 좋으면 → VOTE400에서 동일 ablation을 stratified subgroup별로 반복
# - 결과가 나빠도 → "외부 VAD vs 내부 LoRA-tuned decoder" 비교 narrative로 E1 정당화
