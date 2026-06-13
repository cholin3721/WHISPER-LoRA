# WHISPER-LoRA

**한국어 노인 음성 인식을 위한 Whisper PEFT(LoRA) 파인튜닝 연구**

> VOTE400 낭독체에서 zero-shot CER **20.38% → 10.25%** (상대 **49.7%** 감소). 어려운 그룹 JN2는 **34.94% → 12.40%** (상대 **65%** 감소).

본 저장소는 응용 앱 [**Recorder-App (이야기봄)**](https://github.com/cholin3721/Recorder-App)의 **STT 엔진 학습 코드와 평가 결과**를 담고 있다. 학습된 LoRA 어댑터(19MB)는 그 앱의 백엔드 inference 서비스에서 사용된다.

---

## 결과 요약

| 실험 | 데이터 | 발화 수 | 결과 |
|---|---|---:|---|
| **E0** | AI허브 자유대화(노인남여) | 2,930 | CER **4.25%** (cleaned, raw 6.85%) |
| **E2** | AI허브 10초+ 발화 | 281 | ΔCER **+0.80%p** (H2 가설 반박) |
| **E0'** | VOTE400 낭독체 샘플 | 3,200 | CER **20.38%**, JN2 34.94% |
| **E1'** | VOTE400 낭독체 학습 | 95,110 train | **CER 10.25%** (49.7% 상대 개선) |

### 가설 검증 결과

| 가설 | 결과 |
|---|---|
| **H1**: LoRA로 노인 음성 CER 30%+ 감소 | ✅ **지지** (49.7% 상대 개선 달성) |
| **H2**: 외부 VAD 재튜닝으로 Deletion 감소 | ❌ **반박** (오히려 +0.80%p 악화) |

---

## 학습 설정 (E1')

| 항목 | 값 |
|---|---|
| Base 모델 | `openai/whisper-large-v3` (1.55B params) |
| LoRA 구성 | r=8, α=32, target=`q_proj`/`v_proj`, dropout=0.05 |
| 학습 가능 파라미터 | **3.9M / 1.55B (0.25%)** |
| Train / Val | 95,110 / 16,704 (Val 서브샘플 1,500) |
| Split | 그룹(region+group) stratified + 화자(pid) disjoint, seed=42 |
| 배치 | per_device 8 × accum 2 × 2 GPU = 유효 32 |
| Epoch | 2 (5,946 step) |
| LR | 1e-4 (warmup 50) |
| 정밀도 | **fp16** (RTX 8000 Turing이라 bf16 불가) |
| 학습 시간 | **6.9시간** (2-GPU DDP) |

---

## 그룹별 CER (E1', val 1,500 발화)

| 그룹 | E0' CER | E1' CER | 절대 개선 |
|---|---:|---:|---:|
| **JN2** ⭐ | **34.94%** | **12.40%** | **−22.54%p** |
| GW2 | 24.88% | 9.66% | −15.22%p |
| DG4 | 21.52% | 11.28% | −10.24%p |
| GN2 | 19.74% | 9.15% | −10.59%p |
| (... 나머지 12 그룹 평균) | ~19% | ~10% | ~9%p |

**핵심 발견**: zero-shot에서 가장 어려웠던 그룹(JN2)이 가장 큰 절대 개선을 보임. LoRA가 모델의 약점을 표적 적응함을 시사.

---

## 디렉토리 구조

```
WHISPER/
├── CLAUDE.md                # 프로젝트 가이드 (Claude Code용)
├── README.md                # 본 파일
├── code/                    # 모든 Python 스크립트 + 노트북
│   ├── E0_Baseline_Whisper.py / .ipynb     # AI허브 zero-shot
│   ├── E1_LoRA_Whisper.py                  # AI허브 LoRA (사용 X)
│   ├── E1prime_VOTE400_LoRA.py             # ⭐ VOTE400 LoRA 학습 (본 게임)
│   ├── E1prime_VOTE400_predict.py          # adapter_final 재추론
│   ├── E2_H2_VAD_Experiment.py / .ipynb    # H2 VAD ablation
│   ├── E0prime_VOTE400_baseline.py         # VOTE400 zero-shot 측정
│   ├── vote400_loader.py                   # VOTE400 데이터 로더
│   ├── analyze_results.py                  # CER 재계산
│   ├── analyze_deep.py                     # 화자×길이×오류 분해
│   ├── analyze_cleaned.py                  # AI허브 태그 정제 후 측정
│   ├── make_notebook.py / py_to_ipynb.py   # .py↔.ipynb 변환
│   └── E0_baseline_results.csv …          # (생성된 결과 CSV들)
├── results/                 # CSV + 분석 리포트 + 학습 로그
│   ├── E0_baseline_results.csv
│   ├── E0_baseline_results_cleaned.csv
│   ├── E2_h2_vad_results.csv
│   ├── E0prime_vote400_read_sampled.csv
│   ├── E1prime_vote400_read_pred.csv
│   ├── E1prime_metrics.csv
│   ├── E1prime_train_log.txt
│   ├── E1prime_results.md
│   ├── vote400_read_index.csv             # 낭독체 112,984개 인덱스
│   ├── vote400_dialog_index.csv           # 대화체 1,170개 인덱스
│   └── analysis_*_report.md
├── whisper-lora-vote400/    # ⭐ 학습된 LoRA 어댑터 (19MB)
│   └── adapter_final/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
├── data/                    # ⚠️ git 제외 — VOTE400 EULA로 재배포 불가
│   ├── New_Sample/          # AI허브 5명 샘플
│   └── VOTE400/             # VOTE400 약 400시간
└── docs/
    ├── presentation/        # PPT/PDF/발표 대본
    ├── vote400/             # EULA + 협약서
    └── reports/             # 제출 보고서 (피우다·공모전용)
```

---

## 재현 (Reproduction)

### 환경
- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- `transformers`, `peft`, `accelerate`, `jiwer`, `librosa`, `soundfile`

### AI허브 분석 (로컬, GPU 불필요)
```bash
python code/analyze_results.py      # raw CER 재계산
python code/analyze_deep.py         # 화자×길이×오류 분해 + worst-20
python code/analyze_cleaned.py      # TAG_RE 정제 후 재측정
```

### VOTE400 인덱싱 (GPU 불필요)
```bash
python code/vote400_loader.py --root data/VOTE400 --mode both --out-dir results
```

### VOTE400 zero-shot 측정 (GPU 필요)
```bash
python code/E0prime_VOTE400_baseline.py \
    --root data/VOTE400 --sample-per-group 200 \
    --out results/E0prime_vote400_read_sampled.csv
```

### LoRA 학습 (학교 GPU 서버, 2-GPU DDP)
```bash
torchrun --nproc_per_node=2 code/E1prime_VOTE400_LoRA.py \
    --root data/VOTE400 --out-dir whisper-lora-vote400
```

---

## AI허브 라벨 정제 — 메소드론적 발견

AI허브 transcription에는 `(NO:)`, `(SP:)`, `(FP:)`, `(SN:)` 4종 태그가 12% 발화에 포함되어 있다. Whisper는 깨끗한 본문만 출력하므로 단순 비교 시 모두 Deletion 오류로 잡힌다.

```python
TAG_RE = re.compile(r'\(([A-Z]+):([^)]*)\)')
def clean_text(t):
    out = TAG_RE.sub(lambda m: ' ' + m.group(2).strip() + ' ', t)
    return re.sub(r'\s+', ' ', out).strip()
```

| 지표 | Raw | Cleaned | Δ |
|---|---:|---:|---|
| CER | 6.85% | **4.25%** | -2.60%p |
| Deletion | 4.25% | **0.69%** | **-3.56%p** |

→ **측정된 Deletion의 85%가 모델 오류가 아니라 라벨 형식 차이**였음. VOTE400에는 이 함정이 없다 (정제 후 +0.70%p, 거의 무변).

---

## 데이터셋

### AI허브 "자유대화(노인남여)" 샘플
- 5명, 2,930 발화, 약 5.3시간
- 출처: https://aihub.or.kr/

### MINDsLab-ETRI VOTE400
- 약 400시간 (낭독체 100h + 대화체 300h)
- 화자 100명+, 5개 지역, 16개 그룹
- 협약 체결 후 사용 (학생/연구 범위만)

**사사문구 (필수)**: 본 결과물은 ㈜마인즈랩과 한국전자통신연구원이 연구 과제 수행을 통해 구축 공개한 고령자 음성데이터 VOTE400 데이터셋을 사용함. 해당 연구 과제는 미래창조과학부 및 정보통신기술진흥센터의 정보통신·방송 연구개발 사업의 일환으로 수행하였음. [2017-0-00162]

---

## 응용

본 LoRA 어댑터는 [**Recorder-App (이야기봄)**](https://github.com/cholin3721/Recorder-App)의 STT 백엔드로 사용된다:
- **2026 현대 오토에버 배리어프리 앱 공모전** 출품작
- 노인 구술 → STT → Gemini 2.5-flash로 무협/로맨스/판타지/역사 소설 각색
- Kotlin Android + Python FastAPI 백엔드

---

## 라이선스 및 사용 범위

- 코드: MIT (예정)
- LoRA 어댑터: VOTE400 EULA 종속 — **상업적 이용 불허**, 학생·연구 범위
- 협약 효력: 2026-12-31까지

---

*작성: 202444085 김철중. 2026.*
