# 2. 진척 사항 (Progress)

**프로젝트**: WHISPER-LoRA
**기간**: 2026년 4월 ~ 6월 (현재 6월 5일까지)
**작성자**: 202444085 김철중

---

## 진척 한눈에 보기

```
[완료]  Phase 1: 연구 설계 + 데이터셋 협약 (4월)
[완료]  Phase 2: AI허브 베이스라인 측정 (E0)
[완료]  Phase 3: 가설 검증 미니 실험 (E2 — H2 ablation)
[완료]  Phase 4: 발표 자료 + 중간 보고
[완료]  Phase 5: 학교 GPU 서버 셋업
[완료]  Phase 6: VOTE400 다운로드 + 베이스라인 (E0')
[완료]  Phase 7: E1' LoRA 학습 + 평가 ⭐ (6.9시간 GPU)
[진행]  Phase 8: 결과 분석 + 최종 보고서
```

전체 일정의 **약 95% 완료**. 최종 발표 자료 작성만 남음.

---

## Phase 1 — 연구 설계 (4월 초)

### 핵심 결정
- 연구 문제: 한국어 노인 음성 인식 (CER 개선)
- 기반 모델: Whisper-large-v3 (1.55B params)
- 기법: PEFT (LoRA, rank=8)
- 가설:
  - **H1 (주가설)**: LoRA 파인튜닝으로 노인 음성 CER이 baseline 대비 통계적으로 유의하게 감소
  - **H2 (부가설)**: VAD 임계값 재조정으로 Deletion 오류 추가 감소
- 통제 변수: paired t-test, bootstrap 95% CI, 화자 단위 분할

### 데이터셋 협약
- AI허브 "자유대화(노인남여)" 샘플 — 즉시 사용
- MINDsLab-ETRI VOTE400 — 협약서 제출 후 승인 (5월)

---

## Phase 2 — E0: AI허브 베이스라인 (5월 초)

### 환경
- Google Colab Pro (T4 GPU), 약 15분 추론 (2,930 발화)

### 결과 (정제 후)
```
전체 CER: 4.25%   (raw 6.85% → cleaned 4.25%, -2.60%p)
전체 WER: 17.69%
오류 분해: Del 0.69% / Sub 1.74% / Ins 0.17%
화자별 CER: 3.08% (F/63/수도권) ~ 5.58% (F/60/경상)
```

### 발견 — 라벨 형식 함정
Transcription에 `(NO:)`, `(SP:)`, `(FP:)`, `(SN:)` 4종 태그가 12% 발화에 포함됨. 정제 후:
- CER 6.85% → **4.25%** (-2.60%p)
- Deletion 4.25% → **0.69%** (-3.56%p)

→ **측정된 Deletion의 85%가 라벨 형식 차이였음**.

### 산출
- [`code/E0_Baseline_Whisper.py`](../../../code/E0_Baseline_Whisper.py)
- [`code/analyze_results.py`](../../../code/analyze_results.py), [`code/analyze_deep.py`](../../../code/analyze_deep.py), [`code/analyze_cleaned.py`](../../../code/analyze_cleaned.py)
- [`results/E0_baseline_results.csv`](../../../results/E0_baseline_results.csv) (raw)
- [`results/E0_baseline_results_cleaned.csv`](../../../results/E0_baseline_results_cleaned.csv) (정제 후)

---

## Phase 3 — E2: H2 미니 ablation (5월 중순)

### 실험 설계
- 데이터: AI허브 10초+ 발화 281개
- A: Whisper 기본 / B: Silero VAD (`min_silence_duration_ms=2000`) 사전분할

### 결과 — 가설 반박

| 조건 | CER | Del% |
|---|---:|---:|
| A (Baseline) | 6.23% | 2.37% |
| B (VAD) | **7.03%** | **3.11%** |
| Δ | **+0.80%p** | **+0.74%p** |

### 정성 발견
VAD가 노인 반복 발화를 잘라먹음 ("그래서 병원 그래서 병원에..." → VAD: "그래서 병원에...")

### 결론
**H2는 데이터로 반박**. Whisper 내부 디코더가 이미 발화 내 침묵 강건 처리. 외부 전처리보다 모델 내부 적응(LoRA)이 필요.

### 산출
- [`code/E2_H2_VAD_Experiment.py`](../../../code/E2_H2_VAD_Experiment.py)
- [`code/py_to_ipynb.py`](../../../code/py_to_ipynb.py) (범용 변환기, 이번에 신규 작성)
- [`results/E2_h2_vad_results.csv`](../../../results/E2_h2_vad_results.csv)

---

## Phase 4 — 발표 자료 (5월 중순)

11주차 스타트업 발표 완료. E0/E2 결과로 "가설 → 실험 → 일부 반박 → 방향 재설정" narrative 시연.

---

## Phase 5 — 학교 GPU 서버 셋업 (5월 26-27일)

### 하드웨어
| 항목 | 사양 |
|---|---|
| GPU | NVIDIA Quadro RTX 8000 × 2 (각 48GB VRAM, Turing) |
| CPU | Intel Xeon Gold 6240 × 2 |
| RAM | 512GB |
| SSD | 1TB |
| Ethernet | 100Mbps |

### 셋업
1. SSH 키 등록
2. `/storage/cholin2/whisper` 작업 폴더 + git clone
3. conda `whisper` 환경 (Python 3.10)
4. PyTorch 2.5.1 + CUDA 12.1
5. Turing 아키텍처 제약: fp16 사용, bf16/flash-attn2 미지원

---

## Phase 6 — VOTE400 다운로드 + E0' 베이스라인 (5월 27일)

### 데이터 동기화
- rclone 설치 (sudo 없이 conda로), OAuth 헤드리스 셋업
- Drive → 학교 서버: **1시간 37분** (63GB, 100Mbps)
- 압축 풀이: `cat VOTE400.tar.* | tar -x`
- macOS 메타파일 정리
- 최종: WAV 112,984 = TXT 112,984 (1:1)

### E0' 베이스라인 측정

**실험**: VOTE400 낭독체 16개 그룹 × 200개 = 3,200 발화

| 지표 | 값 |
|---|---:|
| 전체 CER | **20.38%** |
| 전체 WER | 64.63% |
| 정제 후 CER | 21.08% (정제 효과 거의 없음) |

**그룹별 CER**:

| 그룹 | CER | 비고 |
|---|---:|---|
| SE3 | 17.91% | 가장 쉬움 |
| (...12 groups...) | 18~22% | 평균적 |
| GW2 | 24.88% | 어려움 |
| **JN2** | **34.94%** | **압도적 outlier** |

### 발견 — JN2 outlier
다른 그룹 평균 ~19% 대비 JN2만 35%. 전남(JN) 그룹답게 방언 영향. "골다공증" → "콜드라운측" 등.

### 산출
- [`code/vote400_loader.py`](../../../code/vote400_loader.py)
- [`code/E0prime_VOTE400_baseline.py`](../../../code/E0prime_VOTE400_baseline.py)
- [`results/E0prime_vote400_read_sampled.csv`](../../../results/E0prime_vote400_read_sampled.csv)
- [`results/vote400_read_index.csv`](../../../results/vote400_read_index.csv) (낭독체 112,984)
- [`results/vote400_dialog_index.csv`](../../../results/vote400_dialog_index.csv) (대화체 1,170)

---

## Phase 7 — E1': LoRA 파인튜닝 ⭐ (6월 5일)

### 학습 설정

| 항목 | 값 |
|---|---|
| Base 모델 | `openai/whisper-large-v3` (1.55B params) |
| 기법 | LoRA (r=8, alpha=32, target=`q_proj`·`v_proj`, dropout=0.05) |
| 학습 가능 파라미터 | **3,932,160 / 1,547,422,720 (0.25%)** |
| 데이터 | VOTE400 낭독체 Train 95,110 / Val 16,704 (서브샘플 1,500) |
| Split | **그룹(region+group) 비율 유지 + 화자(pid) disjoint**, seed=42 |
| 배치 크기 | per_device 8 × accum 2 × 2 GPU = **유효 32** |
| Epoch / Step | 2 epoch / 5,946 step |
| 정밀도 | **fp16** (Turing 아키텍처라 bf16 불가) |
| 디코딩 (최종 평가) | language=ko, **num_beams=5**, max_new_tokens=128 |
| 하드웨어 | RTX 8000 48GB × 2 (**2-GPU DDP**) |
| 학습 시간 | **약 6.9시간** |

### 최종 결과 ⭐⭐⭐

| 모델 | CER | WER |
|---|---:|---:|
| E0' (zero-shot baseline) | 20.38% | — |
| **E1' (LoRA fine-tuned)** | **10.25%** | 35.05% |
| **개선** | **−10.13%p (상대 49.7% ↓)** | |

→ **PPT 목표 "30% 상대 개선"을 훌쩍 초과한 약 50% 상대 개선 달성**.

### 그룹별 CER (E1', val 1,500 발화)

| 그룹 | 발화수 | CER | | 그룹 | 발화수 | CER |
|---|---:|---:|---|---|---:|---:|
| DG1 | 108 | 9.88% | | GW2 | 128 | 9.66% |
| DG2 | 36 | 9.12% | | GW3 | 103 | 11.43% |
| DG3 | 132 | 11.85% | | JN1 | 67 | 9.58% |
| DG4 | 102 | 11.28% | | JN2 | 128 | **12.40%** |
| GN1 | 75 | 9.45% | | JN3 | 137 | 12.11% |
| GN2 | 109 | 9.15% | | SE1 | 18 | 13.48% |
| GN3 | 137 | 12.19% | | SE2 | 84 | 9.63% |
| GW1 | 21 | 13.72% | | SE3 | 115 | 9.70% |

그룹 간 편차 9.1~13.7% (소표본 GW1·SE1 제외 시 9.1~12.4%) — 비교적 균일.

### 핵심 발견 — JN2 outlier 메워짐

| 그룹 | E0' CER | E1' CER | 절대 개선 | 상대 개선 |
|---|---:|---:|---:|---:|
| **JN2 (가장 어려웠음)** | **34.94%** | **12.40%** | **−22.54%p** | **65%** |
| 평균 (16 그룹) | ~19% | ~10% | ~9%p | ~50% |

→ **어려운 그룹이 가장 큰 절대 개선**. LoRA가 "어려운 화자 그룹"에 특화 적응한다는 증거.
→ 그룹 간 격차도 줄어듦: 17.91~34.94% (1.95배) → 9.1~13.7% (1.51배).

### 학습 추세
- 학습 loss: 11.96 (초기 warmup) → 0.034 (안정 수렴)
- 학습 중 eval CER은 greedy decoding의 repetition 폭주로 들쭉날쭉 (10~150% 범위)
- 최종 beam5 평가는 안정적 (10.25%)

### 산출 ⭐
- [`code/E1prime_VOTE400_LoRA.py`](../../../code/E1prime_VOTE400_LoRA.py): 학습 스크립트
- [`code/E1prime_VOTE400_predict.py`](../../../code/E1prime_VOTE400_predict.py): adapter_final 재추론
- [`results/E1prime_metrics.csv`](../../../results/E1prime_metrics.csv): 학습 loss/eval 곡선
- [`results/E1prime_train_log.txt`](../../../results/E1prime_train_log.txt): 전체 학습 로그 (500KB)
- [`results/E1prime_vote400_read_pred.csv`](../../../results/E1prime_vote400_read_pred.csv): val 1,500 예측
- [`results/E1prime_results.md`](../../../results/E1prime_results.md): 결과 요약 리포트
- [`whisper-lora-vote400/adapter_final/`](../../../whisper-lora-vote400/adapter_final/): **LoRA 어댑터 (19MB)** ⭐

---

## Phase 8 — 결과 분석 + 최종 보고서 (진행 중)

### 완료
- ✅ E1' 결과 분석 + 그룹별 비교
- ✅ JN2 outlier 메워짐 정량 확인
- ✅ 본 제출 패키지 (5개 문서) 작성

### 남은 작업
- [ ] (선택) 전체 11.2만개 inference로 JN2 outlier 신뢰도 확인
- [ ] (선택) paired t-test 통계적 유의성 검정
- [ ] (선택) JN2 정성 분석 — 어떤 발음 유형이 LoRA로 개선됐는지
- [ ] 최종 발표 PPT 자료 작성

---

## Git 활동 요약

### 커밋 이력

| Date | Hash | Message | 변경 |
|---|---|---|---|
| (초기) | `5389b21` | first commit | (초기) |
| 2026-05-19 | `f0e4a1c` | Initial commit: E0 baseline + E2 H2 ablation | 26 files, +9,775 |
| 2026-05-27 | `5c254b4` | Add VOTE400 loader + E0' baseline script | 2 files, +366 |
| 2026-05-27 | `7e742ca` | Add VOTE400 E0' baseline results (3,200 sampled, CER 20.38%) | 1 file |
| 2026-05-27 | `8878ef5` | Add VOTE400 metadata indexes (read 112K + dialog 1.1K utts) | 2 files |
| 2026-06-05 | `e521cd5` | **Add E1' LoRA results (VOTE400 낭독체, CER 20.38%→10.25%)** ⭐ | 6 files |
| 2026-06-05 | `e83dd1c` | **Add E1' LoRA adapter weights for inference reproduction** ⭐ | 6 files |

### 산출물 규모
- 코드: 12개 Python 스크립트 + 노트북 2개
- 결과: 8개 CSV + 4개 분석 리포트
- 모델: 1개 LoRA 어댑터 (19MB)
- 보고서: 본 제출 패키지 (5개 .md)

---

## 일정 대비 진행도

| Phase | 계획 | 실제 | 상태 |
|---|---|---|---|
| 1. 연구 설계 | 4월 | 4월 초 | ✅ |
| 2. AI허브 E0 | 5월 초 | 5월 초 | ✅ |
| 3. H2 ablation | 5월 중 | 5월 중 | ✅ |
| 4. 중간 발표 | 5월 중 | 5월 21일 | ✅ |
| 5. 학교 서버 셋업 | 5월 말 | 5월 26~27일 | ✅ |
| 6. VOTE400 E0' | 5월 말 | 5월 27일 | ✅ |
| 7. **E1' LoRA 학습** | 6월 초 | **6월 5일** | ✅ ⭐ |
| 8. 최종 발표 | 6월 중 | (예정) | 🔄 |

전체 일정 대비 **약 1주 빠른 진행**. **8주 일정의 7개 phase 완료**.
