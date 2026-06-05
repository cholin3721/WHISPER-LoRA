# E1′ — Whisper-large-v3 + LoRA Fine-tuning on VOTE400 (낭독체)

## 최종 결과 (val 1,500, beam5)

| 모델 | CER | WER |
|---|---:|---:|
| E0′ (zero-shot baseline) | 20.38% | — |
| **E1′ (LoRA fine-tuned)** | **10.25%** | 35.05% |
| **개선** | **−10.13%p (상대 49.7% ↓)** | |

PPT 목표("30% 상대 개선")를 상회하는 약 50% 상대 개선 달성.
(adapter_final 재추론 기준 CER 10.81% — 학습 중 best 평가 10.25%와 fp16 추론 미세차)

## 그룹별 CER (E1′ 예측, val 1,500)

| 그룹 | 발화수 | CER | | 그룹 | 발화수 | CER |
|---|---:|---:|---|---|---:|---:|
| DG1 | 108 | 9.88% | | GW2 | 128 | 9.66% |
| DG2 | 36 | 9.12% | | GW3 | 103 | 11.43% |
| DG3 | 132 | 11.85% | | JN1 | 67 | 9.58% |
| DG4 | 102 | 11.28% | | JN2 | 128 | 12.40% |
| GN1 | 75 | 9.45% | | JN3 | 137 | 12.11% |
| GN2 | 109 | 9.15% | | SE1 | 18 | 13.48% |
| GN3 | 137 | 12.19% | | SE2 | 84 | 9.63% |
| GW1 | 21 | 13.72% | | SE3 | 115 | 9.70% |

그룹 간 편차 9.1~13.7% (소표본 GW1·SE1 제외 시 9.1~12.4%) — 비교적 균일.

## 학습 설정

| 항목 | 값 |
|---|---|
| Base model | openai/whisper-large-v3 |
| 기법 | LoRA (r=8, alpha=32, target=q_proj·v_proj, dropout=0.05) |
| 학습 파라미터 | 3,932,160 / 1,547,422,720 (0.25%) |
| 데이터 | VOTE400 낭독체, Train 95,110 / Val(서브샘플) 1,500 |
| split | 그룹(region+group) 비율 유지, 화자(pid) disjoint, seed=42 |
| 배치 | per_device 8 × accum 2 × 2 GPU = 유효 32 |
| epoch / step | 2 epoch / 5,946 step |
| 정밀도 | fp16 (autocast) |
| 디코딩 | language=ko, num_beams=5(최종)/1(학습중), max_new_tokens=128 |
| 하드웨어 | RTX 8000 48GB × 2 (DDP), 약 6.9시간 |

## loss / eval 추세

- 학습 loss: 11.93(초기 warmup) → 0.12 (안정 수렴)
- 자세한 추세: `E1prime_metrics.csv` 참조
- 학습 중 eval(greedy)은 일부 샘플 repetition으로 들쭉날쭉했으나, 최종 beam5 평가는 안정적(10.25%).

## 산출물

- `E1prime_vote400_read_pred.csv` — val 1,500 예측 (E0′와 동일 포맷, 그룹별 비교용)
- `E0prime_vote400_read_sampled.csv` — E0′ 베이스라인 예측 (비교 기준)
- `E1prime_metrics.csv` — 학습 loss/eval 곡선
- `E1prime_train_log.txt` — 전체 학습 로그
- `../whisper-lora-vote400/adapter_final/` — LoRA 어댑터 (19MB)
