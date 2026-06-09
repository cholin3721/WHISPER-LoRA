# 1. 진행 중 발생한 이슈 리포트 (PR Report)

**프로젝트**: WHISPER-LoRA (한국어 노인 음성 인식)
**작성자**: 202444085 김철중
**기간**: 2026년 4월 ~ 6월

본 리포트는 프로젝트 진행 중 마주친 실질적 문제들과 그 해결 과정을 문제·원인·해결·학습 4단계로 정리한다. 총 **16건의 이슈**를 다룬다.

---

## 이슈 분류 요약

| 분류 | 건수 | 주요 사례 |
|---|---:|---|
| 데이터/라벨 | 4 | AI허브 태그 정제, VOTE400 메타파일 |
| 인프라/환경 | 5 | CUDA 매칭, conda 활성화, GitHub PAT |
| 가설 검증 | 2 | H2 반박, 베이스라인 의외성 |
| 운영/도구 | 3 | tmux, SSH 끊김, tar split |
| **LoRA 학습** | **2** | **eval CER 변동, DDP 종료 경고** |

---

## 이슈 #1 — AI허브 transcription에 라벨 태그 발견

**문제**: E0 베이스라인 측정 시 CER 6.85% / Deletion 4.25%로 측정됨. PPT 가설(노인 음성 Deletion 압도)이 데이터로 확인되는 듯했음.

**원인**: 정성 분석 결과, AI허브 transcription에 `(NO:)`, `(SP:)`, `(FP:)`, `(SN:)` 같은 4종 라벨 태그가 12% 발화에 포함되어 있음. Whisper는 깨끗한 본문만 출력하므로 단순 비교 시 태그 마커가 모두 Deletion 오류로 잡힘.

```
정답: (NO:)언니 엄마는(SP: 노프로블람이라매요 )이랬더니
예측: 언니 엄마는 노 프로블럼 이래메요 이랬더니
→ Whisper는 정확하지만 태그가 "삭제됨"으로 계산
```

**해결**: 정규식으로 태그 정제 로직 작성.

```python
TAG_RE = re.compile(r'\(([A-Z]+):([^)]*)\)')
def clean_text(t):
    out = TAG_RE.sub(lambda m: ' ' + m.group(2).strip() + ' ', t)
    return re.sub(r'\s+', ' ', out).strip()
```

정제 후 결과:
- CER 6.85% → **4.25%** (-2.60%p)
- Deletion 4.25% → **0.69%** (-3.56%p)

→ **측정된 Deletion 오류의 85%가 실제로는 모델 실수가 아니라 라벨 형식 차이**.

**학습**: 측정 결과가 가설과 일치한다고 곧바로 받아들이지 말 것. 데이터 형식과 모델 출력 형식의 차이를 먼저 점검해야 함. 데이터셋의 어노테이션 컨벤션은 측정 메트릭에 큰 영향을 미친다.

---

## 이슈 #2 — H2 가설 (외부 VAD 재튜닝) 데이터로 반박

**문제**: PPT 원안의 H2 가설은 "노인 음성의 발화 내 긴 침묵을 VAD가 발화 종료로 오인 → Deletion 오류 발생" 이었음. 외부 VAD 임계값을 2초 침묵 허용으로 재조정하면 Deletion이 줄어들 것으로 예상.

**실험**: AI허브 10초+ 발화 281개에 두 조건 비교.
- A: 기본 Whisper
- B: Silero VAD (`min_silence_duration_ms=2000`) 사전분할

**결과 — 가설 정반대**:

| 조건 | CER | Del% |
|---|---:|---:|
| A (Baseline) | 6.23% | 2.37% |
| B (Elderly VAD) | 7.03% | 3.11% |
| Δ | **+0.80%p** | **+0.74%p** |

**원인 분석 (정성)**:
VAD가 노인 발화 특유의 **반복 발화(disfluency)**를 발화 경계로 오인하여 앞부분을 잘라먹음.

| 정답 | A (Baseline) | B (VAD) |
|---|---|---|
| 그래서 병원 **그래서 병원에** 가서… | 정답 그대로 | "그래서 병원에 가서…" |
| 그 방법 중 **그 방법 중에서** 도움을… | 정답 그대로 | "그 방법 중에서…" |

발화별 추세: 17개 개선 / 229개 동일 / 35개 악화 — **악화 vs 개선 2:1**.

**해결**: H2 가설 폐기. PPT의 narrative를 "가설을 검증한 결과 반박되었고, Whisper-large-v3의 내부 디코더가 이미 발화 내 침묵을 강건하게 처리함이 데이터로 확인됨. 외부 전처리보다 모델 내부 적응(LoRA)이 필요함"으로 전환.

**학습**: 가설은 데이터로 반증 가능해야 과학이다 (Popper). 가설이 빗나간 방식이 흥미로우면, 그 자체가 강력한 연구 narrative가 된다.

---

## 이슈 #3 — Google Drive 부분 업로드 (zip 162MB만 받힘)

**문제**: New_Sample.zip(344MB)을 Google Drive에 업로드 후 Colab에서 사용하려는데 `BadZipFile: File is not a zip file` 에러.

**진단 명령**:
```bash
!ls -la /content/drive/MyDrive/New_Sample.zip
!file /content/drive/MyDrive/New_Sample.zip
!head -c 4 /content/drive/MyDrive/New_Sample.zip | xxd
```

결과:
- 크기: **162MB** (원본 344MB의 47%)
- 매직 바이트: `PK\x03\x04` (정상 zip 시그니처)
- 파일 타입: "Zip archive data"

즉 zip 시작부는 정상이지만 끝부분(end-of-central-directory record)이 잘렸음.

**원인**: Drive 웹 업로드가 큰 파일에서 끊김.

**해결**: Colab Files 패널에 직접 드래그앤드롭 업로드. Drive 우회.

**학습**: 큰 파일은 Drive 웹 UI보다 데스크탑 동기화 앱이 안정적. 항상 다운로드 후 매직 바이트나 크기 검증.

---

## 이슈 #4 — VOTE400 zip 폴더 구조 차이 (`New_Sample/` 최상위 없음)

**문제**: 사용자별로 압축한 zip의 폴더 구조가 다름. 어떤 zip은 `New_Sample/` 최상위 폴더로 묶고, 다른 zip은 그게 없이 바로 `원천데이터/`, `라벨링데이터/`가 있음. 로더 코드가 깨짐.

**해결**: 자동 감지 로직 추가.

```python
_nested = os.path.join(EXTRACT_DIR, "New_Sample")
DATA_ROOT = _nested if os.path.isdir(_nested) else EXTRACT_DIR
```

**학습**: 외부에서 받은 데이터셋의 폴더 구조를 코드에 박지 말 것. defensive coding으로 변형 가능성 흡수.

---

## 이슈 #5 — VOTE400 다운로드 도중 SSH 끊김 (34GB 손실)

**문제**: rclone으로 학교 서버에서 Google Drive의 VOTE400(63GB)을 다운로드. 1시간 34분 후 SSH 연결 끊김. `rclone copy` 프로세스가 SSH 세션에 attached 되어 있어 같이 죽음. 34GB 진행분 손실 (rclone이 partial 파일 mid-resume 미지원).

**원인**: tmux 없이 foreground로 rclone 실행. SSH timeout 또는 네트워크 끊김 시 SIGHUP으로 프로세스 종료.

**해결**: tmux 안에서 재실행.

```bash
tmux new -s download
rclone copy "gdrive:..." . -P --transfers=4
# Ctrl+B → d 로 detach. SSH 끊겨도 tmux 세션은 학교 서버에서 계속 동작.
```

재시도 후 1시간 37분 만에 완료.

**학습**: 30분 이상 걸리는 모든 명령은 무조건 tmux 또는 nohup. SSH 자체가 신뢰할 수 없는 채널.

---

## 이슈 #6 — macOS 메타데이터 파일 압축에 포함

**문제**: VOTE400 압축 풀고 파일 카운트하니 WAV 225,968개 / TXT 112,994개. 비율 2:1로 이상함.

**원인**: 데이터셋이 macOS에서 압축돼 각 wav마다 `._wav파일이름` 형식의 메타파일이 함께 묶임. AppleDouble 인코딩.

**해결**: 일괄 삭제.

```bash
find /storage/cholin2/whisper/data/VOTE400 -name "._*" -delete
find /storage/cholin2/whisper/data/VOTE400 -name ".DS_Store" -delete
```

정리 후 WAV 112,984 = TXT 112,984 (1:1 매칭 확인).

**학습**: 데이터셋 받으면 메타파일 정리부터. `find -name "._*"` 또는 `find -name ".DS_Store"`로 macOS 잡파일 처리.

---

## 이슈 #7 — PyTorch + CUDA 버전 매칭

**문제**: conda로 PyTorch 설치 후 `torch.cuda.is_available()`이 False 반환.

**원인**: PyTorch 빌드와 시스템 CUDA 드라이버 버전 불일치.

**해결**: 학교 서버 드라이버 확인 후 호환되는 PyTorch 설치.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

`pytorch-cuda=12.1` 명시. CUDA 드라이버 12.9는 PyTorch 12.1 빌드와 backward compatible.

**학습**: GPU 환경 셋업의 가장 흔한 함정. RTX 8000은 Turing 아키텍처라 **bf16/flash-attention 2 미지원** → `fp16` 사용 필수.

---

## 이슈 #8 — conda 환경 활성화 누락

**문제**: tmux 세션에서 학습 스크립트 실행 시 `ModuleNotFoundError: No module named 'transformers'`.

**원인**: tmux 새 창은 새 shell이라 conda 활성화 상태가 풀림.

**해결**: 매 tmux 세션마다 `conda activate whisper` 먼저 실행.

**학습**: shell 환경 격리 이해. tmux 새 창, ssh 새 연결, cron job 등은 모두 새 shell.

---

## 이슈 #9 — GitHub Password Authentication 차단

**문제**: 학교 서버에서 `git push` 시 `Password authentication is not supported for Git operations` 에러.

**원인**: GitHub가 2021년 8월부터 HTTPS 비밀번호 인증 차단.

**해결 (SSH key 방식)**:

```bash
ssh-keygen -t ed25519 -C "cholin3721@github" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub   # GitHub Settings에 등록
git remote set-url origin git@github.com:cholin3721/WHISPER-LoRA.git
```

**학습**: 현대 Git 인증은 SSH key 또는 token 기반. 한 번 셋업하면 영구적이라 SSH key 추천.

---

## 이슈 #10 — VOTE400 폴더 구조의 미지의 그룹 번호

**문제**: VOTE400 낭독체 폴더가 `DG1, DG2, DG3, DG4`, `GN1, GN2, GN3` 같이 지역코드 + 숫자 형식. README가 "지역 구분 폴더"라고만 적고 숫자의 의미를 설명 안 함. PPT의 "정상/구음장애/80+" subgroup과 매핑되는지 불명.

**해결**: 폴더별 PID 추출 + 교집합 검사로 그룹의 의미를 데이터 기반 추론.

```bash
comm -12 \
  <(ls .../DG1 | cut -d'_' -f1-2 | sort -u) \
  <(ls .../DG2 | cut -d'_' -f1-2 | sort -u) | wc -l
# 결과: 0 (공통 화자 없음)
```

결론: 폴더 간 교집합 0 → **다른 화자 그룹** (수집 차수 아님).

**해결 (실용적 우회)**: 그룹의 명시적 의미 추론 대신, 각 그룹에 zero-shot Whisper 돌려서 **CER 기반으로 어려운 그룹 자동 발견**. E0' 결과 JN2 (전남 그룹 2)가 CER 34.94%로 outlier 발견.

**학습**: 데이터셋의 메타데이터가 부족할 때, **데이터로부터 메타데이터를 유도**하는 접근. CER이 곧 grouping의 ground truth가 됨.

---

## 이슈 #11 — tmux scrollback 잘림

**문제**: E0' 베이스라인 결과(전체 CER, 그룹별 16개 표)가 tmux에 출력됐는데, 결과 일부가 위로 스크롤해도 안 보임.

**해결 (단기)**: CSV에서 직접 재계산.

**해결 (장기)**: tee로 화면+파일 동시 저장 + tmux 설정 변경.

```bash
python -u code/script.py ... 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M).log
echo "set -g history-limit 100000" >> ~/.tmux.conf
```

**학습**: 장기 실행 결과는 화면 출력만 믿지 말고 파일로 보존. `tee`는 기본 도구화.

---

## 이슈 #12 — tar split 압축 풀이 (60GB)

**문제**: VOTE400이 `VOTE400.tar.aa` ~ `VOTE400.tar.ag` 7개 분할(총 63GB)로 배포됨.

**해결**: 합치면서 동시에 추출 (디스크 효율적).

```bash
cat VOTE400.tar.* | tar -x
```

디스크 사용: 합치고 풀기(63+90GB) → streaming 추출(90GB만).

**학습**: Unix 파이프의 효율성. 큰 데이터일수록 stream 처리가 디스크/속도 면에서 큰 차이.

---

## 이슈 #13 — Python sort 시 dict 비교 TypeError

**문제**: E2 실험 마지막 단계에서 `TypeError: '<' not supported between instances of 'dict' and 'dict'`.

**원인**: 튜플 안에 dict가 들어있어 sort의 lexicographic 비교가 dict로 fall-through.

**해결**: `key=` 명시.
```python
diffs.sort(key=lambda x: x[0], reverse=True)
```

**학습**: Python sort에서 비교 불가 객체가 튜플에 있으면 항상 `key=` 지정.

---

## 이슈 #14 — 베이스라인 측정값과 PPT 예측의 큰 불일치

**문제**: PPT 원안은 노인 음성 STT의 CER을 20~30%로 추정 (영어권 연구 인용). 실측은 AI허브 4.25% / VOTE400 20.38% / VOTE400 JN2 34.94%로 5배 격차.

**원인 (분석)**:

| 요인 | AI허브 (4.25%) | VOTE400 (20.38%) |
|---|---|---|
| 화자 수 | 5명 | 약 100명+ |
| 지역 다양성 | 수도권 위주 | 5개 지역 |
| 데이터 의도 | 챗봇 학습용 | 노인 음성 연구용 |
| 라벨 형식 영향 | 큼 (-2.60%p 정제) | 거의 없음 (+0.70%p) |
| 방언 영향 | 거의 없음 | JN2 등 두드러짐 |

**해결**: PPT narrative 재구성. AI허브는 "표준 노인 슬라이스", VOTE400은 "진짜 노인 인구 대표 샘플". **H1(LoRA) 검증의 무대를 VOTE400으로 명확히 이동**.

**학습**: 가설의 절댓값은 비주관적 측정으로 검증되어야 함. 두 데이터셋 간 격차 자체가 메소드론적 발견.

---

## 이슈 #15 — LoRA 학습 중 eval CER이 들쭉날쭉 (greedy decoding의 함정)

**문제**: E1' LoRA 학습 중 step별 eval CER이 매우 불안정. 예시:

| epoch | eval_loss | eval_cer | 비고 |
|---:|---:|---:|---|
| 0.50 | 0.047 | **150.9%** | 폭주 |
| 0.67 | 0.042 | 18.6% | 안정 |
| 0.84 | 0.038 | 10.97% | 안정 |
| 1.01 | 0.038 | **74.9%** | 폭주 |
| 1.18 | 0.037 | 16.9% | 안정 |
| 1.35 | 0.036 | **63.1%** | 폭주 |
| ... | ... | ... | ... |

eval_loss는 일관되게 0.034 부근으로 수렴하는데 eval_cer만 들쭉날쭉.

**원인**: 학습 시간 절약을 위해 학습 중 eval은 **greedy decoding (num_beams=1)** 사용. fp16 + greedy + LoRA 학습 초기엔 모델이 가끔 **repetition 폭주** (같은 토큰 무한 반복)를 일으킴. eval_loss는 teacher-forcing 기반이라 정상인데, generate-then-decode 기반 CER만 폭주.

**해결**:
- 학습 중 eval은 빠른 모니터링용으로 두고, **최종 평가만 num_beams=5** 로 재실행
- `E1prime_VOTE400_predict.py` 별도 작성: 학습 끝난 adapter_final 체크포인트를 beam5로 추론
- 최종 결과: **CER 10.25%** (학습 중 잘 나오던 eval과 일치)

```python
# 학습 시 (빠른 모니터링)
training_args.predict_with_generate = True
training_args.generation_num_beams = 1   # 빠른 greedy

# 별도 평가 스크립트
model.generate(..., num_beams=5)   # 최종 정확 평가
```

**학습**: 학습 중 metric과 최종 평가 metric을 분리 관리해야 함. greedy decoding은 빠르지만 노이즈가 크다. 학습 중 metric의 절댓값보다 추세를 봐야 하며, 최종 결론은 항상 beam search로 재평가.

---

## 이슈 #16 — DDP 종료 시 NCCL 경고

**문제**: 2-GPU DDP 학습 마치고 종료 시 다음 경고:

```
[rank0]:[W605 19:14:05.794157683 ProcessGroupNCCL.cpp:1250] Warning:
WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL.
On normal program exit, the application should call destroy_process_group
to ensure that any pending NCCL operations have finished in this process.
```

**원인**: PyTorch 2.4+에서 추가된 경고. HuggingFace Trainer가 학습 끝나고 NCCL process group을 명시적으로 `destroy_process_group()` 호출 안 함. 학습 결과 자체엔 영향 없음 (경고 수준).

**해결**: 학습 결과에 영향 없으므로 무시 가능. 향후 cleanup 함수 추가 고려.

```python
import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()
```

**학습**: 경고와 에러를 구분할 것. 경고는 향후 개선 거리이지 즉시 대응 X. 학습 metric이 정상이면 경고는 logbook에만 남기면 됨.

---

## 종합 — 이번 프로젝트에서 배운 메타 교훈

1. **데이터를 먼저 보라**: 측정 메트릭은 데이터 형식에 민감. AI허브 태그 정제로 CER이 6.85% → 4.25%로 떨어진 사례가 결정적.
2. **가설 반증은 손실이 아니다**: H2가 데이터로 반박된 것이 PPT narrative를 더 강하게 만듦.
3. **가설 지지도 정직하게**: H1이 49.7% 상대 개선으로 PPT 목표(30%)를 초과한 것도 데이터로 입증됨. 베이스라인을 올바른 데이터셋(VOTE400)에서 잡았기에 가능.
4. **인프라가 절반**: SSH·conda·rclone·tmux 같은 도구 셋업 시간이 모델 학습 시간만큼 든다.
5. **로그를 남겨라**: tmux 스크롤이 잘리면 결과 다시 못 봄. `tee`는 기본 습관.
6. **학습 metric은 추세, 평가 metric은 최종 결론**: 학습 중 greedy CER이 들쭉날쭉해도 좌절 X. 최종은 beam search로.
7. **데이터셋이 같은 라벨이라고 같은 분포가 아니다**: "노인 음성"이라는 동일 라벨에서 5배 격차가 가능.
