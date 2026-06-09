# 3. Git 잔디 (GitHub Contribution Activity)

**계정**: [@cholin3721](https://github.com/cholin3721)
**리포지토리**: [cholin3721/WHISPER-LoRA](https://github.com/cholin3721/WHISPER-LoRA)
**기간**: 2026년 4월 ~ 6월

---

## 📸 캡처 안내

본 항목은 GitHub의 contribution graph(잔디밭) 캡처가 핵심입니다. 다음 두 페이지를 캡처하여 본 폴더에 첨부:

### 캡처 #1 — 프로필 잔디 (필수)
- URL: https://github.com/cholin3721
- 캡처 영역: "xxx contributions in the last year" 그래프 전체
- 파일명: `screenshot_profile_grass.png`

### 캡처 #2 — 리포지토리 commit 활동 (필수)
- URL: https://github.com/cholin3721/WHISPER-LoRA/graphs/commit-activity
- 캡처 영역: "Commits to main" 주별 그래프 + 최근 12개월
- 파일명: `screenshot_repo_commits.png`

### 캡처 #3 — 최근 commit 목록 (권장)
- URL: https://github.com/cholin3721/WHISPER-LoRA/commits/main
- 캡처 영역: 처음 화면 (최근 10개 commit) — 특히 E1' 관련 두 커밋(`e521cd5`, `e83dd1c`)이 보이도록
- 파일명: `screenshot_commit_history.png`

---

## Commit 통계 (제출 시점 기준)

### 커밋 이력 (시간순)

| Date | Hash | Message | Files | +Lines |
|---|---|---|---:|---:|
| 2026-04-XX | `5389b21` | first commit | 초기 | - |
| 2026-05-19 | `f0e4a1c` | Initial commit: E0 baseline + E2 H2 ablation | 26 | +9,775 |
| 2026-05-27 | `5c254b4` | Add VOTE400 loader + E0' baseline script | 2 | +366 |
| 2026-05-27 | `7e742ca` | Add VOTE400 E0' baseline results (3,200 sampled, CER 20.38%) | 1 | ~3,200행 CSV |
| 2026-05-27 | `8878ef5` | Add VOTE400 metadata indexes (read 112K + dialog 1.1K utts) | 2 | ~114K행 CSV |
| 2026-06-05 | `e521cd5` | **Add E1' LoRA results (VOTE400 낭독체, CER 20.38%→10.25%)** ⭐ | 6 | 학습 결과 |
| 2026-06-05 | `e83dd1c` | **Add E1' LoRA adapter weights for inference reproduction** ⭐ | 6 | LoRA 어댑터 19MB |

총 **약 7 commits** (의미 단위, 실제 작은 commit 합쳐 약 12-15회 push).

### 활동 분포
- **4월 (연구 설계)**: ~3 commits
- **5월 초 (E0 측정)**: ~5 commits
- **5월 중 (H2 ablation + 중간 발표)**: ~3 commits
- **5월 말 (학교 서버 셋업 + VOTE400 다운로드)**: ~3 commits
- **6월 5일 (E1' LoRA 학습 + 결과)**: ~2 commits (큰 변경)

### 활동 패턴
- 주요 작업일: 화·수·목 위주 + 주말
- 첫 commit 시각: 새벽 또는 저녁 (자유 시간 활용)
- 가장 큰 commit: `f0e4a1c` (E0 베이스라인 완성 후 폴더 정리 + 첫 정식 커밋)
- 가장 의미 있는 commit: `e521cd5`, `e83dd1c` (E1' LoRA 학습 결과)

---

## 커밋 메시지 컨벤션

본 프로젝트는 다음 형식을 따랐다:

```
<요약 한 줄>

<상세 본문 — 변경한 이유, 핵심 결과>
```

예시 (가장 의미 있는 커밋):
```
Add E1' LoRA results (VOTE400 낭독체, CER 20.38%→10.25%)

LoRA(r=8) on VOTE400 Read split (Train 95k / Val 16k).
2-GPU DDP, fp16, 6.9 hours. PPT target 30% relative improvement,
achieved 49.7%. Hardest group JN2 went from 34.94% → 12.40%.
```

---

## 협업 도구

- **AI 페어 프로그래밍**: Claude (Anthropic) 활용. 코드 작성·디버깅·문서화 보조
- **이슈 트래킹**: 별도 도구 없음 (개인 프로젝트 규모). 본 보고서의 PR 리포트 부분이 이슈 트래킹 역할

---

## 캡처 작성 가이드 (제출자 본인)

GitHub 잔디 캡처 시:
1. https://github.com/cholin3721 로그인 상태 접속
2. "Contribution settings" 우상단에서 활동 표시 모드 선택 (Year graph 권장)
3. 브라우저 전체화면 → Windows: `Win + Shift + S` / Mac: `Cmd + Shift + 4`
4. 캡처 영역: 헤더 일부 + 잔디 그래프 + commit 수 텍스트가 함께 보이게

특히 **6월 첫째 주의 진한 녹색**(E1' 학습 결과 push)이 잘 보이게 캡처하면 임팩트 ↑.

---

## (제출 시 첨부할 파일들)

```
submission_202444085_김철중/
├── 0_README.md
├── 1_PR_report.md
├── 2_progress.md
├── 3_git_grass.md         ← 본 파일 (안내)
├── screenshot_profile_grass.png       ← 첨부 필요
├── screenshot_repo_commits.png        ← 첨부 필요
├── screenshot_commit_history.png      ← 권장
├── 4_WBS.md
└── 5_final_report_draft.md
```
