# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research workspace for measuring `openai/whisper-large-v3` (zero-shot) CER/WER on Korean elderly speech. There are two experimental tracks:

- **AI허브 track** (E0/E1/E2): Original 5-speaker sample, 2,930 utterances. E0 cleaned CER = **4.25%**. E1 LoRA skeleton exists but has not been run. E2 H2 VAD ablation is complete (H2 refuted).
- **VOTE400 track** (E0′/E1′): The full VOTE400 dataset (낭독체 + 대화체) has arrived at `data/VOTE400/`. E0′ zero-shot baseline on 낭독체 3,200 samples = **CER 20.38%** — this is now the primary baseline against which E1′ will be judged. E1′ LoRA script exists but has not been run.

There is no build, lint, or test suite — the deliverables are Colab notebooks and CSVs of predictions/analysis.

## Directory layout

```
WHISPER/
├── CLAUDE.md
├── code/        — all Python scripts and notebooks (E0/E1/E2 + analysis + converters)
├── results/     — CSV outputs and analysis markdown reports
├── data/
│   ├── New_Sample/     — AI허브 5-speaker sample (WAV + JSON)
│   ├── VOTE400/        — full VOTE400 dataset (낭독체 + 대화체, ~400h)
│   └── vote400_raw/    — original split tarballs (VOTE400.tar.aa–ag)
└── docs/
    ├── vote400/        — EULA + 협약서 PDFs
    └── presentation/   — PPT, PDF, 발표 대본, simulation.mp4
```

**Path convention in scripts**: analysis scripts use `Path(__file__).resolve().parent.parent` as `ROOT` so they work regardless of where the project folder is moved. Hardcoded absolute Windows paths have been removed — keep it that way.

## School GPU server — training target hardware

The school GPU server (target for E1 LoRA training) has the following spec — Whisper-large-v3 LoRA fits very comfortably:

| Component | Spec | Notes |
|---|---|---|
| CPU | Intel Xeon Gold 6240 × 2 | 36 cores / 72 threads total |
| GPU | NVIDIA Quadro RTX 8000 × 2 | **48GB VRAM each (96GB total)** — Turing arch (2018) |
| RAM | DDR4-2933 64GB × 8 = 512GB | overkill |
| SSD | 1TB | for active dataset + checkpoints |
| HDD | 12TB (7200RPM) | for archive / raw VOTE400 |
| Ethernet | 100Mbps | **only real bottleneck** — VOTE400 (~46GB) takes 1–3 hr to download |

**Architectural caveats** (RTX 8000 is Turing, not Ampere):
- **No bf16 support** → use `fp16=True`, not `bf16=True`. Already set correctly in [code/E1_LoRA_Whisper.py](code/E1_LoRA_Whisper.py).
- **No flash-attention 2** (requires Ampere+) — use default SDPA attention.
- No TF32 either. Plain fp16 mixed precision is the path.

**What this hardware enables** (vs. the original PPT assumption of "single GPU, 8 weeks, tight budget"):
- **batch size 16+ with gradient_checkpointing OFF** — original script had `per_device_train_batch_size=4, gradient_accumulation_steps=4, gradient_checkpointing=True` for T4 16GB. On RTX 8000 48GB, can change to `batch_size=16, accumulation=1, gradient_checkpointing=False` for faster training.
- **2-GPU DDP** — `torchrun --nproc_per_node=2 ...` cuts training time in half. VOTE400 50hr-subset × 3 epochs ≈ 5–8 hours single GPU → 3–4 hours DDP.
- **Parallel ablations** — GPU 0 for E1 (rank=8), GPU 1 for hyperparameter sweep (rank=16/32 or different lr) simultaneously.

**Operational note**: when [code/E1_LoRA_Whisper.py](code/E1_LoRA_Whisper.py) is moved from Colab to this server, the Drive-mount and zip-extract cells (Cell 2) get replaced with local paths, and the Training args in Cell 10 should be bumped per the numbers above.

## Current measured E0 baseline — read this carefully

The AI허브 sample produces **two different baseline numbers depending on whether label tags are stripped**. Both are correct; they answer different questions.

| Metric | Raw CER | Cleaned CER | Notes |
|---|---:|---:|---|
| Overall | 6.85% | **4.25%** | -2.60 %p from tag cleanup |
| Deletion (char%) | 4.25% | **0.69%** | **~85% of measured Del was actually label-format mismatch**, not model error |
| Substitution | 1.70% | 1.74% | unchanged (real model errors) |
| Insertion | 0.15% | 0.17% | unchanged |

**Use 4.25% (cleaned) as the headline E0 baseline going forward.** The 6.85% number in older slides / older `analyze_results.py` output is unfair to Whisper — it counts AI허브 labeling tags like `(NO:)`, `(SP:)`, `(FP:)`, `(SN:)` that Whisper doesn't (and shouldn't) emit.

**Per-speaker CER, cleaned** (sorted easiest → hardest):

  | speaker_id | gender | age | region | utts | CER (raw) | CER (cleaned) |
  |---|---|---|---|---|---:|---:|
  | 1526682663 | F | 63 | 수도권 | 900 | 3.25% | 3.08% |
  | 1520916170 | M | 75 | 수도권 | 490 | 7.62% | **4.16%** |
  | 1531552733 | F | 69 | 수도권 | 870 | 8.77% | **4.78%** |
  | 1527825984 | F | 71 | 충청 | 480 | 9.52% | **5.14%** |
  | 1522434093 | F | 60 | 경상 | 190 | 6.02% | 5.58% |

Inter-speaker gap collapsed from 3.0× (3.25–9.52%) to 1.8× (3.08–5.58%) after cleaning. Most of what looked like speaker difficulty was tag density, not acoustic difficulty.

## E2 H2 mini-ablation result — H2 is *refuted* on this sample

The H2 hypothesis (external VAD retuning reduces Deletion) was tested on 281 utterances ≥10s. See [results/E2_h2_vad_results.csv](results/E2_h2_vad_results.csv).

| Condition | CER | Del% | Sub% | Ins% |
|---|---:|---:|---:|---:|
| A (Baseline, E0 settings) | **6.23%** | **2.37%** | 1.61% | 0.19% |
| B (Silero VAD, min_silence=2000ms) | 7.03% | 3.11% | 1.73% | 0.25% |
| **Δ** | **+0.80%p** | **+0.74%p** | +0.12%p | +0.06%p |

Per-utterance trend: 17 improved (6%), 229 unchanged (81.5%), **35 worsened (12.5%)**. Worsening outpaces improvement ~2:1.

**Root cause (qualitative)**: Silero VAD with `min_silence_duration_ms=2000` interprets short pauses between elderly repetition (e.g. "그래서 병원 그래서 병원에…") as speech boundaries and trims the first repetition, *creating* Deletion errors that didn't exist in Condition A. Whisper-large-v3's internal pause handling is already robust; bolting external VAD on top hurts more than it helps.

**Caveat**: 279 of 281 utterances came from speaker `1531552733` (F/69/수도권). The result is effectively single-speaker. VOTE400 re-runs are needed before generalizing.

### Strategic implications

1. **This sample is too easy for E1 to demonstrate effect**. Cleaned baseline of 4.25% means the PPT's "30% relative improvement" target is 2.97% — within the noise floor of a 5-speaker eval set. Do not run E1 on AI허브 alone expecting publishable results.
2. **H2 (external VAD) is refuted on this sample**. Direction for E2 should pivot to *internal* decoder adjustments (length penalty, suppress_tokens, etc.) rather than external pre-segmentation — or this leg of the research should be dropped.
3. **The real research bet is VOTE400's hard subgroups**. If 구음장애 / 80+ groups stay near 4–5% CER even on VOTE400, the H1 narrative is in trouble and the project should pivot to something else (e.g. robustness benchmarking).

## Label tag cleaning — load-bearing regex

AI허브 transcripts embed four tag types within speech: 12% of utterances contain at least one. After cleaning, 32% of utterances change — this is not optional preprocessing; every metric and training pipeline must apply it.

| Tag | Meaning | Count | Example | Cleaning |
|---|---|---:|---|---|
| `(SP:content)` | 발음/방언 변이 | 197 | `(SP:깻잎)도 그렇고` | keep content |
| `(NO:content)` | 잡음 중첩 발화 | 139 | `(NO:생각이 많아지는)` | keep content |
| `(FP:content)` | 필러/추임새 | 42 | `(FP:뭐)` | keep content |
| `(SN:content)` | 단발 잡음 | 33 | `(SN:방식)` | keep content |

The canonical cleaning regex (duplicated across [code/analyze_cleaned.py](code/analyze_cleaned.py), [code/E1_LoRA_Whisper.py](code/E1_LoRA_Whisper.py), and [code/E2_H2_VAD_Experiment.py](code/E2_H2_VAD_Experiment.py)):

```python
TAG_RE = re.compile(r'\(([A-Z]+):([^)]*)\)')
def clean_text(t):
    out = TAG_RE.sub(lambda m: ' ' + m.group(2).strip() + ' ', t)
    return re.sub(r'\s+', ' ', out).strip()
```

**Why "keep content, not strip entirely"**: inspection of the 932 cleaned utterances shows the tag content is actual spoken text (the model predicts it back nearly verbatim in many cases). Stripping content would over-shrink references and inflate CER unfairly the other direction.

## How the Python files relate

Two groups: **inference/training** (run on GPU) vs **analysis** (run locally on CSVs). All live under [code/](code/).

**Shared modules**:
- [code/vote400_loader.py](code/vote400_loader.py) — VOTE400 data loader. Provides `load_read()` and `load_dialog()`, returning lists of dicts with `wav_path`, `transcript`, and metadata. Also runnable as a CLI to build the CSV indexes (`--out-dir results`). No TAG_RE — VOTE400 transcripts are clean text.

**Inference/Training (GPU, Colab or school server)**:

*AI허브 track:*
- [code/E0_Baseline_Whisper.py](code/E0_Baseline_Whisper.py) — zero-shot baseline. Source of truth for AI허브 inference logic. Colab `# %%` cell format.
- [code/E0_Baseline_Whisper.ipynb](code/E0_Baseline_Whisper.ipynb) — generated by **legacy** `make_notebook.py`; Drive-mounted Colab variant. Ships with empty outputs — **a template, not a record.**
- [code/make_notebook.py](code/make_notebook.py) — **E0-specific** generator for the `.ipynb`. Kept for E0 reproducibility but **new experiments should use `py_to_ipynb.py` instead**.
- [code/E1_LoRA_Whisper.py](code/E1_LoRA_Whisper.py) — LoRA training skeleton for AI허브 (PEFT, rank=8, `q_proj`/`v_proj` targets, `1527825984` held-out val). **Not yet executed.** For a VOTE400 finetune, use E1′ instead.
- [code/E2_H2_VAD_Experiment.py](code/E2_H2_VAD_Experiment.py) → [code/E2_H2_VAD_Experiment.ipynb](code/E2_H2_VAD_Experiment.ipynb) — H2 mini-ablation (Silero-VAD). **Already run; H2 refuted.** Inference-only, fits on free Colab T4 in <1 hour.

*VOTE400 track:*
- [code/E0prime_VOTE400_baseline.py](code/E0prime_VOTE400_baseline.py) — zero-shot baseline on VOTE400 낭독체. **Already run** (3,200 samples, CER 20.38% → `results/E0prime_vote400_read_sampled.csv`). Uses `vote400_loader.load_read()`. Same decoding params as E0 (`language="ko"`, `num_beams=5`, `max_new_tokens=128`). Supports `--sample-per-group N` (default 200) or `--sample-per-group 0` for full ~112K utterances.
- [code/E1prime_VOTE400_LoRA.py](code/E1prime_VOTE400_LoRA.py) — LoRA fine-tuning on VOTE400 낭독체. **Not yet executed.** Uses `vote400_loader.load_read()`. Stratified train/val split by group. Target hardware: RTX 8000, `batch_size=16`, `fp16=True`.

**Notebook converter**:
- [code/py_to_ipynb.py](code/py_to_ipynb.py) — **generic converter** from `# %%`-marked `.py` to `.ipynb`. Use for all new experiments: `python code/py_to_ipynb.py code/<source.py>`. Parses `# %% [markdown]` and `# %% --- Cell N: ... ---` markers; markdown cells strip leading `# ` per line.

**Analysis (local Windows / PowerShell, no GPU)**:
- [code/analyze_results.py](code/analyze_results.py) — simple recomputation of raw CER/WER from the CSV. Pre-cleaning numbers only.
- [code/analyze_deep.py](code/analyze_deep.py) — per-speaker × error-type × length-bin breakdown + worst-20-case extraction. Outputs `results/analysis_deep_report.md` and `results/worst_cases.csv`. This is what surfaced the label-format problem.
- [code/analyze_cleaned.py](code/analyze_cleaned.py) — applies `TAG_RE` cleaning and re-measures everything. Outputs `results/analysis_cleaned_report.md` and `results/E0_baseline_results_cleaned.csv`.

**Consistency rules**:
- **AI허브 TAG_RE / clean_text**: when the cleaning logic changes, update all of: `E0_Baseline_Whisper.py`, `make_notebook.py`, `analyze_results.py`, `analyze_deep.py`, `analyze_cleaned.py`, `E1_LoRA_Whisper.py`, `E2_H2_VAD_Experiment.py`. These are intentionally duplicated (no shared module for AI허브 yet).
- **VOTE400 clean_text**: only in `vote400_loader.py`. E0′ and E1′ both import from there — one change propagates automatically. VOTE400 has no TAG_RE (no annotation tags in transcripts).
- **Decoding params** (`language="ko"`, `task="transcribe"`, `num_beams=5`, `max_new_tokens=128`): must match across E0/E0′ and any evaluation scripts — changing these invalidates cross-run comparisons.

## Data layout and folder-name parsing

### AI허브 (New_Sample)

Paired WAV (`원천데이터/...`) and JSON (`라벨링데이터/...`) files mirroring each other folder-for-folder. Each speaker folder name encodes metadata split by `_`:

```
노인남여_노인대화07_F_1522434093_60_경상_실내
  [0]    [1]    [2]    [3]      [4]  [5]  [6]
                gender speaker  age  region indoor/outdoor
```

The loaders parse positions **2 (gender)**, **3 (speaker_id)**, **4 (age, int)**, **5 (region)**. Per-utterance JSON puts the ground-truth transcript at `발화정보.stt` and duration at `발화정보.recrdTime` (seconds, string). All audio is 16 kHz mono; loaders resample with `librosa.load(..., sr=16000)`.

**Colab zip-extract gotcha**: the AI허브 sample sometimes extracts with a top-level `New_Sample/` directory and sometimes without. E2's Colab cell auto-detects via `os.path.isdir(EXTRACT_DIR + "/New_Sample")`. Future E0/E1 notebooks should follow the same pattern.

### VOTE400 (data/VOTE400)

Two subsets with different file/folder conventions, handled by [code/vote400_loader.py](code/vote400_loader.py):

**낭독체 (VOTE400_Read)** — one sentence per file:
```
Audio/<REGION><GROUP>/PID_<ID>_<DATE>_<NO>_<REGION>.wav
Transcriptions/<REGION><GROUP>/PID_<ID>_<DATE>_<NO>_<REGION>.txt
```
Folder name splits into `region` (letters) + `group` (digit) via `GROUP_FOLDER_RE`. 16 groups (DG1, GN1–3, …).

**대화체 (VOTE400_Dialog)** — full conversation session per file:
```
Audio/<REGION>/<PID>_<GENDER>_<AGE>_<REGION>_<YEAR>-<DATE>-<TIME>-<SID>.wav
Transcriptions/<REGION>/<PID>_<GENDER>_<AGE>_<REGION>_<YEAR>-<DATE>-<TIME>-<SID>.txt
```
Filename parsed by `DIALOG_FILENAME_RE` to extract `pid`, `gender`, `age` (int).

**VOTE400 transcriptions have no tag annotations** — `clean_text` in `vote400_loader.py` is plain whitespace normalization only (no TAG_RE).

Metadata indexes generated by `vote400_loader.py --mode both` are already at `results/vote400_read_index.csv` and `results/vote400_dialog_index.csv`.

## Running things

All commands run from the project root:

**Local analysis (Windows / PowerShell)** — no GPU needed:

```powershell
python code/analyze_results.py    # raw CER summary (original, pre-cleaning)
python code/analyze_deep.py       # full breakdown: speaker × error-type × length, worst-20 cases
python code/analyze_cleaned.py    # apply tag cleaning, re-measure, write cleaned CSV + report
```

**VOTE400 metadata indexing** — no GPU needed:

```bash
python code/vote400_loader.py \
    --root data/VOTE400 --mode both --out-dir results
# writes results/vote400_read_index.csv + results/vote400_dialog_index.csv
```

**VOTE400 E0′ baseline inference** — GPU required, runs on school server:

```bash
# Quick (3,200 samples, ~30 min on RTX 8000):
python code/E0prime_VOTE400_baseline.py \
    --root data/VOTE400 --sample-per-group 200 \
    --out results/E0prime_vote400_read_sampled.csv

# Full (~112K utterances, ~10 hr on RTX 8000):
python code/E0prime_VOTE400_baseline.py \
    --root data/VOTE400 --sample-per-group 0 \
    --out results/E0prime_vote400_read_full.csv
```

**VOTE400 E1′ LoRA training** — school GPU server, 2-GPU DDP:

```bash
torchrun --nproc_per_node=2 code/E1prime_VOTE400_LoRA.py \
    --root data/VOTE400 --out-dir whisper-lora-vote400 \
    --batch-size 16 --epochs 2
```

**AI허브 E0 full inference** — Colab only (Drive-mount required):

1. Upload `data/New_Sample.zip` to Google Drive.
2. Open `code/E0_Baseline_Whisper.ipynb`, set runtime to T4 GPU.
3. Run cells top-to-bottom.

**Regenerating notebooks**:

```bash
python code/make_notebook.py                             # E0 only (legacy)
python code/py_to_ipynb.py code/E2_H2_VAD_Experiment.py # generic — any new experiment
```

## Conventions worth preserving

- **Korean output and comments are intentional** — deliverables are a Korean-language research report. Don't translate console output to English.
- **Apply `clean_text` before any CER/WER calculation** going forward. The original `analyze_results.py` numbers (6.85% etc.) are kept for historical comparison but should not be cited as the baseline in new slides or papers.
- **Decoding params are fixed** for baseline fairness: `language="ko"`, `task="transcribe"`, `num_beams=5`, `max_new_tokens=128`. Changing any of these invalidates comparison with the recorded E0 numbers.
- **Empty-hypothesis handling**: refs are filtered to non-empty; matching hyps that are empty get replaced with a single space (`" "`) so `jiwer` does not error.
- **Char-level error breakdown uses `jiwer.process_words` on space-separated characters** — a deliberate trick to get Del/Ins/Sub counts at character granularity. Not a bug.
- **UTF-8 stdout reconfiguration** at the top of analysis scripts (`sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`) is required on Windows PowerShell — without it, Korean output renders as `������`. Preserve this in new analysis scripts.
- **AI허브 E1 val speaker**: `1527825984` (F/71/충청) — the hardest cleaned speaker (5.14% CER). AI허브 E1 is deprioritized anyway; E1′ (VOTE400) already uses stratified split by group instead.
- **Script-relative paths**: use `Path(__file__).resolve().parent.parent` as `ROOT`, then `ROOT / 'results' / 'X.csv'`. Never hardcode `c:\202444085_Assemble\...` again.
