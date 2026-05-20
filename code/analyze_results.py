import csv
from collections import defaultdict
from pathlib import Path

# 스크립트(code/) 기준 상대 경로 — 폴더 이름 바뀌어도 안 깨짐
CSV_PATH = Path(__file__).resolve().parent.parent / 'results' / 'E0_baseline_results.csv'

with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f))

print(f'Total utterances: {len(reader)}')

# Speaker stats
speakers = defaultdict(list)
for r in reader:
    speakers[r['speaker_id']].append(r)

print(f'Total speakers: {len(speakers)}')
print()

for sid in sorted(speakers.keys()):
    items = speakers[sid]
    info = items[0]
    total_dur = sum(float(r['duration']) for r in items)
    g = info["gender"]
    a = info["age"]
    reg = info["region"]
    print(f'  [{g}/{a}/{reg}] {sid}: {len(items)} utts, {total_dur/60:.1f} min')

# CER calculation
try:
    from jiwer import cer, wer
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'jiwer', '-q'])
    from jiwer import cer, wer

refs = [r['transcript'].strip() for r in reader if r['transcript'].strip()]
hyps = [r['prediction'].strip() if r['prediction'].strip() else ' ' for r in reader if r['transcript'].strip()]

total_cer = cer(refs, hyps)
total_wer = wer(refs, hyps)

print()
print('=' * 50)
print('E0 Baseline Results')
print('=' * 50)
print(f'  CER: {total_cer*100:.2f}%')
print(f'  WER: {total_wer*100:.2f}%')
print(f'  Utterances: {len(refs)}')
print('=' * 50)

# Per-speaker CER
print()
print('Per-speaker CER:')
for sid in sorted(speakers.keys()):
    items = speakers[sid]
    info = items[0]
    g = info["gender"]
    a = info["age"]
    reg = info["region"]
    sr = [r['transcript'].strip() for r in items if r['transcript'].strip()]
    sh = [r['prediction'].strip() if r['prediction'].strip() else ' ' for r in items if r['transcript'].strip()]
    if sr:
        sc = cer(sr, sh)
        print(f'  {sid} [{g}/{a}/{reg}]: CER={sc*100:.2f}% ({len(sr)} utts)')

# Error type breakdown
from jiwer import process_words
td, ti, ts, tr = 0, 0, 0, 0
for ref, hyp in zip(refs, hyps):
    rc = list(ref.replace(' ', ''))
    hc = list(hyp.replace(' ', ''))
    try:
        out = process_words(' '.join(rc), ' '.join(hc))
        td += out.deletions
        ti += out.insertions
        ts += out.substitutions
        tr += len(rc)
    except Exception:
        tr += len(rc)

n = max(tr, 1)
print()
print('Error Type Breakdown (char-level):')
print(f'  Total ref chars: {tr}')
print(f'  Deletion:     {td} ({td/n*100:.1f}%)')
print(f'  Insertion:    {ti} ({ti/n*100:.1f}%)')
print(f'  Substitution: {ts} ({ts/n*100:.1f}%)')
