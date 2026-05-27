"""
E0' (E0 prime): VOTE400 낭독체 Zero-Shot Whisper 베이스라인

PPT 가설 검증의 핵심 — 폴더(그룹)별 CER을 측정해서
'어떤 그룹이 어려운 발화자인가'를 데이터 기반으로 발견한다.

기본 모드: 각 그룹에서 N개 발화 샘플링 (빠른 첫 측정, ~30분 on GPU)
전체 모드: 11만개 다 돌리기 (~10시간 on RTX 8000)

CLI:
  # 빠른 첫 측정 (그룹당 200개씩 = 16그룹 × 200 = 3,200개)
  python code/E0prime_VOTE400_baseline.py \\
      --root /storage/cholin2/whisper/data/VOTE400 \\
      --sample-per-group 200 \\
      --out results/E0prime_vote400_read_sampled.csv

  # 전체 측정
  python code/E0prime_VOTE400_baseline.py \\
      --root /storage/cholin2/whisper/data/VOTE400 \\
      --sample-per-group 0 \\
      --out results/E0prime_vote400_read_full.csv
"""
import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# 같은 폴더 내 loader 모듈 임포트
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vote400_loader import load_read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='VOTE400 폴더 경로')
    ap.add_argument('--sample-per-group', type=int, default=200,
                    help='폴더당 발화 수 (0 = 전체)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch-size', type=int, default=8,
                    help='Whisper 추론 배치 크기 (RTX 8000 48GB면 16+도 가능)')
    ap.add_argument('--out', default='results/E0prime_vote400_read_sampled.csv',
                    help='결과 CSV (프로젝트 루트 기준 상대 경로)')
    ap.add_argument('--model', default='openai/whisper-large-v3')
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────
    # 1. 데이터 로드 + 샘플링
    # ─────────────────────────────────────────
    print('🔍 VOTE400 낭독체 인덱싱 중...')
    items = load_read(Path(args.root) / 'VOTE400_Read')
    print(f'   전체 {len(items):,}개 발화')

    if args.sample_per_group > 0:
        random.seed(args.seed)
        by_group = defaultdict(list)
        for it in items:
            by_group[f"{it['region']}{it['group']}"].append(it)
        sampled = []
        for k in sorted(by_group.keys()):
            pool = by_group[k]
            random.shuffle(pool)
            sampled.extend(pool[:args.sample_per_group])
        items = sampled
        print(f'   샘플링 후 {len(items):,}개 (그룹당 최대 {args.sample_per_group})')

    # ─────────────────────────────────────────
    # 2. Whisper 모델 로드
    # ─────────────────────────────────────────
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n⏳ {args.model} 로딩 중... (device={device})')
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    ).to(device)
    model.eval()
    print('✅ 모델 로드 완료')

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language='ko', task='transcribe'
    )

    # ─────────────────────────────────────────
    # 3. 배치 추론
    # ─────────────────────────────────────────
    from tqdm import tqdm

    print(f'\n⏳ 추론 시작 (batch={args.batch_size}, 총 {len(items):,}개)')
    t0 = time.time()

    for i in tqdm(range(0, len(items), args.batch_size), desc='Transcribing'):
        batch = items[i:i + args.batch_size]
        audios = [librosa.load(it['wav_path'], sr=16000)[0] for it in batch]
        inputs = processor(audios, sampling_rate=16000,
                           return_tensors='pt', padding=True)
        feats = inputs.input_features.to(device)
        if device == 'cuda':
            feats = feats.half()
        with torch.no_grad():
            pred_ids = model.generate(
                feats,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=128,
                num_beams=5,
            )
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        for it, pred in zip(batch, preds):
            it['prediction'] = pred.strip()

    elapsed = time.time() - t0
    print(f'✅ 추론 완료: {elapsed/60:.1f}분 / {len(items)/elapsed:.1f}발화/초')

    # ─────────────────────────────────────────
    # 4. CER 계산 + 그룹별 요약
    # ─────────────────────────────────────────
    from jiwer import cer, wer

    refs = [it['transcript'] for it in items]
    hyps = [it.get('prediction', '').strip() or ' ' for it in items]
    overall_cer = cer(refs, hyps)
    overall_wer = wer(refs, hyps)

    print('\n' + '=' * 60)
    print(f'📊 VOTE400 낭독체 zero-shot 베이스라인 (E0\')')
    print('=' * 60)
    print(f'  발화 수: {len(items):,}')
    print(f'  전체 CER: {overall_cer*100:.2f}%')
    print(f'  전체 WER: {overall_wer*100:.2f}%')
    print(f'  추론 시간: {elapsed/60:.1f}분')

    # 그룹별 CER
    print('\n=== 그룹별 CER (어려운 그룹이 누군지 보기) ===')
    by_group = defaultdict(list)
    for it in items:
        by_group[f"{it['region']}{it['group']}"].append(it)
    print(f'{"그룹":>6} {"발화수":>8} {"CER":>8}')
    rows = []
    for k in sorted(by_group.keys()):
        bs = by_group[k]
        g_refs = [it['transcript'] for it in bs]
        g_hyps = [it.get('prediction', '').strip() or ' ' for it in bs]
        g_cer = cer(g_refs, g_hyps)
        rows.append((k, len(bs), g_cer))
    # CER 오름차순
    for k, n, c in sorted(rows, key=lambda x: x[2]):
        print(f'{k:>6} {n:>8} {c*100:>7.2f}%')

    # ─────────────────────────────────────────
    # 5. CSV 저장 (분석 스크립트에서 재사용)
    # ─────────────────────────────────────────
    fields = ['mode', 'region', 'group', 'pid', 'date', 'utt_no',
              'wav_path', 'transcript', 'prediction']
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for it in items:
            writer.writerow(it)
    print(f'\n✅ 저장: {out_path}')


if __name__ == '__main__':
    main()
