"""
E1' 예측 추출: 학습된 best LoRA 어댑터로 val 1,500개를 추론해 보고서용 CSV 생성.

- 학습(E1prime_VOTE400_LoRA.py)과 **동일한 val split(seed=42)** 을 재현하므로
  결과 CSV의 1,500개는 학습 시 평가에 쓰인 것과 같은 발화다.
- 출력 CSV 포맷은 E0'(results/E0prime_vote400_read_sampled.csv)와 동일 →
  그룹별 CER 비교/오류 사례 분석을 바로 할 수 있다.
- 디코딩 파라미터는 최종 평가와 동일: language="ko", num_beams=5, max_new_tokens=128.

CLI:
  CUDA_VISIBLE_DEVICES=0 python code/E1prime_VOTE400_predict.py \\
      --root data/VOTE400 \\
      --adapter whisper-lora-vote400/adapter_final \\
      --out results/E1prime_vote400_read_pred.csv
"""
import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vote400_loader import load_read


def build_val_split(dataset, val_size=1500, seed=42):
    """학습 스크립트와 동일한 split 로직 (그룹 비율 유지 + 서브샘플). 동일 seed → 동일 1,500개."""
    by_group = defaultdict(set)
    for it in dataset:
        by_group[f"{it['region']}{it['group']}"].add(it['pid'])

    random.seed(seed)
    val_pids = set()
    for g, pids in by_group.items():
        pids_list = sorted(list(pids))
        random.shuffle(pids_list)
        num_val = max(1, int(len(pids_list) * 0.1))
        val_pids.update(pids_list[:num_val])

    val_data = [d for d in dataset if d['pid'] in val_pids]

    if val_size > 0 and len(val_data) > val_size:
        val_by_group = defaultdict(list)
        for d in val_data:
            val_by_group[f"{d['region']}{d['group']}"].append(d)
        sampled = []
        for g in sorted(val_by_group):
            pool = val_by_group[g]
            random.shuffle(pool)
            take = max(1, round(val_size * len(pool) / len(val_data)))
            sampled.extend(pool[:take])
        random.shuffle(sampled)
        val_data = sampled[:val_size]
    return val_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/storage/cholin2/whisper/data/VOTE400')
    ap.add_argument('--adapter', default='/storage/cholin2/whisper/whisper-lora-vote400/adapter_final',
                    help='학습된 LoRA 어댑터 경로 (load_best 적용된 adapter_final)')
    ap.add_argument('--model', default='openai/whisper-large-v3')
    ap.add_argument('--val-size', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--out', default='results/E1prime_vote400_read_pred.csv')
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parent.parent
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 동일 val split 재현
    print('🔍 VOTE400 낭독체 인덱싱 중...')
    dataset = load_read(Path(args.root) / 'VOTE400_Read')
    print(f'   전체 {len(dataset):,}개')
    val_data = build_val_split(dataset, args.val_size, args.seed)
    print(f'   val {len(val_data):,}개 (학습과 동일 split)')

    # 2. 모델 + 어댑터 로드
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel
    import librosa

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n⏳ 모델 로딩... (device={device})')
    processor = WhisperProcessor.from_pretrained(args.model, language='ko', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    ).to(device)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.generation_config.language = 'ko'
    model.generation_config.task = 'transcribe'
    model.generation_config.forced_decoder_ids = None
    print('✅ base + LoRA 어댑터 로드 완료')

    # 3. 배치 추론 (최종 평가와 동일 디코딩: beam5, max128)
    from tqdm import tqdm
    print(f'\n⏳ 추론 시작 (batch={args.batch_size}, {len(val_data):,}개, beam5)')
    t0 = time.time()
    for i in tqdm(range(0, len(val_data), args.batch_size), desc='Transcribing'):
        batch = val_data[i:i + args.batch_size]
        audios = [librosa.load(it['wav_path'], sr=16000)[0] for it in batch]
        inputs = processor(audios, sampling_rate=16000, return_tensors='pt', padding=True)
        feats = inputs.input_features.to(device)
        if device == 'cuda':
            feats = feats.half()
        with torch.no_grad():
            pred_ids = model.generate(feats, max_new_tokens=128, num_beams=5)
        preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
        for it, pred in zip(batch, preds):
            it['prediction'] = pred.strip()
    elapsed = time.time() - t0
    print(f'✅ 추론 완료: {elapsed/60:.1f}분')

    # 4. CER 계산 + 그룹별 요약
    from jiwer import cer, wer
    refs = [it['transcript'] for it in val_data]
    hyps = [it.get('prediction', '').strip() or ' ' for it in val_data]
    overall_cer = cer(refs, hyps)
    overall_wer = wer(refs, hyps)

    print('\n' + '=' * 60)
    print("📊 E1' 예측 결과 (val 1,500, beam5)")
    print('=' * 60)
    print(f'  전체 CER: {overall_cer*100:.2f}%   WER: {overall_wer*100:.2f}%')

    print('\n=== 그룹별 CER ===')
    by_group = defaultdict(list)
    for it in val_data:
        by_group[f"{it['region']}{it['group']}"].append(it)
    print(f'{"그룹":>6} {"발화수":>7} {"CER":>8}')
    for k in sorted(by_group):
        bs = by_group[k]
        g_cer = cer([it['transcript'] for it in bs],
                    [it.get('prediction', '').strip() or ' ' for it in bs])
        print(f'{k:>6} {len(bs):>7} {g_cer*100:>7.2f}%')

    # 5. CSV 저장 (E0'와 동일 포맷)
    fields = ['mode', 'region', 'group', 'pid', 'date', 'utt_no',
              'wav_path', 'transcript', 'prediction']
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for it in val_data:
            w.writerow(it)
    print(f'\n✅ 저장: {out_path}')


if __name__ == '__main__':
    main()
