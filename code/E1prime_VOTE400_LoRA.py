"""
E1': Whisper-large-v3 + LoRA Fine-Tuning on VOTE400 (낭독체)
학교 GPU 서버 (RTX 8000 48GB x 2) 환경에 맞춘 파인튜닝 스크립트.
"""
import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import librosa
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
from jiwer import cer, wer

# 현재 폴더에서 vote400_loader 임포트
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vote400_loader import load_read

from dataclasses import dataclass
from typing import Any

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """On-the-fly 전처리 콜레이터.

    {wav_path, transcript} 를 받아 batch 단위로 log-mel feature + token ids 로 변환한다.
    datasets.map(num_proc=N) 사전 전처리는 librosa+토크나이저 멀티프로세싱 deadlock을
    유발했고(코드 주석 경고대로), 91GB 디스크 캐시 + 21분 대기도 발생했다.
    DataLoader(num_workers=N) 가 이 변환을 백그라운드로 병렬 prefetch 하므로
    사전 전처리 단계 자체를 없앤다.
    """
    processor: Any

    def __call__(self, features):
        input_features, label_features = [], []
        for f in features:
            audio, _ = librosa.load(f["wav_path"], sr=16000)
            feat = self.processor.feature_extractor(
                audio, sampling_rate=16000
            ).input_features[0]
            input_features.append({"input_features": feat})
            label_features.append(
                {"input_ids": self.processor.tokenizer(f["transcript"]).input_ids}
            )

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

class AutocastSeq2SeqTrainer(Seq2SeqTrainer):
    """eval의 generate를 fp16 autocast로 감싸는 Trainer.

    fp16=True 학습에서 Seq2SeqTrainer는 학습 forward는 autocast로 감싸지만
    eval의 generate(예측)는 감싸지 않는다. 그 결과 Whisper encoder conv에서
    input(float32) vs bias(fp16) dtype 충돌이 발생한다. prediction_step을
    autocast로 감싸 학습 설정은 그대로 두고 eval만 fp16 일관성을 보장한다.
    """
    def prediction_step(self, *args, **kwargs):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return super().prediction_step(*args, **kwargs)


def clean_text(t):
    return ' '.join(t.split())

def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    label_str = [clean_text(s) for s in label_str]
    pred_str  = [s.strip() if s.strip() else " " for s in pred_str]

    return {
        "cer": cer(label_str, pred_str),
        "wer": wer(label_str, pred_str),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/storage/cholin2/whisper/data/VOTE400',
                    help='VOTE400 폴더 경로')
    ap.add_argument('--out-dir', default='/storage/cholin2/whisper/whisper-lora-vote400',
                    help='학습 결과 저장 경로')
    ap.add_argument('--model', default='openai/whisper-large-v3')
    ap.add_argument('--batch-size', type=int, default=8,
                    help='GPU당 배치 크기. app.py가 GPU1 12.7GB 점유 중이라 가용 34GB → 8 적합 '
                         '(16은 OOM). accum=2와 함께 유효 배치 32 유지')
    ap.add_argument('--epochs', type=int, default=2, help='학습 에포크 수')
    ap.add_argument('--val-size', type=int, default=1500,
                    help='validation 서브샘플 크기 (0=전체). 학습 중 eval(generate) 비용 절감용')
    args = ap.parse_args()

    # 1. 데이터 로드 및 분할 (Stratified Split by Group)
    print("🔍 VOTE400 낭독체 데이터 로딩 중...")
    read_root = Path(args.root) / 'VOTE400_Read'
    if not read_root.exists():
        print(f"❌ 경로를 찾을 수 없습니다: {read_root}")
        sys.exit(1)
        
    dataset = load_read(read_root)
    print(f"✅ 로드 완료: {len(dataset):,}개 발화")

    # 화자 ID(pid) 기준으로 10%를 validation으로 분할하되 그룹 비율 유지
    # 그룹 키는 region+group (예: "DG1", "GN1") — group 숫자만 쓰면 지역이 섞임
    by_group = defaultdict(set)
    for it in dataset:
        by_group[f"{it['region']}{it['group']}"].add(it['pid'])

    random.seed(42)
    val_pids = set()
    for g, pids in by_group.items():
        pids_list = sorted(list(pids))
        random.shuffle(pids_list)
        num_val = max(1, int(len(pids_list) * 0.1))
        val_pids.update(pids_list[:num_val])

    train_data = [d for d in dataset if d['pid'] not in val_pids]
    val_data   = [d for d in dataset if d['pid'] in val_pids]

    print(f"Train: {len(train_data):,} 발화 (화자 {len(set(d['pid'] for d in train_data))}명)")
    print(f"Val:   {len(val_data):,} 발화 (화자 {len(val_pids)}명)")

    # val 전체(16K+)를 매 eval마다 beam-search generate하면 학습보다 비싸진다.
    # 학습 중 모니터링용으로 그룹 비율을 유지하며 서브샘플. (train_data는 불변 → train map 캐시 재사용)
    if args.val_size > 0 and len(val_data) > args.val_size:
        val_by_group = defaultdict(list)
        for d in val_data:
            val_by_group[f"{d['region']}{d['group']}"].append(d)
        sampled = []
        for g in sorted(val_by_group):
            pool = val_by_group[g]
            random.shuffle(pool)
            take = max(1, round(args.val_size * len(pool) / len(val_data)))
            sampled.extend(pool[:take])
        random.shuffle(sampled)
        val_data = sampled[:args.val_size]
        print(f"   → Val 서브샘플링: {len(val_data):,}개 (eval generate 비용 절감)")

    # 2. Processor & Model 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    processor = WhisperProcessor.from_pretrained(args.model, language="ko", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    # eval generate 시 언어 자동감지(detect_language) 대신 한국어 고정.
    # 자동감지 경로가 fp16에서 dtype 충돌(Input float vs bias Half)을 일으켰고,
    # 한국어 고정은 E0'(language="ko")와도 일관된다.
    model.generation_config.language = "ko"
    model.generation_config.task = "transcribe"

    model.config.use_cache = False
    
    # 3. LoRA 설정
    # task_type 미지정: Whisper는 input_features(오디오) 기반이라
    # task_type="SEQ_2_SEQ_LM"(input_ids 기반 텍스트 seq2seq 가정)를 주면
    # PeftModelForSeq2SeqLM.forward가 decoder에 input_ids를 중복 전달해 충돌한다.
    # 생략하면 기본 PeftModel이 forward를 base Whisper에 그대로 위임한다.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Trainer 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,   # batch8 × accum2 × 2GPU = 유효 배치 32
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        warmup_steps=100,
        num_train_epochs=args.epochs,
        fp16=True,
        eval_strategy="steps",
        eval_steps=500,    # 첫 eval 검증을 ~29분으로 당기고 체크포인트를 일찍 확보
        save_steps=500,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=1,   # 학습 중 eval은 greedy로 빠르게. 최종 평가만 beam5(아래).
        save_total_limit=2,
        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to=[],
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,    # 콜레이터의 on-the-fly 전처리를 백그라운드 병렬 prefetch
        remove_unused_columns=False, # wav_path/transcript를 콜레이터까지 전달 (자동 제거 방지)
    )

    # 5. HF Dataset 준비 (사전 전처리 없음 — 콜레이터가 batch 단위로 즉석 변환)
    train_ds = Dataset.from_list(
        [{"wav_path": d["wav_path"], "transcript": d["transcript"]} for d in train_data]
    )
    val_ds = Dataset.from_list(
        [{"wav_path": d["wav_path"], "transcript": d["transcript"]} for d in val_data]
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    trainer = AutocastSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
        processing_class=processor.feature_extractor,
    )

    print("⏳ 학습 시작...")
    trainer.train()
    print("✅ 학습 완료")

    # 최종 평가는 E0'(beam5)와 공정 비교를 위해 beam5로 한 번 더.
    trainer.args.generation_num_beams = 5
    metrics = trainer.evaluate()
    print("=" * 50)
    print(f"📊 E1' (LoRA on VOTE400) 결과  (val {len(val_data):,}개, beam5)")
    print(f"   CER: {metrics['eval_cer']*100:.2f}%")
    print(f"   WER: {metrics['eval_wer']*100:.2f}%")
    print("=" * 50)

    model.save_pretrained(os.path.join(args.out_dir, "adapter_final"))
    processor.save_pretrained(os.path.join(args.out_dir, "adapter_final"))
    print(f"✅ 어댑터 저장 완료: {args.out_dir}/adapter_final")

if __name__ == '__main__':
    main()
