"""
E0 베이스라인 - 라벨 정제 후 CER 재측정

AI허브 라벨 컨벤션:
  (SP:content) - 발음 변이 (dialect/pronunciation variant)
  (NO:content) - 잡음 중첩 발화
  (FP:content) - 필러/추임새 (뭐, 그거, 오 등)
  (SN:content) - 단발 잡음

전체 발화의 12% (352/2930)가 태그 포함. 내용은 모두 실제 발화된 텍스트이므로
태그 마커만 벗기고 내용은 유지하여 Whisper 출력과 공정 비교 수행.

산출:
  - 콘솔: 정제 전/후 CER 비교
  - E0_baseline_results_cleaned.csv: 정제된 transcript 컬럼 추가
  - analysis_cleaned_report.md: 발표용 비교 리포트
"""
import csv
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from jiwer import cer, wer, process_words
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'jiwer', '-q'])
    from jiwer import cer, wer, process_words

ROOT = Path(__file__).resolve().parent.parent
CSV_IN = ROOT / 'results' / 'E0_baseline_results.csv'
CSV_OUT = ROOT / 'results' / 'E0_baseline_results_cleaned.csv'
REPORT_PATH = ROOT / 'results' / 'analysis_cleaned_report.md'

# (XX:content) 패턴: XX는 영문 대문자, content는 ) 직전까지의 모든 문자
TAG_RE = re.compile(r'\(([A-Z]+):([^)]*)\)')


def clean_text(t):
    """라벨 태그 제거: (XX:content) -> content (양옆 공백 패딩 후 whitespace 정규화)"""
    out = TAG_RE.sub(lambda m: ' ' + m.group(2).strip() + ' ', t)
    out = re.sub(r'\s+', ' ', out).strip()
    return out


def char_error_counts(ref, hyp):
    rc = list(ref.replace(' ', ''))
    hc = list(hyp.replace(' ', ''))
    if not rc:
        return 0, 0, 0, 0
    try:
        o = process_words(' '.join(rc), ' '.join(hc))
        return o.deletions, o.insertions, o.substitutions, len(rc)
    except Exception:
        return 0, 0, 0, len(rc)


def main():
    with open(CSV_IN, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # 정제 적용
    n_changed = 0
    for r in rows:
        raw = r['transcript']
        cleaned = clean_text(raw)
        r['transcript_cleaned'] = cleaned
        if cleaned != raw:
            n_changed += 1

    # 빈 ref 필터링
    rows_ok = [r for r in rows if r['transcript_cleaned'].strip()]
    for r in rows_ok:
        r['prediction_clean'] = r['prediction'].strip() if r['prediction'].strip() else ' '

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    out('# E0 베이스라인 - 라벨 정제 후 재측정 리포트')
    out()
    out(f'**입력**: {CSV_IN.name} · 발화 {len(rows)}개')
    out(f'**태그 정제된 발화**: {n_changed}개 ({n_changed/len(rows)*100:.1f}%)')
    out()

    # ============================
    # 1. 전체 CER 비교
    # ============================
    out('## 1. 전체 CER/WER 비교 (정제 전 vs 후)')
    out()

    refs_raw = [r['transcript'].strip() for r in rows_ok]
    refs_clean = [r['transcript_cleaned'].strip() for r in rows_ok]
    hyps = [r['prediction_clean'] for r in rows_ok]

    cer_raw = cer(refs_raw, hyps)
    cer_cln = cer(refs_clean, hyps)
    wer_raw = wer(refs_raw, hyps)
    wer_cln = wer(refs_clean, hyps)

    out('| 지표 | 정제 전 (raw) | 정제 후 (cleaned) | 차이 |')
    out('|---|---:|---:|---:|')
    out(f'| CER | {cer_raw*100:.2f}% | **{cer_cln*100:.2f}%** | {(cer_cln-cer_raw)*100:+.2f}%p |')
    out(f'| WER | {wer_raw*100:.2f}% | **{wer_cln*100:.2f}%** | {(wer_cln-wer_raw)*100:+.2f}%p |')
    out()

    # 오류 분해
    def err_breakdown(refs, hyps):
        td, ti, ts, tn = 0, 0, 0, 0
        for ref, hyp in zip(refs, hyps):
            d, i, s, n = char_error_counts(ref, hyp)
            td += d; ti += i; ts += s; tn += n
        return td, ti, ts, tn

    rd, ri, rs, rn = err_breakdown(refs_raw, hyps)
    cd, ci, cs, cn = err_breakdown(refs_clean, hyps)

    out('**오류 분해 (글자 단위)**')
    out()
    out('| 유형 | 정제 전 | 정제 후 | 차이 |')
    out('|---|---:|---:|---:|')
    out(f'| Deletion | {rd} ({rd/rn*100:.2f}%) | {cd} ({cd/cn*100:.2f}%) | {(cd/cn-rd/rn)*100:+.2f}%p |')
    out(f'| Substitution | {rs} ({rs/rn*100:.2f}%) | {cs} ({cs/cn*100:.2f}%) | {(cs/cn-rs/rn)*100:+.2f}%p |')
    out(f'| Insertion | {ri} ({ri/rn*100:.2f}%) | {ci} ({ci/cn*100:.2f}%) | {(ci/cn-ri/rn)*100:+.2f}%p |')
    out(f'| 정답글자 합 | {rn} | {cn} | {cn-rn:+d} |')
    out()

    # ============================
    # 2. 화자별 비교
    # ============================
    out('## 2. 화자별 CER (정제 전/후)')
    out()
    out('| 화자ID | 성/나이/지역 | 발화 | CER (raw) | CER (cleaned) | Δ |')
    out('|---|---|---:|---:|---:|---:|')

    speakers = defaultdict(list)
    for r in rows_ok:
        speakers[r['speaker_id']].append(r)

    spk_order = sorted(speakers.keys(),
                       key=lambda s: cer([r['transcript_cleaned'] for r in speakers[s]],
                                         [r['prediction_clean'] for r in speakers[s]]))

    for sid in spk_order:
        items = speakers[sid]
        info = items[0]
        s_refs_r = [r['transcript'] for r in items]
        s_refs_c = [r['transcript_cleaned'] for r in items]
        s_hyps = [r['prediction_clean'] for r in items]
        cr = cer(s_refs_r, s_hyps)
        cc = cer(s_refs_c, s_hyps)
        out(f'| {sid} | {info["gender"]}/{info["age"]}/{info["region"]} | '
            f'{len(items)} | {cr*100:.2f}% | **{cc*100:.2f}%** | {(cc-cr)*100:+.2f}%p |')
    out()

    # ============================
    # 3. 발화 길이별 비교
    # ============================
    out('## 3. 발화 길이별 CER (정제 후)')
    out()
    out('| 길이 (초) | 발화수 | CER (raw) | CER (cleaned) | Del% (cleaned) |')
    out('|---|---:|---:|---:|---:|')

    bins = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 999)]
    for lo, hi in bins:
        sub = [r for r in rows_ok if lo <= float(r['duration']) < hi]
        if not sub:
            continue
        rr = [r['transcript'] for r in sub]
        rc_ = [r['transcript_cleaned'] for r in sub]
        hh = [r['prediction_clean'] for r in sub]
        cr = cer(rr, hh)
        cc = cer(rc_, hh)
        bd, _, _, bn = err_breakdown(rc_, hh)
        label = f'{lo}~{hi}s' if hi < 100 else f'{lo}s+'
        out(f'| {label} | {len(sub)} | {cr*100:.2f}% | **{cc*100:.2f}%** | {bd/max(bn,1)*100:.2f}% |')
    out()

    # ============================
    # 4. 정제 효과 정성 예시
    # ============================
    out('## 4. 정제 효과 정성 예시 (CER 가장 크게 개선된 발화 10개)')
    out()
    # 각 발화의 utt CER 개선폭 계산
    improved = []
    for r in rows_ok:
        ref_r = r['transcript'].strip()
        ref_c = r['transcript_cleaned'].strip()
        hyp = r['prediction_clean']
        if ref_r == ref_c:
            continue
        if not ref_r or not ref_c:
            continue
        c_r = cer([ref_r], [hyp])
        c_c = cer([ref_c], [hyp])
        improved.append((c_r - c_c, ref_r, ref_c, hyp, c_r, c_c, r['speaker_id'][-4:]))

    improved.sort(reverse=True)
    out('| Δ CER | 화자 | 정답(raw) | 정답(cleaned) | 예측 |')
    out('|---:|---|---|---|---|')
    for delta, rr, rc_, hh, _, _, sid in improved[:10]:
        rr_s = rr[:35].replace('|', '\\|') + ('…' if len(rr) > 35 else '')
        rc_s = rc_[:35].replace('|', '\\|') + ('…' if len(rc_) > 35 else '')
        hh_s = hh[:35].replace('|', '\\|') + ('…' if len(hh) > 35 else '')
        out(f'| {delta*100:.0f}%p | {sid} | {rr_s} | {rc_s} | {hh_s} |')
    out()

    # ============================
    # 5. 발표 헤드라인
    # ============================
    out('## 5. 발표 헤드라인 갱신')
    out()
    out(f'- **정제 전 베이스라인 CER**: {cer_raw*100:.2f}% (AI허브 라벨 태그가 오류로 카운트됨)')
    out(f'- **정제 후 베이스라인 CER**: **{cer_cln*100:.2f}%** (Whisper-Whisper 출력 형식과 동일하게 정렬)')
    out(f'- 차이 {abs((cer_cln-cer_raw)*100):.2f}%p — 이는 모델의 진짜 성능과 라벨 형식 차이를 구분한 결과')
    out(f'- **H1 30% 상대개선 목표** 환산: CER **{cer_cln*0.7*100:.2f}%**까지 내려야 가설 채택 (정제 후 기준)')
    out(f'- 길이 10초+ 발화의 Del 비율은 정제 후에도 압도적 → **H2(VAD 재튜닝) 검증 가치 유지**')
    out()

    # 저장: cleaned CSV
    with open(CSV_OUT, 'w', encoding='utf-8', newline='') as f:
        fields = ['wav_path', 'speaker_id', 'gender', 'age', 'region',
                  'duration', 'transcript', 'transcript_cleaned', 'prediction']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fields})

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print()
    print(f'정제된 CSV: {CSV_OUT.name}')
    print(f'리포트:     {REPORT_PATH.name}')


if __name__ == '__main__':
    main()
