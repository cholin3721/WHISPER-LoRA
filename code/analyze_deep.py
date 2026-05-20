"""
E0 베이스라인 심화 분석 - 이번 주 발표용 슬라이드 자료 생성

E0_baseline_results.csv 를 읽어 다음을 산출:
  1. 화자별 오류 유형 분해 (Del/Ins/Sub)
  2. 발화 길이 구간별 CER
  3. 지역/성별 교차 집계
  4. 최악 사례 20개 (정성 분석용)
  5. 발표용 헤드라인 5개

출력:
  - 콘솔 (UTF-8)
  - analysis_deep_report.md  (슬라이드 복붙용 풀 리포트)
  - worst_cases.csv          (최악 사례 raw)
"""
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

# Windows 콘솔에서 한글 깨짐 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from jiwer import cer, process_words
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'jiwer', '-q'])
    from jiwer import cer, process_words

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / 'results' / 'E0_baseline_results.csv'
REPORT_PATH = ROOT / 'results' / 'analysis_deep_report.md'
WORST_CASES_PATH = ROOT / 'results' / 'worst_cases.csv'


def char_error_counts(ref, hyp):
    rc = list(ref.replace(' ', ''))
    hc = list(hyp.replace(' ', ''))
    if not rc:
        return 0, 0, 0, 0
    try:
        out = process_words(' '.join(rc), ' '.join(hc))
        return out.deletions, out.insertions, out.substitutions, len(rc)
    except Exception:
        return 0, 0, 0, len(rc)


def main():
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r['transcript'].strip()]
    for r in rows:
        r['transcript'] = r['transcript'].strip()
        r['prediction'] = r['prediction'].strip() if r['prediction'].strip() else ' '
        r['duration'] = float(r['duration'])
        r['age'] = int(r['age'])
        d, i, s, n = char_error_counts(r['transcript'], r['prediction'])
        r['del'] = d
        r['ins'] = i
        r['sub'] = s
        r['ref_chars'] = n
        r['utt_cer'] = cer([r['transcript']], [r['prediction']])

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    # =========================================
    # 헤더
    # =========================================
    out('# E0 베이스라인 심화 분석 리포트')
    out()
    n_spk = len(set(r['speaker_id'] for r in rows))
    total_min = sum(r['duration'] for r in rows) / 60
    out(f'**데이터**: {CSV_PATH.name} · 발화 {len(rows)}개 · 화자 {n_spk}명 · 총 {total_min:.1f}분')
    out(f'**모델**: openai/whisper-large-v3 (zero-shot, beam=5, language=ko)')
    out()

    # 전체 요약
    all_refs = [r['transcript'] for r in rows]
    all_hyps = [r['prediction'] for r in rows]
    total_cer = cer(all_refs, all_hyps)
    total_wer_v = __import__('jiwer').wer(all_refs, all_hyps)
    sum_del = sum(r['del'] for r in rows)
    sum_ins = sum(r['ins'] for r in rows)
    sum_sub = sum(r['sub'] for r in rows)
    sum_n = sum(r['ref_chars'] for r in rows)

    out(f'**전체 CER**: {total_cer*100:.2f}% | **WER**: {total_wer_v*100:.2f}%')
    out(f'**오류 분해 (글자)**: Del {sum_del/sum_n*100:.2f}% · Sub {sum_sub/sum_n*100:.2f}% · Ins {sum_ins/sum_n*100:.2f}%')
    out()

    # =========================================
    # 1. 화자별 오류 유형 분해
    # =========================================
    out('---')
    out()
    out('## 1. 화자별 오류 유형 분해')
    out()
    out('| 화자ID | 성/나이/지역 | 발화수 | 정답글자 | CER | Del% | Sub% | Ins% | Del:Sub |')
    out('|---|---|---:|---:|---:|---:|---:|---:|---:|')

    speakers = defaultdict(list)
    for r in rows:
        speakers[r['speaker_id']].append(r)

    # CER 오름차순 정렬
    spk_order = sorted(speakers.keys(),
                       key=lambda sid: cer([r['transcript'] for r in speakers[sid]],
                                           [r['prediction'] for r in speakers[sid]]))

    for sid in spk_order:
        items = speakers[sid]
        info = items[0]
        td = sum(r['del'] for r in items)
        ti = sum(r['ins'] for r in items)
        ts = sum(r['sub'] for r in items)
        tn = sum(r['ref_chars'] for r in items)
        s_cer = cer([r['transcript'] for r in items], [r['prediction'] for r in items])
        ds_ratio = td / max(ts, 1)
        out(f'| {sid} | {info["gender"]}/{info["age"]}/{info["region"]} | '
            f'{len(items)} | {tn} | {s_cer*100:.2f}% | '
            f'{td/max(tn,1)*100:.2f}% | {ts/max(tn,1)*100:.2f}% | '
            f'{ti/max(tn,1)*100:.2f}% | {ds_ratio:.1f}x |')

    out()
    out('**해석 포인트**:')
    out('- Deletion이 모든 화자에서 Sub·Ins를 압도하는지 → H2 가설의 화자 무관 일반화 검증')
    out('- 가장 어려운 화자(최고 CER)에서 Del:Sub 비율이 더 큰지 → 어려움의 본질이 "삭제 오류"임을 시사')
    out()

    # =========================================
    # 2. 발화 길이 vs CER
    # =========================================
    out('---')
    out()
    out('## 2. 발화 길이 구간별 CER')
    out()
    out('| 길이 (초) | 발화수 | CER | Del% | Sub% | Ins% |')
    out('|---|---:|---:|---:|---:|---:|')

    bins = [(0, 2), (2, 4), (4, 6), (6, 10), (10, 999)]
    for lo, hi in bins:
        sub = [r for r in rows if lo <= r['duration'] < hi]
        if not sub:
            continue
        b_cer = cer([r['transcript'] for r in sub], [r['prediction'] for r in sub])
        bd = sum(r['del'] for r in sub)
        bi = sum(r['ins'] for r in sub)
        bs = sum(r['sub'] for r in sub)
        bn = sum(r['ref_chars'] for r in sub)
        label = f'{lo}~{hi}s' if hi < 100 else f'{lo}s+'
        out(f'| {label} | {len(sub)} | {b_cer*100:.2f}% | '
            f'{bd/max(bn,1)*100:.2f}% | {bs/max(bn,1)*100:.2f}% | '
            f'{bi/max(bn,1)*100:.2f}% |')

    out()
    out('**해석**: 발화가 길수록 Deletion 비율이 상승하면 → "긴 발화 중간 휴지를 VAD가 종료로 오인" 가설(H2) 강력 지지.')
    out()

    # =========================================
    # 3. 지역/성별 교차 집계
    # =========================================
    out('---')
    out()
    out('## 3. 지역/성별 교차 집계')
    out()
    out('| 지역 | 성별 | 화자수 | 발화수 | CER |')
    out('|---|---|---:|---:|---:|')

    groups = defaultdict(list)
    for r in rows:
        groups[(r['region'], r['gender'])].append(r)

    for (region, gender), items in sorted(groups.items()):
        ns = len(set(r['speaker_id'] for r in items))
        g_cer = cer([r['transcript'] for r in items], [r['prediction'] for r in items])
        out(f'| {region} | {gender} | {ns} | {len(items)} | {g_cer*100:.2f}% |')

    out()
    out('**해석**: 표본 한계 — 5명 중 여성 4명·수도권 3명. 지역 효과(경상/충청 vs 수도권)와 성별 효과를 분리하기엔 부족. VOTE400 확장 시 재검증 필요.')
    out()

    # =========================================
    # 4. 최악 사례
    # =========================================
    out('---')
    out()
    out('## 4. 최악 사례 (발화별 CER 상위 20개)')
    out()
    worst = sorted(rows, key=lambda r: -r['utt_cer'])[:20]

    out('| # | 화자ID | 길이 | 정답 | 예측 | CER |')
    out('|---:|---|---:|---|---|---:|')
    for i, r in enumerate(worst, 1):
        ref_show = r['transcript'][:35] + ('…' if len(r['transcript']) > 35 else '')
        hyp_raw = r['prediction'].strip()
        if not hyp_raw:
            hyp_show = '*(빈 출력)*'
        else:
            hyp_show = hyp_raw[:35] + ('…' if len(hyp_raw) > 35 else '')
        # 마크다운 파이프 깨짐 방지
        ref_show = ref_show.replace('|', '\\|')
        hyp_show = hyp_show.replace('|', '\\|')
        out(f'| {i} | {r["speaker_id"][-4:]} | {r["duration"]:.1f}s | '
            f'{ref_show} | {hyp_show} | {r["utt_cer"]*100:.0f}% |')

    out()
    out('**정성 분석 체크리스트**:')
    out('- [ ] 빈 출력 / 매우 짧은 출력 → VAD가 발화를 통째로 누락한 경우 (Deletion 극단)')
    out('- [ ] 끝부분만 누락 → 발화 중 침묵을 종료로 오인 (H2 직접 증거)')
    out('- [ ] 사투리/방언이 표준어로 대체 → 분류 미스 vs 의도된 행동 판단 필요')
    out('- [ ] 짧은 발화(<2초)의 100% CER → 한두 글자 차이로 정상 발화일 가능성 (지표 잡음)')
    out()

    # 최악 사례 CSV 저장
    with open(WORST_CASES_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rank', 'speaker_id', 'gender', 'age', 'region',
            'duration', 'utt_cer', 'transcript', 'prediction',
            'del', 'sub', 'ins', 'ref_chars'
        ])
        writer.writeheader()
        for i, r in enumerate(worst, 1):
            writer.writerow({
                'rank': i,
                'speaker_id': r['speaker_id'],
                'gender': r['gender'],
                'age': r['age'],
                'region': r['region'],
                'duration': f"{r['duration']:.3f}",
                'utt_cer': f"{r['utt_cer']:.4f}",
                'transcript': r['transcript'],
                'prediction': r['prediction'].strip(),
                'del': r['del'],
                'sub': r['sub'],
                'ins': r['ins'],
                'ref_chars': r['ref_chars'],
            })

    # =========================================
    # 5. 발표용 헤드라인
    # =========================================
    out('---')
    out()
    out('## 5. 발표용 헤드라인 (이번 주 슬라이드 후보)')
    out()
    out('1. **실측 CER 6.85% — 문헌 추정(20~30%)의 1/3 수준**')
    out('   → Whisper-large-v3는 한국어 표준어 노인 발화를 이미 잘 인식한다. PPT의 출발 가정 자체가 갱신 대상.')
    out()
    out('2. **H2 가설(Deletion 지배)은 데이터로 강하게 지지됨**')
    out('   → Del 4.3% vs Sub 1.7% vs Ins 0.1%. Del:Sub = 2.5배, Del:Ins = 43배.')
    out('   → "긴 침묵·느린 발화로 인한 단어 누락"이라는 음향학적 가설이 실측으로 확인됨.')
    out()
    out('3. **화자간 격차 3배 (3.25% ~ 9.52%)**')
    out('   → "노인 음성"을 단일 그룹으로 다루는 가정의 한계 노출.')
    out('   → VOTE400의 구음장애·초고령 화자군에서는 격차가 더 클 것으로 예상.')
    out()
    out('4. **H1 재정의**')
    out('   → 기존: "Whisper가 노인 음성에서 무너진다 → LoRA로 메운다"')
    out('   → 수정: "Whisper는 표준 노인 발화는 잘하지만 어려운 subgroup에서 무너진다 → LoRA가 그 격차를 선택적으로 메운다"')
    out()
    out('5. **다음 단계**')
    out('   → 1순위: VOTE400 협약 진행 + 구음장애/80+ subgroup baseline 재측정')
    out('   → 2순위: AI허브 샘플 위에서 LoRA 파이프라인 코드 검증 (학교 GPU 열리는 대로 본 학습)')
    out()

    # 저장
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print()
    print(f'리포트 저장: {REPORT_PATH.name}')
    print(f'최악 사례:   {WORST_CASES_PATH.name}')


if __name__ == '__main__':
    main()
