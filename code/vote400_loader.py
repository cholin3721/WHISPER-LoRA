"""
VOTE400 데이터셋 로더 + 메타데이터 인덱싱

낭독체(VOTE400_Read)와 대화체(VOTE400_Dialog) 둘 다 지원.
폴더명·파일명에서 메타데이터 추출하여 dict 리스트로 반환.

낭독체 구조:
  VOTE400_Read/Audio/<REGION><GROUP>/PID_<ID>_<DATE>_<NO>_<REGION>.wav
  예: VOTE400_Read/Audio/DG1/PID_000000026_20191009_00001_DG.wav
  → region=DG, group=1, pid=PID_000000026

대화체 구조:
  VOTE400_Dialog/Audio/<REGION>/<PID>_<GENDER>_<AGE>_<REGION>_<YEAR>-<DATE>-<TIME>-<SID>.wav
  예: VOTE400_Dialog/Audio/CB/AOR_F_84_CB_2018-0814-1756-07.wav
  → region=CB, pid=AOR, gender=F, age=84

CLI 사용:
  python code/vote400_loader.py --root /path/to/VOTE400 --out results/vote400_index.csv
"""
import argparse
import csv
import re
from pathlib import Path

READ_FILENAME_RE = re.compile(r'^PID_(\d+)_(\d+)_(\d+)_([A-Z]+)\.wav$')
DIALOG_FILENAME_RE = re.compile(
    r'^([A-Z]+)_([FM])_(\d+)_([A-Z]+)_(\d{4})-(\d{4})-(\d{4})-(\d+)\.wav$'
)
GROUP_FOLDER_RE = re.compile(r'^([A-Z]+)(\d+)$')   # DG1, GN3 ...


def clean_text(t):
    """VOTE400 transcription 정제. AI허브 다르게 태그 없음, strip만."""
    return ' '.join(t.split())


def load_read(read_root):
    """낭독체 로드. read_root = .../VOTE400_Read"""
    items = []
    audio_root = Path(read_root) / 'Audio'
    trans_root = Path(read_root) / 'Transcriptions'

    for folder in sorted(audio_root.iterdir()):
        if not folder.is_dir():
            continue
        m = GROUP_FOLDER_RE.match(folder.name)
        if not m:
            continue
        region, group = m.group(1), int(m.group(2))

        for wav in sorted(folder.glob('*.wav')):
            fm = READ_FILENAME_RE.match(wav.name)
            if not fm:
                continue
            pid_num, date, utt_no, _file_region = fm.groups()
            pid = f'PID_{pid_num}'

            txt_path = trans_root / folder.name / (wav.stem + '.txt')
            if not txt_path.exists():
                continue
            with open(txt_path, 'r', encoding='utf-8') as f:
                transcript = clean_text(f.read())
            if not transcript:
                continue

            items.append({
                'mode': 'read',
                'region': region,
                'group': group,
                'pid': pid,
                'date': date,
                'utt_no': utt_no,
                'wav_path': str(wav),
                'txt_path': str(txt_path),
                'transcript': transcript,
            })
    return items


def load_dialog(dialog_root):
    """대화체 로드. dialog_root = .../VOTE400_Dialog"""
    items = []
    audio_root = Path(dialog_root) / 'Audio'
    trans_root = Path(dialog_root) / 'Transcriptions'

    for folder in sorted(audio_root.iterdir()):
        if not folder.is_dir():
            continue
        region = folder.name

        for wav in sorted(folder.glob('*.wav')):
            fm = DIALOG_FILENAME_RE.match(wav.name)
            if not fm:
                continue
            pid, gender, age, _region, year, mmdd, time, sid = fm.groups()

            txt_path = trans_root / folder.name / (wav.stem + '.txt')
            if not txt_path.exists():
                continue
            with open(txt_path, 'r', encoding='utf-8') as f:
                transcript = clean_text(f.read())
            if not transcript:
                continue

            items.append({
                'mode': 'dialog',
                'region': region,
                'group': None,             # 대화체엔 그룹 번호 없음
                'pid': pid,
                'gender': gender,
                'age': int(age),
                'date': f'{year}{mmdd}',
                'time': time,
                'sid': sid,
                'wav_path': str(wav),
                'txt_path': str(txt_path),
                'transcript': transcript,
            })
    return items


def write_index(items, out_csv):
    """CSV 인덱스 저장."""
    if not items:
        print(f'⚠️  로드된 항목 없음 — {out_csv} 안 만듦')
        return
    fields = sorted({k for it in items for k in it.keys()})
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for it in items:
            writer.writerow(it)
    print(f'✅ 저장: {out_csv} ({len(items):,}개)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='VOTE400 폴더 경로 (e.g. /storage/cholin2/whisper/data/VOTE400)')
    ap.add_argument('--mode', choices=['read', 'dialog', 'both'], default='both')
    ap.add_argument('--out-dir', default='results',
                    help='결과 CSV 저장 폴더 (프로젝트 루트 기준)')
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(__file__).resolve().parent.parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ('read', 'both'):
        print('🔍 낭독체 로드 중...')
        read_items = load_read(root / 'VOTE400_Read')
        print(f'   {len(read_items):,}개')
        write_index(read_items, out_dir / 'vote400_read_index.csv')

        # 그룹별 / 화자별 통계
        print('\n=== 낭독체 그룹별 분포 ===')
        from collections import defaultdict
        by_group = defaultdict(lambda: {'utts': 0, 'pids': set()})
        for it in read_items:
            key = f"{it['region']}{it['group']}"
            by_group[key]['utts'] += 1
            by_group[key]['pids'].add(it['pid'])
        for k in sorted(by_group.keys()):
            v = by_group[k]
            print(f'  {k}: {v["utts"]:,}발화 / {len(v["pids"])}명')

    if args.mode in ('dialog', 'both'):
        print('\n🔍 대화체 로드 중...')
        dialog_items = load_dialog(root / 'VOTE400_Dialog')
        print(f'   {len(dialog_items):,}개')
        write_index(dialog_items, out_dir / 'vote400_dialog_index.csv')

        # 나이대별 분포 (대화체엔 나이 있음)
        print('\n=== 대화체 나이대별 분포 ===')
        from collections import Counter
        age_bins = Counter()
        for it in dialog_items:
            a = it['age']
            if a < 65:
                age_bins['<65'] += 1
            elif a < 75:
                age_bins['65~74'] += 1
            elif a < 85:
                age_bins['75~84'] += 1
            else:
                age_bins['85+'] += 1
        for k in ['<65', '65~74', '75~84', '85+']:
            print(f'  {k}: {age_bins[k]:,}개')


if __name__ == '__main__':
    main()
