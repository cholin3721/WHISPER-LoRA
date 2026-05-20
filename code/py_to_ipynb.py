"""
Colab-style `# %%` cell-mode Python file을 Jupyter .ipynb로 변환.

사용법:
    python py_to_ipynb.py E2_H2_VAD_Experiment.py
    # -> E2_H2_VAD_Experiment.ipynb 생성

셀 마커 규칙 (Spyder/VSCode 호환):
    # %%                       → 코드 셀 시작
    # %% --- Cell N: 설명 ---   → 코드 셀 (제목 포함, 라벨은 셀 안에 포함)
    # %% [markdown]             → 마크다운 셀 시작
        # 본문은 모두 `#` 주석으로 시작 — 변환 시 `#` 제거하고 본문만 사용

레거시 make_notebook.py(E0 전용)는 E0 ipynb 재생성용으로 그대로 두되,
새 실험(E1, E2 등)은 이 범용 변환기를 사용한다.
"""
import json
import re
import sys
from pathlib import Path


CELL_HEADER_RE = re.compile(r'^# %%(.*)$')


def parse_cells(text):
    """Returns list of (cell_type, source_lines)."""
    cells = []
    current_type = None
    current_lines = []

    for line in text.splitlines():
        m = CELL_HEADER_RE.match(line)
        if m:
            # Flush previous
            if current_type is not None:
                cells.append((current_type, current_lines))
            # Start new cell
            suffix = m.group(1).strip()
            if suffix.startswith('[markdown]'):
                current_type = 'markdown'
            else:
                current_type = 'code'
            current_lines = []
        else:
            if current_type is None:
                # Preamble before first cell — treat as code in cell 0
                current_type = 'code'
            current_lines.append(line)

    if current_type is not None:
        cells.append((current_type, current_lines))

    return cells


def strip_markdown_comments(lines):
    """Markdown 셀에서 각 줄의 선두 `# ` 또는 `#`를 제거."""
    out = []
    for line in lines:
        if line.startswith('# '):
            out.append(line[2:])
        elif line.startswith('#'):
            out.append(line[1:])
        elif line.strip() == '':
            out.append('')
        else:
            out.append(line)
    return out


def trim_blank_edges(lines):
    """앞뒤 빈 줄 제거."""
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def to_source(lines):
    """nbformat의 source 필드 형식 (마지막 줄 빼고 모두 \\n으로 끝남)."""
    if not lines:
        return []
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]


def convert(py_path):
    py_path = Path(py_path)
    text = py_path.read_text(encoding='utf-8')
    parsed = parse_cells(text)

    nb_cells = []
    for cell_type, lines in parsed:
        lines = trim_blank_edges(lines)
        if not lines:
            continue
        if cell_type == 'markdown':
            lines = strip_markdown_comments(lines)
            lines = trim_blank_edges(lines)
            if not lines:
                continue
            nb_cells.append({
                'cell_type': 'markdown',
                'metadata': {},
                'source': to_source(lines),
            })
        else:
            nb_cells.append({
                'cell_type': 'code',
                'metadata': {},
                'source': to_source(lines),
                'outputs': [],
                'execution_count': None,
            })

    nb = {
        'nbformat': 4,
        'nbformat_minor': 0,
        'metadata': {
            'colab': {'provenance': [], 'gpuType': 'T4'},
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
            'language_info': {'name': 'python'},
            'accelerator': 'GPU',
        },
        'cells': nb_cells,
    }

    out_path = py_path.with_suffix('.ipynb')
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'생성: {out_path.name} ({len(nb_cells)}개 셀)')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('사용법: python py_to_ipynb.py <source.py> [<source2.py> ...]')
        sys.exit(1)
    for arg in sys.argv[1:]:
        convert(arg)
