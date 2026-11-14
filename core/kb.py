"""교재(kb/)를 소단원 조각으로 읽는다. docs/sw/02_설계.md 결정 6.

조각 하나 = 섹션 제목(# ID 제목) + 소단원 제목(##) + 그 아래 본문과 Q/A.
표 파일(<섹션 ID>-table<n>.csv)은 그 섹션의 조각 하나로 넣는다.
HTML 파일은 같은 단원을 다른 형식으로 준 것이라 읽지 않는다(SPEC.md 1절).
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

_TABLE_NAME = re.compile(r"^(?P<section>.+)-table\d+$")


@dataclass(frozen=True)
class KBChunk:
    section_id: str  # 인용에 쓰는 섹션 ID. 예: PY-01-01
    subject: str     # 과목 폴더 이름. python, prompt, finance
    title: str       # 섹션 제목. 예: 프로그래밍과 파이썬
    heading: str     # 소단원 제목. 표 조각은 "표"
    text: str        # 색인하고 모델에 넘기는 글(제목 포함)


def _split_markdown(section_id: str, subject: str, raw: str) -> List[KBChunk]:
    lines = raw.strip().split("\n")
    title = lines[0].lstrip("# ").strip()
    if title.startswith(section_id):
        title = title[len(section_id):].strip()
    parts, heading, body = [], None, []
    for line in lines[1:]:
        if line.startswith("## "):
            if heading is not None or "".join(body).strip():
                parts.append((heading or "", body))
            heading, body = line[3:].strip(), []
        else:
            body.append(line)
    if heading is not None or "".join(body).strip():
        parts.append((heading or "", body))
    chunks = []
    for heading, body in parts:
        head = f"# {section_id} {title}\n" + (f"## {heading}\n" if heading else "")
        chunks.append(KBChunk(section_id, subject, title, heading, head + "\n".join(body).strip()))
    return chunks


def load_chunks(kb_dir) -> List[KBChunk]:
    """kb_dir 아래 과목 폴더를 읽어 조각 목록을 만든다. 순서는 파일 이름순이다."""
    kb_dir = Path(kb_dir)
    chunks: List[KBChunk] = []
    for subject_dir in sorted(p for p in kb_dir.iterdir() if p.is_dir()):
        titles = {}
        for path in sorted(subject_dir.glob("*.md")):
            found = _split_markdown(path.stem, subject_dir.name, path.read_text(encoding="utf-8"))
            if found:
                titles[path.stem] = found[0].title
            chunks.extend(found)
        for path in sorted(subject_dir.glob("*.csv")):
            m = _TABLE_NAME.match(path.stem)
            section_id = m.group("section") if m else path.stem
            title = titles.get(section_id, "")
            text = f"# {section_id} {title}\n## 표\n" + path.read_text(encoding="utf-8").strip()
            chunks.append(KBChunk(section_id, subject_dir.name, title, "표", text))
    return chunks
