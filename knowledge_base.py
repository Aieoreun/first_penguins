from pathlib import Path

KB_ROOT = Path(__file__).resolve().parent / "kb"


def load_knowledge_base():
    sections = {}

    for path in sorted(KB_ROOT.glob("*/*.md")):
        section_id = path.stem
        if section_id in sections:
            raise ValueError(f"중복 섹션: {section_id}")

        tables = sorted(path.parent.glob(f"{section_id}-table*.csv"))
        sources = [path, *tables]

        sections[section_id] = {
            "section_id": section_id,
            "subject": path.parent.name,
            "text": "\n\n".join(
                f"원본 파일: {source.name}\n"
                + source.read_text(encoding="utf-8")
                for source in sources
            ),
            "sources": [
                str(source.relative_to(KB_ROOT))
                for source in sources
            ],
        }

    if not sections:
        raise ValueError("수업 자료를 찾지 못했습니다.")

    return sections