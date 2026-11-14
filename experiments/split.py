"""공개 세트를 조정용(dev)과 확인용(holdout)으로 나눈다.

프롬프트를 고칠 때는 조정용 문항만 들여다본다. 확인용은 점수만 본다. 문항 내용을 보고 프롬프트를
고치면 그 문항에만 맞춰지기 때문이다. 나누는 기준은 문항 ID의 해시라서 언제 돌려도 같다.
분류 프롬프트에 예시로 들어간 공개 문항은 모두 조정용에 넣는다.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_SHARE = 40  # 확인용 비율(%)


def prompt_example_ids():
    text = (ROOT / "docs" / "sw" / "03_분류_프롬프트_초안.md").read_text(encoding="utf-8")
    return set(re.findall(r"`(pub-[0-9a-f]{8})`", text))


def which(item_id: str, examples=None) -> str:
    examples = prompt_example_ids() if examples is None else examples
    if item_id in examples:
        return "dev"
    bucket = int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 100
    return "holdout" if bucket < HOLDOUT_SHARE else "dev"


def load_items(split: str = "all"):
    examples = prompt_example_ids()
    items = []
    for endpoint in ("tutor", "guard"):
        for line in (ROOT / "public" / f"{endpoint}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if split == "all" or which(item["id"], examples) == split:
                    items.append(item)
    return items


if __name__ == "__main__":
    from collections import Counter
    for name in ("dev", "holdout"):
        items = load_items(name)
        print(name, len(items), dict(Counter(i["endpoint"] for i in items)),
              dict(sorted(Counter(i["labels"]["category"] for i in items).items())))
