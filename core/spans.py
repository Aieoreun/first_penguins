"""모델이 베껴 적은 구절을 후보 답변 안에서 찾는다. docs/sw/02_설계.md 결정 11.

모델은 구절을 조금 다르게 적을 수 있다(띄어쓰기, 문장부호, 한두 글자). 응답에는 모델이 적은 글이
아니라 후보 답변에 실제로 있는 글자를 내보낸다. 찾는 순서:
  1. 그대로 들어 있는가
  2. 공백·문장부호·대소문자를 빼고 비교하면 들어 있는가
  3. 길이를 조금씩 바꿔 가며 창을 한 글자씩 옮겨, 가장 비슷한 자리를 찾는다
가장 비슷한 자리도 기준보다 덜 비슷하면 그 구절은 버린다.
"""
import unicodedata
from difflib import SequenceMatcher
from typing import List, Optional, Sequence, Tuple

# 채점기(grader/scoring.py)가 구간을 비교할 때 빼는 문장부호와 같다
PUNCT = set('.,!?;:\'"“”‘’`()[]{}<>«»·…~-–—/\\')
MIN_SIMILARITY = 0.75  # 이보다 덜 비슷하면 버린다
LENGTH_SLACK = 0.3     # 창 길이를 구절 길이의 ±30%까지 바꿔 본다


def _normalized(text: str) -> Tuple[str, List[int]]:
    """비교용 글자열과, 각 글자가 원문의 몇 번째 글자에서 왔는지."""
    chars, index = [], []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).lower():
            if not c.isspace() and c not in PUNCT:
                chars.append(c)
                index.append(i)
    return "".join(chars), index


def locate(candidate: str, phrase: str) -> Optional[str]:
    """candidate 안에서 phrase에 해당하는 부분을 찾아 candidate의 글자 그대로 돌려준다."""
    phrase = (phrase or "").strip()
    if not candidate or not phrase:
        return None
    if phrase in candidate:
        return phrase
    cand, index = _normalized(candidate)
    target, _ = _normalized(phrase)
    if not cand or not target:
        return None
    at = cand.find(target)
    if at != -1:
        return candidate[index[at]:index[at + len(target) - 1] + 1]
    n = len(target)
    slack = max(1, int(n * LENGTH_SLACK))
    best, best_span = 0.0, None
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(target)
    for size in range(max(1, n - slack), min(len(cand), n + slack) + 1):
        for start in range(0, len(cand) - size + 1):
            matcher.set_seq1(cand[start:start + size])
            if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
                continue
            ratio = matcher.ratio()
            if ratio > best:
                best, best_span = ratio, (start, start + size)
    if best_span is None or best < MIN_SIMILARITY:
        return None
    return candidate[index[best_span[0]]:index[best_span[1] - 1] + 1]


def locate_all(candidate: str, phrases: Sequence[str]) -> List[str]:
    """찾은 구절만, 겹치는 것을 빼고 돌려준다."""
    out: List[str] = []
    for p in phrases:
        if not isinstance(p, str):
            continue
        found = locate(candidate, p)
        if found and not any(found in o or o in found for o in out):
            out.append(found)
    return out
