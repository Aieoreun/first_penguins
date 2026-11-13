import re
import unicodedata

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
# Collapse single-syllable Korean spacing obfuscation like "시 스 템" -> "시스템"
_HANGUL_SINGLE_GAP_RE = re.compile(r"(?<=[가-힣])\s+(?=[가-힣](?:\s|$|[.,!?]))")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for matching/classification (never use to mutate verbatim span targets)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def collapse_korean_obfuscation(text: str) -> str:
    """Collapse artificial 1-syllable spacing used in jailbreak/keyword evasion."""
    s = normalize_text(text)
    # Iteratively collapse runs of single Hangul syllables separated by single spaces
    tokens = s.split(" ")
    out: list[str] = []
    buf: list[str] = []
    for tok in tokens:
        if len(tok) == 1 and "\uac00" <= tok <= "\ud7a3":
            buf.append(tok)
        else:
            if len(buf) >= 2:
                out.append("".join(buf))
            elif buf:
                out.extend(buf)
            buf = []
            out.append(tok)
    if len(buf) >= 2:
        out.append("".join(buf))
    elif buf:
        out.extend(buf)
    return " ".join(out)
