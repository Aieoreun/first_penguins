"""Normalize only the tutor category model's view; retain original requests."""
from copy import copy
import re

from .focused_excerpt import Selected as SelectedFocused


# Protect source syntax and literal identifiers before touching prose. Unclosed
# fences/quotes are protected too: an incomplete example is still user data.
_PROTECTED = re.compile(
    r'```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|'
    r'"""[\s\S]*?(?:"""|$)|\'\'\'[\s\S]*?(?:\'\'\'|$)|'
    r'`[^`\n]*(?:`|(?=\n)|$)|'
    r'"(?:\\.|[^"\\\n])*(?:"|(?=\n)|$)|'
    r"'(?:\\.|[^'\\\n])*(?:'|(?=\n)|$)|"
    r'“[^”\n]*(?:”|(?=\n)|$)|‘[^’\n]*(?:’|(?=\n)|$)|'
    r'(?:https?://|www\.)[^\s<>"\'`]+|'
    r'[^\s<>"\'`@]+@[^\s<>"\'`@]+|'
    r'(?:아이디|ID|id|계정|이메일|전화번호|계좌번호|주민등록번호|학번)\s*[:=]\s*[^\s,;]+|'
    r'(?<!\w)\+?\d[\d ()-]{5,}\d(?!\w)|'
    r'(?m:^(?: {4}|\t)[^\n]+$)|'
    r'(?m:^[ \t]*(?:[A-Za-z_]\w*\s*=|(?:for|if|while|def|class|import|from|return)\b|'
    r'[A-Za-z_]\w*(?:\.\w+)*\s*\()[^\n]*$)'
)
_STANDALONE = re.compile(r'(?<!\S)(?:ㅋ{2,}|ㅎ{2,}|ㅠ{2,}|ㅜ{2,})(?!\S)')


def normalize_category_query(text):
    """Drop standalone repeated jamo in prose, preserving literals and signal."""
    pieces = []
    def prose(start, end):
        def replace(match):
            left, right = start + match.start(), start + match.end()
            standalone = ((left == 0 or text[left - 1].isspace())
                          and (right == len(text) or text[right].isspace()))
            return ' ' if standalone else match.group()
        return re.sub(r'[ \t\r\f\v]+', ' ', _STANDALONE.sub(replace, text[start:end]))

    start = 0
    for match in _PROTECTED.finditer(text):
        pieces.append(prose(start, match.start()))
        pieces.append(match.group())
        start = match.end()
    pieces.append(prose(start, len(text)))
    # Only trim prose boundaries: never strip indentation from protected code.
    pieces[0] = pieces[0].lstrip()
    pieces[-1] = pieces[-1].rstrip()
    result = ''.join(pieces)
    return result if result.strip() else text


class _CategoryView:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def transform(self, texts):
        return self.vectorizer.transform([normalize_category_query(text) for text in texts])


class Pipeline(SelectedFocused):
    tutor_category_normalization = 'protected_standalone_jamo'

    def _new_state(self, body, guard):
        state = super()._new_state(body, guard)
        state._normalized_tutor_guard = guard
        return state

    async def _classify(self, state):
        # This marker survives parent classifiers replacing state.debug. States
        # not made by this candidate conservatively keep the original view.
        if getattr(state, '_normalized_tutor_guard', True):
            return await super()._classify(state)
        request_pipeline = copy(self)
        request_pipeline.vectorizer = _CategoryView(self.vectorizer)
        # Only this copy's category-vectorizer reference differs. Section,
        # harmful subtype, guard, search and LLM consume the unchanged state.
        return await super(Pipeline, request_pipeline)._classify(state)
