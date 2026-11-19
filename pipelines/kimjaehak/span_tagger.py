"""Scratch character-span model using only bundled public output annotations.

No stored question-to-answer mapping, pretrained weights, or incoming-request
training. Features describe local character and word context, not row IDs.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def char_features(text):
    words = list(re.finditer(r"\S+", text))
    positions = {}
    for n, word in enumerate(words):
        for pos in range(word.start(), word.end()):
            positions[pos] = (n, word)
    result = []
    for pos, char in enumerate(text):
        feature = {"char": char, "digit": char.isdigit(), "space": char.isspace()}
        for offset in (-8, -5, -3, -2, -1, 1, 2, 3, 5, 8):
            at = pos + offset
            feature[f"char{offset}"] = text[at] if 0 <= at < len(text) else "<boundary>"
        for width in (2, 4, 8, 16):
            feature[f"left{width}"] = text[max(0, pos-width):pos]
            feature[f"right{width}"] = text[pos+1:pos+1+width]
        if pos in positions:
            n, word = positions[pos]
            feature.update(word=word.group(), first=pos == word.start(), last=pos == word.end()-1)
            for offset in (-2, -1, 1, 2):
                at = n + offset
                feature[f"word{offset}"] = words[at].group() if 0 <= at < len(words) else "<boundary>"
        result.append(feature)
    return result


def span_mask(text, spans):
    mask = [False] * len(text)
    for span in spans:
        start = 0
        if not span or span not in text:
            raise ValueError("Annotation must be a nonempty candidate substring")
        while (at := text.find(span, start)) >= 0:
            mask[at:at+len(span)] = [True] * len(span)
            start = at + len(span)
    return mask


def decode_mask(text, mask):
    spans, start = [], None
    for i, selected in enumerate([*mask, False]):
        if selected and start is None:
            start = i
        elif not selected and start is not None:
            if phrase := text[start:i].strip():
                spans.append(phrase)
            start = None
    return spans


def public_rows():
    return [row for row in map(json.loads, (ROOT / "public/guard.jsonl").read_text().splitlines())
            if row.get("candidate_answer") is not None]


def refine_spans(candidate, original, predicted, confidence, threshold):
    if not original or not predicted or confidence < threshold:
        return original, False
    a = {i for i, selected in enumerate(span_mask(candidate, original)) if selected and not candidate[i].isspace()}
    b = {i for i, selected in enumerate(span_mask(candidate, predicted)) if selected and not candidate[i].isspace()}
    # Preserve the LLM's grounded verdict and require agreement on the error.
    if not a or not b or len(a & b) / min(len(a), len(b)) < .5:
        return original, False
    return predicted, True


class SpanTagger:
    def __init__(self, rows, regularization=100):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from threadpoolctl import threadpool_limits
        features, labels = [], []
        for row in rows:
            text = row["candidate_answer"]
            features.extend(char_features(text))
            labels.extend(span_mask(text, row["labels"]["unsupported"]))
        with threadpool_limits(limits=1):
            self.vectorizer = DictVectorizer()
            self.model = LogisticRegression(C=regularization, max_iter=500, class_weight="balanced")
            self.model.fit(self.vectorizer.fit_transform(features), labels)

    def predict(self, text):
        if not text:
            return [], 1.0
        probability = self.model.predict_proba(self.vectorizer.transform(char_features(text)))
        mask = self.model.classes_[probability.argmax(axis=1)]
        # Every character must be confident; averages can hide uncertain edges.
        return decode_mask(text, mask), float(probability.max(axis=1).min())
