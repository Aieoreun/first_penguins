"""Separate harmful requests from abusive speech using public tutor action labels."""
import json

from core.schema import Label
from .learned_gate import ROOT
from .learned_sections import TutorOnly


def harmful_training_rows():
    return [row for row in map(json.loads, (ROOT / "public/tutor.jsonl").read_text().splitlines())
            if row["labels"]["category"] == "harmful_content"
            and row["labels"]["action"] in {"refuse", "redirect"}]


def train_harmful_classifier(regularization, rows=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    rows = harmful_training_rows() if rows is None else rows
    texts = [next(m["content"] for m in reversed(row["messages"]) if m["role"] == "user") for row in rows]
    labels = [row["labels"]["action"] for row in rows]
    with threadpool_limits(limits=1):
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)
        model = LogisticRegression(C=regularization, max_iter=500, class_weight="balanced")
        model.fit(vectorizer.fit_transform(texts), labels)
    return vectorizer, model


class Pipeline(TutorOnly):
    learned_harmful_c = 1000
    learned_harmful_threshold = .95
    training_source = "bundled_public_categories_evidence_and_harmful_actions"

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self.harmful_vectorizer, self.harmful_classifier = train_harmful_classifier(self.learned_harmful_c)

    async def _classify(self, state):
        probability = self.classifier.predict_proba(self.vectorizer.transform([state.last_user]))[0]
        index = probability.argmax()
        if self.classifier.classes_[index] == "harmful_content" and probability[index] >= self.learned_gate_threshold:
            sub = self.harmful_classifier.predict_proba(self.harmful_vectorizer.transform([state.last_user]))[0]
            sub_index = sub.argmax()
            action = self.harmful_classifier.classes_[sub_index]
            if sub[sub_index] >= self.learned_harmful_threshold and action in {"refuse", "redirect"}:
                state.label = Label.HARMFUL_REQUEST if action == "refuse" else Label.HARMFUL_SPEECH
                state.section = state.chunks[0].section_id if state.chunks else None
                state.debug = {"learned_gate": True, "learned_harmful": True,
                               "confidence": float(probability[index]), "subtype_confidence": float(sub[sub_index])}
                return
        # No confident category + subtype pair: preserve the existing LLM path.
        await super()._classify(state)
