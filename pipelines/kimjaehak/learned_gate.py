"""Optional scratch classifier trained only on bundled public categories.

Public scores for this variant are in-sample, not unseen validation. No question
lookup, reference answer, request ID, or incoming-request storage is used.
Install requirements-jh-research.txt for the optional sklearn dependencies.
"""
import json
from pathlib import Path

from core.schema import parse_label
from .adaptive import Pipeline as Adaptive


ROOT = Path(__file__).resolve().parents[2]


def train_public_classifier(regularization):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    rows = [json.loads(line) for name in ("tutor", "guard")
            for line in (ROOT / "public" / f"{name}.jsonl").read_text().splitlines()]
    texts = [next(m["content"] for m in reversed(row["messages"]) if m["role"] == "user") for row in rows]
    labels = [row["labels"]["category"] for row in rows]
    with threadpool_limits(limits=1):
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                     max_features=60000, sublinear_tf=True)
        model = LogisticRegression(C=regularization, max_iter=500, class_weight="balanced")
        model.fit(vectorizer.fit_transform(texts), labels)
    return vectorizer, model


class Pipeline(Adaptive):
    learned_gate_threshold = .995
    learned_gate_c = 1000
    training_source = "bundled_public_categories"

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        # Fit before the server is ready, never during a request.
        self.vectorizer, self.classifier = train_public_classifier(self.learned_gate_c)

    async def _classify(self, state):
        probability = self.classifier.predict_proba(self.vectorizer.transform([state.last_user]))[0]
        index = probability.argmax()
        category = self.classifier.classes_[index]
        label = parse_label(str(category))
        # Public guard labels do not distinguish harmful request from speech.
        # Leave that subtype to the existing LLM pipeline for every request.
        if probability[index] >= self.learned_gate_threshold and label is not None and category != "harmful_content":
            state.label = label
            state.section = state.chunks[0].section_id if state.chunks else None
            state.debug = {"learned_gate": True, "confidence": float(probability[index])}
            return
        await super()._classify(state)


class Relaxed(Pipeline):
    """More public fitting; compare separately against grouped OOF diagnostics."""
    learned_gate_threshold = .98


class Broad(Pipeline):
    learned_gate_threshold = .95
