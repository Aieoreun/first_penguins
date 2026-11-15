"""Optional public-trained section classifier; no stored question/answer table.

This is an in-sample fitting experiment. Grouped OOF diagnostics are reported
separately; low confidence always leaves the ordinary search result intact.
"""
import json

from .learned_gate import Broad, ROOT


def train_section_classifier(regularization):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    rows = [row for row in map(json.loads, (ROOT / "public/tutor.jsonl").read_text().splitlines())
            if len(row["labels"].get("evidence") or []) == 1]
    texts = [next(m["content"] for m in reversed(row["messages"]) if m["role"] == "user") for row in rows]
    labels = [row["labels"]["evidence"][0] for row in rows]
    with threadpool_limits(limits=1):
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1,
                                     max_features=60000, sublinear_tf=True)
        model = LogisticRegression(C=regularization, max_iter=500, class_weight="balanced")
        model.fit(vectorizer.fit_transform(texts), labels)
    return vectorizer, model


class Pipeline(Broad):
    learned_section_c = 1000
    learned_section_threshold = .98
    learned_section_require_retrieved = True
    learned_section_use_for_output = True

    def __init__(self, searcher, llm):
        super().__init__(searcher, llm)
        self.section_vectorizer, self.section_classifier = train_section_classifier(self.learned_section_c)
        self._known_sections = set(searcher.section_ids())

    async def _classify(self, state):
        await super()._classify(state)
        if state.has_candidate and not self.learned_section_use_for_output:
            return
        probability = self.section_classifier.predict_proba(self.section_vectorizer.transform([state.last_user]))[0]
        index = probability.argmax()
        section = str(self.section_classifier.classes_[index])
        candidates = {chunk.section_id for chunk in state.chunks} if self.learned_section_require_retrieved else self._known_sections
        if probability[index] >= self.learned_section_threshold and section in candidates and section in self._known_sections:
            state.section = section
            state.debug["learned_section"] = True


class WholeKB(Pipeline):
    learned_section_require_retrieved = False


class TutorOnly(WholeKB):
    learned_section_use_for_output = False


class WideTutor(TutorOnly):
    learned_section_threshold = .95


class Stronger(TutorOnly):
    learned_gate_c = 3000


class StrongerConservative(Stronger):
    learned_gate_threshold = .98
