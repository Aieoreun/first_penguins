"""Alternative scratch text features; no pretrained weights or answer data."""

FEATURES = ("char", "word", "char_wb", "mixed")


def vectorizer(kind):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, make_pipeline
    from sklearn.preprocessing import Normalizer
    common = dict(min_df=2, max_features=60000, sublinear_tf=True)
    if kind in {"char", "char_wb"}:
        return TfidfVectorizer(analyzer=kind, ngram_range=(2, 5), **common)
    if kind == "word":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", **common)
    if kind == "mixed":
        return make_pipeline(FeatureUnion(
            [("char", vectorizer("char")), ("word", vectorizer("word"))],
            transformer_weights={"char": 1.0, "word": 0.5}), Normalizer())
    raise ValueError(f"Unknown text features: {kind}")
