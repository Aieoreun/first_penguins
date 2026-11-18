"""Scratch category models; only bundled public data and organizer embeddings.

No incoming-request storage, lookup table, or pretrained classifier. Keep this
module independent of pipeline imports so offline CV cannot invoke the API.
"""
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

KINDS = ('char_lr', 'dense_lr', 'char_mlp', 'dense_mlp')


def public_data():
    from .input_examples import PolicyExamples
    examples = PolicyExamples(True)  # validates public IDs, hashes and vectors
    return examples.rows, examples.texts, np.asarray(examples.vectors, dtype=np.float64), examples.model


class CategoryModel:
    def __init__(self, kind, seed=20261118, width=32, alpha=.1, c=3000):
        if kind not in KINDS:
            raise ValueError('Unknown category model')
        self.kind, self.seed, self.width, self.alpha, self.c = kind, seed, width, alpha, c
        self.vectorizer = None

    def fit(self, texts, labels, vectors=None):
        started = time.monotonic()
        if self.kind.startswith('char'):
            self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5),
                min_df=2, max_features=60000, sublinear_tf=True)
            x = self.vectorizer.fit_transform(texts)
        else:
            x = self._vectors(vectors, len(texts))
        if self.kind.endswith('lr'):
            self.model = LogisticRegression(C=self.c, max_iter=1000, class_weight='balanced')
            fit_kwargs = {}
        else:
            self.model = MLPClassifier(hidden_layer_sizes=(self.width,), activation='relu',
                solver='lbfgs', alpha=self.alpha, max_iter=600, max_fun=30000,
                random_state=self.seed, tol=1e-5, early_stopping=False)
            fit_kwargs = {'sample_weight': compute_sample_weight('balanced', labels)}
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            self.model.fit(x, labels, **fit_kwargs)
        self.diagnostics = {'fit_seconds': time.monotonic() - started,
            'feature_count': x.shape[1], 'convergence_warnings': sum(
                issubclass(w.category, ConvergenceWarning) for w in caught),
            'iterations': np.asarray(self.model.n_iter_).tolist(),
            'parameters': (sum(v.size for v in self.model.coefs_) +
                           sum(v.size for v in self.model.intercepts_))
                           if self.kind.endswith('mlp') else self.model.coef_.size + self.model.intercept_.size}
        return self

    @staticmethod
    def _vectors(vectors, count):
        x = np.asarray(vectors, dtype=np.float64)
        if x.shape != (count, 1536) or not np.isfinite(x).all():
            raise ValueError('Expected finite organizer embeddings, shape (n,1536)')
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError('Empty embedding vector')
        return x / norms

    def probabilities(self, texts, vectors=None):
        x = self.vectorizer.transform(texts) if self.vectorizer is not None else self._vectors(vectors, len(texts))
        return self.model.predict_proba(x)

    @property
    def classes(self):
        return self.model.classes_
