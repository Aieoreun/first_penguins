"""Input guard classifier: CPU TF-IDF + rule layer with gpt-5-nano grey-zone escalation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

from app.config import ClientPool, Settings
from app.normalize import collapse_korean_obfuscation, normalize_text
from app.prompts import (
    BLOCK_CATEGORIES,
    CATEGORIES,
    GUARD_INPUT_JSON_SCHEMA,
    GUARD_INPUT_SYSTEM_PREFIX,
)

logger = logging.getLogger("first_penguins.guard_classifier")

# Unambiguous low-signal patterns (after normalize + collapse_korean_obfuscation)
_HAS_MEANINGFUL_TEXT_RE = re.compile(r"[가-힣]|[a-zA-Z]{2,}|\d{2,}")
_PURE_LOW_SIGNAL_RE = re.compile(
    r"^[\s\.,!?~`@#$%^&*()_\-+=\[\]{};:'\"<>/\\|ㄱ-ㅎㅏ-ㅣㅋㅎㅠㅜㅇㄴㅁ]+$"
)
_KEYBOARD_MASH_RE = re.compile(
    r"^(?:ㅁㄴㅇㄹ|ㅂㅈㄷㄱ|ㅋㅌㅊㅍ|asdf|qwer|zxcv|test|hi|hello|\?+|!+|\.+|ㅋ+|ㅎ+|ㅠ+|ㅜ+)+$",
    re.IGNORECASE,
)

# High-precision jailbreak / prompt-injection signatures (applied after collapse_korean_obfuscation)
_JAILBREAK_PATTERNS = (
    re.compile(r"시스템\s*프롬프트"),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"(?:이전|기존|위의)\s*(?:지시|지침|명령|규칙)(?:을|를)?\s*(?:모두|싹|다)?\s*무시"),
    re.compile(r"제한\s*없는\s*(?:ai|인공지능|모델|챗봇)", re.IGNORECASE),
    re.compile(r"(?:탈옥|jailbreak)", re.IGNORECASE),
    re.compile(r"\bDAN\b"),
    re.compile(r"(?:개발자\s*모드|developer\s*mode)", re.IGNORECASE),
    re.compile(r"(?:프롬프트|내부\s*지침|시스템\s*규칙)(?:을|를)?\s*(?:출력|공개|알려|보여)"),
)


@dataclass
class _TrainExample:
    example_id: str
    text: str
    last_user: str
    decision: str
    category: str
    messages: list[dict[str, str]]
    weight: float = 1.0


def _extract_last_user_and_context(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Return (collapsed_last_user, collapsed_context) from conversation messages."""
    if not messages:
        return "", ""
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx == -1:
        last_user_idx = len(messages) - 1

    last_user = collapse_korean_obfuscation(messages[last_user_idx].get("content", ""))
    ctx_parts: list[str] = []
    for i, m in enumerate(messages):
        if i == last_user_idx:
            continue
        role = m.get("role", "user")
        content = collapse_korean_obfuscation(m.get("content", ""))
        if content:
            ctx_parts.append(f"{role}: {content}")
    ctx = " | ".join(ctx_parts[-4:])
    return last_user, ctx


def _build_classifier_text(messages: list[dict[str, str]]) -> str:
    """Build normalized text emphasizing the final user turn while retaining context."""
    last_user, ctx = _extract_last_user_and_context(messages)
    if ctx:
        return f"[LAST] {last_user} [LAST] {last_user} [CTX] {ctx}"
    return f"[LAST] {last_user}"


def _format_messages_for_llm(messages: list[dict[str, str]], max_turns: int = 6) -> str:
    lines: list[str] = []
    recent = messages[-max_turns:]
    for m in recent:
        role = m.get("role", "user")
        content = collapse_korean_obfuscation(m.get("content", ""))
        lines.append(f"{role}: {content[:800]}")
    last_user, _ = _extract_last_user_and_context(messages)
    if last_user:
        lines.append(f"\n[마지막 학습자 발화 (최종 판정 기준)]\n{last_user[:800]}")
    return "\n".join(lines)


class GuardClassifier:
    """Two-stage input guardrail: fast CPU TF-IDF + rules, escalating grey-zone to gpt-5-nano."""

    def __init__(self, settings: Settings, pool: ClientPool):
        self.settings = settings
        self.pool = pool

        self._train_examples: list[_TrainExample] = []
        self._vectorizer: Optional[FeatureUnion] = None
        self._train_tfidf: Any = None
        self._block_clf: Any = None
        self._cat_clf: Any = None
        self._block_threshold: float = 0.50

        # Optional embedding-space classifier + Mahalanobis OOD detector
        self._embed_block_clf: Any = None
        self._embed_cat_clf: Any = None
        self._ood_centroids: Optional[dict[str, np.ndarray]] = None
        self._ood_precision: Optional[np.ndarray] = None

        self._loaded = False

    def load_or_train(self) -> None:
        """Load public/guard.jsonl (+ public/tutor.jsonl) and train CPU classifiers."""
        examples: list[_TrainExample] = []

        guard_path = self.settings.public_dir / "guard.jsonl"
        if guard_path.exists():
            examples.extend(self._load_jsonl(guard_path, weight=2.0, primary=True))

        tutor_path = self.settings.public_dir / "tutor.jsonl"
        if tutor_path.exists():
            examples.extend(self._load_jsonl(tutor_path, weight=1.0, primary=False))

        self._train_examples = examples
        if not examples:
            logger.warning("No training examples found in %s; GuardClassifier will rely on rules/LLM.", self.settings.public_dir)
            self._loaded = True
            return

        texts = [ex.text for ex in examples]
        y_block = np.array([1 if ex.decision == "block" else 0 for ex in examples], dtype=np.int64)
        y_cat = np.array([ex.category for ex in examples])
        weights = np.array([ex.weight for ex in examples], dtype=np.float64)

        self._vectorizer = self._make_vectorizer()
        X = self._vectorizer.fit_transform(texts)
        self._train_tfidf = X

        # Train binary block classifier (calibrated)
        min_block_count = int(min(np.sum(y_block == 0), np.sum(y_block == 1)))
        base_block = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=42)
        if min_block_count >= 5:
            self._block_clf = CalibratedClassifierCV(base_block, cv=5, method="sigmoid")
            self._block_clf.fit(X, y_block, sample_weight=weights)
        elif min_block_count >= 2:
            self._block_clf = CalibratedClassifierCV(base_block, cv=min_block_count, method="sigmoid")
            self._block_clf.fit(X, y_block, sample_weight=weights)
        else:
            base_block.fit(X, y_block, sample_weight=weights)
            self._block_clf = base_block

        # Train 10-class category classifier
        _, cat_counts = np.unique(y_cat, return_counts=True)
        min_cat_count = int(np.min(cat_counts)) if len(cat_counts) else 0
        base_cat = LogisticRegression(C=3.0, class_weight="balanced", max_iter=1000, random_state=42)
        if min_cat_count >= 5:
            self._cat_clf = CalibratedClassifierCV(base_cat, cv=5, method="sigmoid")
            self._cat_clf.fit(X, y_cat, sample_weight=weights)
        elif min_cat_count >= 3:
            self._cat_clf = CalibratedClassifierCV(base_cat, cv=min_cat_count, method="sigmoid")
            self._cat_clf.fit(X, y_cat, sample_weight=weights)
        else:
            base_cat.fit(X, y_cat, sample_weight=weights)
            self._cat_clf = base_cat

        # Calibrate block threshold via out-of-fold CV to maximize Block F1
        self._block_threshold = self._calibrate_block_threshold(texts, y_block, weights)

        # Optional precomputed embeddings
        self._load_precomputed_embeddings(examples)
        self._loaded = True

    @staticmethod
    def _make_vectorizer() -> FeatureUnion:
        return FeatureUnion([
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    sublinear_tf=True,
                    min_df=2,
                    max_features=40000,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    max_features=20000,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
        ])

    @staticmethod
    def _load_jsonl(path: Path, weight: float, primary: bool) -> list[_TrainExample]:
        out: list[_TrainExample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                labels = row.get("labels") or {}
                category = labels.get("category", "normal")
                if category not in CATEGORIES:
                    continue
                decision = labels.get("decision")
                if decision not in ("allow", "block"):
                    decision = "block" if category in BLOCK_CATEGORIES else "allow"
                msgs = row.get("messages") or []
                last_user, _ = _extract_last_user_and_context(msgs)
                out.append(
                    _TrainExample(
                        example_id=str(row.get("id", "")),
                        text=_build_classifier_text(msgs),
                        last_user=last_user,
                        decision=decision,
                        category=category,
                        messages=msgs,
                        weight=weight if primary else 1.0,
                    )
                )
        return out

    def _calibrate_block_threshold(
        self, texts: list[str], y_block: np.ndarray, weights: np.ndarray
    ) -> float:
        min_cls = int(min(np.sum(y_block == 0), np.sum(y_block == 1)))
        if min_cls < 5 or len(texts) < 20:
            return 0.50

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(texts), dtype=np.float64)
        texts_arr = np.array(texts, dtype=object)

        for train_idx, val_idx in skf.split(texts_arr, y_block):
            vec = self._make_vectorizer()
            X_tr = vec.fit_transform(texts_arr[train_idx].tolist())
            X_va = vec.transform(texts_arr[val_idx].tolist())
            clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_tr, y_block[train_idx], sample_weight=weights[train_idx])
            oof_probs[val_idx] = clf.predict_proba(X_va)[:, 1]

        primary_mask = weights > 1.0
        eval_y = y_block[primary_mask] if np.any(primary_mask) else y_block
        eval_p = oof_probs[primary_mask] if np.any(primary_mask) else oof_probs

        best_thr = 0.50
        best_f1 = -1.0
        for thr in np.linspace(0.20, 0.80, 61):
            preds = (eval_p >= thr).astype(np.int64)
            score = f1_score(eval_y, preds, zero_division=0)
            if score > best_f1 or (score == best_f1 and abs(thr - 0.50) < abs(best_thr - 0.50)):
                best_f1 = float(score)
                best_thr = float(round(thr, 3))
        return best_thr

    def _load_precomputed_embeddings(self, examples: list[_TrainExample]) -> None:
        npz_path = self.settings.assets_dir / "guard_embeddings.npz"
        if not npz_path.exists():
            return
        try:
            data = np.load(npz_path, allow_pickle=False)
            if "embeddings" not in data:
                return
            embs = np.asarray(data["embeddings"], dtype=np.float64)
            if "decisions" in data and "categories" in data:
                y_blk = np.array([1 if str(d) == "block" else 0 for d in data["decisions"]], dtype=np.int64)
                y_cat = np.array([str(c) for c in data["categories"]])
            elif len(embs) == len(examples):
                y_blk = np.array([1 if ex.decision == "block" else 0 for ex in examples], dtype=np.int64)
                y_cat = np.array([ex.category for ex in examples])
            else:
                return

            self._embed_block_clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=1000, random_state=42)
            self._embed_block_clf.fit(embs, y_blk)

            self._embed_cat_clf = LogisticRegression(C=3.0, class_weight="balanced", max_iter=1000, random_state=42)
            self._embed_cat_clf.fit(embs, y_cat)

            centroids: dict[str, np.ndarray] = {}
            centered: list[np.ndarray] = []
            for cat in np.unique(y_cat):
                mask = y_cat == cat
                mu = embs[mask].mean(axis=0)
                centroids[str(cat)] = mu
                centered.append(embs[mask] - mu)
            all_centered = np.vstack(centered)
            cov = np.cov(all_centered, rowvar=False) + 1e-4 * np.eye(embs.shape[1])
            self._ood_centroids = centroids
            self._ood_precision = np.linalg.pinv(cov)
        except Exception as exc:
            logger.warning("Failed to load guard_embeddings.npz: %s", exc)

    def _mahalanobis_min_distance(self, emb: np.ndarray) -> float:
        if not self._ood_centroids or self._ood_precision is None:
            return 0.0
        dists = []
        for mu in self._ood_centroids.values():
            diff = emb - mu
            d2 = float(diff @ self._ood_precision @ diff)
            dists.append(np.sqrt(max(d2, 0.0)))
        return min(dists) if dists else 0.0

    def _apply_rules(self, last_user: str) -> tuple[Optional[tuple[str, str, float]], bool]:
        """Return (immediate_result, jailbreak_boost_flag)."""
        stripped = last_user.strip()
        if not stripped:
            return ("allow", "low_signal_input", 0.99), False

        if _PURE_LOW_SIGNAL_RE.match(stripped) or _KEYBOARD_MASH_RE.match(stripped):
            return ("allow", "low_signal_input", 0.98), False

        if not _HAS_MEANINGFUL_TEXT_RE.search(stripped) and len(stripped) <= 12:
            return ("allow", "low_signal_input", 0.96), False

        for pat in _JAILBREAK_PATTERNS:
            if pat.search(stripped):
                return None, True

        return None, False

    def _retrieve_knn(self, query_text: str, k: int = 4) -> list[dict[str, Any]]:
        if self._vectorizer is None or self._train_tfidf is None or not self._train_examples:
            return []
        q_vec = self._vectorizer.transform([query_text])
        sims = (self._train_tfidf @ q_vec.T).toarray().ravel()
        top_idx = np.argsort(sims)[::-1][:k]
        out: list[dict[str, Any]] = []
        for idx in top_idx:
            ex = self._train_examples[int(idx)]
            out.append({
                "id": ex.example_id,
                "messages": ex.messages,
                "decision": ex.decision,
                "category": ex.category,
                "similarity": float(sims[int(idx)]),
            })
        return out

    async def _llm_classify(
        self,
        messages: list[dict[str, str]],
        similar_examples: Optional[list[dict[str, Any]]],
        query_text: str,
    ) -> Optional[tuple[str, str, float]]:
        if self.pool.chat_client is None:
            return None

        exemplars = similar_examples if similar_examples is not None else self._retrieve_knn(query_text, k=4)
        ex_lines: list[str] = []
        for i, ex in enumerate(exemplars[:4], start=1):
            ex_msgs = ex.get("messages") or []
            last_u = ""
            for m in reversed(ex_msgs):
                if m.get("role") == "user":
                    last_u = normalize_text(m.get("content", ""))[:200]
                    break
            dec = ex.get("decision") or (ex.get("labels") or {}).get("decision", "")
            cat = ex.get("category") or (ex.get("labels") or {}).get("category", "")
            if last_u and dec and cat:
                ex_lines.append(f"{i}. 발화: {last_u} -> decision={dec}, category={cat}")

        user_parts: list[str] = []
        if ex_lines:
            user_parts.append("[유사 판정 예시]\n" + "\n".join(ex_lines))
        user_parts.append("[현재 대화 이력]\n" + _format_messages_for_llm(messages))
        user_prompt = "\n\n".join(user_parts)

        try:
            async with self.pool.semaphore:
                resp = await self.pool.chat_client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": GUARD_INPUT_SYSTEM_PREFIX},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_schema", "json_schema": GUARD_INPUT_JSON_SCHEMA},
                    extra_body={"reasoning_effort": "low"},
                    max_completion_tokens=1024,
                )
            choice = resp.choices[0]
            raw = (choice.message.content or "").strip()
            if not raw:
                logger.warning(
                    "GuardClassifier LLM empty response: finish_reason=%s",
                    getattr(choice, "finish_reason", None),
                )
                return None
            parsed = json.loads(raw)
            category = parsed.get("category")
            decision = parsed.get("decision")
            if category in CATEGORIES:
                # Strictly enforce the 6-allow / 4-block partition from category
                reconciled_dec = "block" if category in BLOCK_CATEGORIES else "allow"
                return reconciled_dec, category, 0.90
            if decision in ("allow", "block"):
                fallback_cat = "harmful_content" if decision == "block" else "normal"
                return decision, fallback_cat, 0.80
        except Exception as exc:
            logger.warning("GuardClassifier LLM fallback failed: %s", type(exc).__name__)
        return None

    async def classify(
        self,
        messages: list[dict[str, str]],
        similar_examples: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, str, float]:
        """Classify conversation input into (decision, category, confidence)."""
        if not self._loaded:
            self.load_or_train()

        last_user, _ = _extract_last_user_and_context(messages)
        immediate, jb_boost = self._apply_rules(last_user)
        if immediate is not None:
            return immediate

        query_text = _build_classifier_text(messages)

        # CPU TF-IDF prediction
        p_block = 0.50
        top_cat = "normal"
        top_cat_prob = 0.50
        cat_margin = 0.0

        if self._vectorizer is not None and self._block_clf is not None and self._cat_clf is not None:
            X = self._vectorizer.transform([query_text])
            p_block = float(self._block_clf.predict_proba(X)[0, 1])

            cat_probs = self._cat_clf.predict_proba(X)[0]
            classes = list(self._cat_clf.classes_)
            order = np.argsort(cat_probs)[::-1]
            top_cat = str(classes[int(order[0])])
            top_cat_prob = float(cat_probs[int(order[0])])
            second_prob = float(cat_probs[int(order[1])]) if len(order) > 1 else 0.0
            cat_margin = top_cat_prob - second_prob

        if jb_boost:
            p_block = max(p_block, 0.93)
            top_cat = "jailbreak_injection"
            top_cat_prob = max(top_cat_prob, 0.90)
            cat_margin = max(cat_margin, 0.50)

        cpu_decision = "block" if p_block >= self._block_threshold else "allow"
        cpu_category = top_cat if top_cat in CATEGORIES else ("harmful_content" if cpu_decision == "block" else "normal")

        # Reconcile CPU decision & category
        if cpu_decision == "block" and cpu_category not in BLOCK_CATEGORIES:
            cpu_category = "harmful_content"
        elif cpu_decision == "allow" and cpu_category in BLOCK_CATEGORIES:
            cpu_category = "normal"

        cpu_conf = max(p_block, 1.0 - p_block)

        if self.pool.chat_client is not None:
            llm_res = await self._llm_classify(messages, similar_examples, query_text)
            if llm_res is not None:
                llm_dec, llm_cat, _ = llm_res
                if llm_dec == "block":
                    agreed = bool(jb_boost or cpu_decision == "block")
                    conf = 0.95 if agreed else 0.72
                else:
                    conf = 0.92 if cpu_decision == "allow" else 0.78
                return llm_dec, llm_cat, conf

        return cpu_decision, cpu_category, float(round(cpu_conf, 4))
