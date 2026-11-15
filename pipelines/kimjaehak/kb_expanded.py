"""Supplement lexical retrieval with questions generated from the KB alone.

Generated strings only affect index scores. The answer context always contains
the original KB text, never a generated question or a stored reference answer.
"""
import hashlib
import json
from pathlib import Path

from core.search import BM25
from core.state import Chunk
from .adaptive import Pipeline as Adaptive
from .learned_sections import TutorOnly


DATA = Path(__file__).resolve().parent / "data"


class ExpandedIndex:
    def __init__(self, searcher, records):
        self.searcher = searcher
        self.order = searcher.section_ids()
        questions = [(row["section"],question) for row in records if row["error"] is None for question in row["questions"]]
        if any(sid not in self.order for sid,_ in questions):
            raise ValueError("Unknown generated-question section")
        self.ids = [sid for sid,_ in questions]
        self.index = BM25([question for _,question in questions])

    def best_chunk(self, query, weight):
        raw = self.searcher.bm25.scores(query)
        ranking = sorted(range(len(raw)),key=lambda i:-raw[i])
        representatives = {}
        for i in ranking:
            representatives.setdefault(self.searcher.chunks[i].section_id,i)
        base = {sid:raw[i] for sid,i in representatives.items()}
        generated = {sid:0. for sid in self.order}
        for sid,score in zip(self.ids,self.index.scores(query)):
            generated[sid] = max(generated[sid],score)
        base_scale = max(base.values(),default=0) or 1
        generated_scale = max(generated.values(),default=0) or 1
        scores = {sid:value/base_scale+weight*generated[sid]/generated_scale for sid,value in base.items()}
        if not scores or not max(scores.values()):
            return None
        section = max(representatives,key=lambda sid:scores[sid])
        return self.searcher.chunks[representatives[section]]


class Pipeline(Adaptive):
    query_expansion_weight = .1

    def __init__(self, searcher, llm):
        super().__init__(searcher,llm)
        raw = (DATA / "kb_questions.jsonl").read_bytes()
        metadata = json.loads((DATA / "kb_questions_manifest.json").read_text())
        sections = {sid:searcher.section_text(sid) for sid in searcher.section_ids()}
        digest = hashlib.sha256(json.dumps(sections,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if metadata["source_sha256"] != digest or metadata["questions_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("KB question index provenance mismatch")
        self.expanded_index = ExpandedIndex(searcher,list(map(json.loads,raw.decode().splitlines())))

    async def _search(self, state, extra=""):
        await super()._search(state,extra)
        query = "\n".join(m["content"] for m in state.messages)
        extra = (state.query + "\n" + extra).strip()
        if extra:
            query += "\n" + extra
        best = self.expanded_index.best_chunk(query,self.query_expansion_weight)
        if best is None or (state.chunks and state.chunks[0].section_id == best.section_id):
            return
        count = len(state.chunks) or 1
        rest = [c for c in state.chunks if (c.section_id,c.text) != (best.section_id,best.text)]
        state.chunks = ([Chunk(best.section_id,best.text)] + rest)[:count]


class Fitted(Pipeline,TutorOnly):
    """Keep the fitted public classifier while expanding its search fallback."""
