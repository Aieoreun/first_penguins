"""KB-only short view alongside the unchanged heading-level embedding index."""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from core.search import load_vectors, normalize, save_vectors

ROOT = Path(__file__).resolve().parents[2]
SHORT_BODY_TARGET = 220  # soft character target; never split a paragraph/Q&A/code/table


def short_chunks(chunks, target=SHORT_BODY_TARGET):
    """Keep parent IDs/headings; Q&A pairs and fenced blocks remain atomic."""
    result = []
    for chunk in chunks:
        if chunk.heading == '표':
            result.append(chunk)
            continue
        lines = chunk.text.splitlines()
        header = [lines.pop(0)]
        if lines and lines[0].startswith('## '):
            header.append(lines.pop(0))
        body = '\n'.join(lines).strip()
        # A Q&A may have blank lines/code inside; preserve it up to the next Q.
        parts = re.split(r'(?m)(?=^### Q\.)', body)
        units = []
        for part in parts:
            if part.startswith('### Q.'):
                units.append(part.strip())
                continue
            block, fence = [], None
            for line in part.splitlines():
                marker = re.match(r'^\s*(`{3,}|~{3,})', line)
                if marker:
                    token = marker.group(1)
                    if fence is None:
                        fence = token
                    elif token[0] == fence[0] and len(token) >= len(fence):
                        fence = None
                if not line.strip() and fence is None:
                    if block:
                        units.append('\n'.join(block).strip())
                        block = []
                else:
                    block.append(line)
            if block:
                units.append('\n'.join(block).strip())
        packed, current = [], ''
        for unit in units:
            if current and (len(current) + 2 + len(unit) > target or unit.startswith('### Q.')
                            or current.startswith('### Q.')):
                packed.append(current)
                current = ''
            current = current + '\n\n' + unit if current else unit
        if current:
            packed.append(current)
        for part in packed or ['']:
            result.append(replace(chunk, text='\n'.join(header) + '\n' + part))
    return result


def index_path():
    return Path(os.environ.get('JH_SHORT_INDEX', ROOT / 'index/jh_short_embeddings.json'))


def valid_vectors(vectors, count, dimension=None):
    if not vectors or len(vectors) != count or not vectors[0]:
        return False
    dimension = dimension or len(vectors[0])
    return all(len(v) == dimension and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
               and any(x != 0 for x in v) for v in vectors)


async def prepare_short_index():
    """Run before uvicorn; no request-time builds and no changes to server.py."""
    from core.kb import load_chunks
    from core.llm import LLM
    llm = LLM()
    try:
        if not llm.can_embed:
            raise ValueError('Dual retrieval requires embedding configuration')
        chunks = short_chunks(load_chunks(os.environ.get('KB_DIR', ROOT / 'kb')))
        path = index_path()
        vectors = load_vectors(path, llm.embedding_model, chunks)
        if not valid_vectors(vectors, len(chunks)):
            vectors = []
            for start in range(0, len(chunks), 32):
                batch = chunks[start:start + 32]
                values = await llm.embed([c.text for c in batch], timeout=60)
                if not valid_vectors(values, len(batch), len(vectors[0]) if vectors else None):
                    raise ValueError('Invalid short embedding batch')
                vectors.extend(values)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
            try:
                save_vectors(temporary, llm.embedding_model, chunks, vectors)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return {'short_chunks':len(chunks), 'short_dimension':len(vectors[0]),
                'short_body_target':SHORT_BODY_TARGET,
                'short_text_sha256':hashlib.sha256(json.dumps(
                    [c.text for c in chunks], ensure_ascii=False).encode()).hexdigest(),
                'short_index_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    finally:
        await llm._chat.close()
        if llm._embed is not None:
            await llm._embed.close()


class DenseView:
    def __init__(self, chunks, vectors, dimension=None):
        if not valid_vectors(vectors, len(chunks), dimension):
            raise ValueError('Missing or incompatible embedding index; use experiments/jh_server.py')
        self.chunks = chunks
        self.matrix = np.asarray([normalize(v) for v in vectors], dtype=np.float64)

    def search(self, vector, top=10):
        if not valid_vectors([vector], 1, self.matrix.shape[1]):
            raise ValueError('Query embedding dimension mismatch')
        # einsum avoids thread-pool startup for these small matrices.
        scores = np.einsum('ij,j->i', self.matrix, normalize(vector), optimize=False)
        order = np.argsort(-scores, kind='stable')[:top]
        return [self.chunks[i] for i in order]


def candidate_pool(section, last, routes, fallback=()):
    """Round-robin unique sections, not child chunks; no route gets buried."""
    routes = [list(dict.fromkeys(c.section_id for c in route)) for route in routes]
    pool = [section] + [c.section_id for c in last]
    for rank in range(max(map(len, routes), default=0)):
        pool.extend(route[rank] for route in routes if rank < len(route))
    pool.extend(c.section_id for c in fallback)
    return list(dict.fromkeys(sid for sid in pool if sid))
