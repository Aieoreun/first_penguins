"""Offline, public-derived retrieval coverage; never a private input collector.

Measures raw retrieval before input/source gates. It does not reconstruct the
actual six-source pool because that pool also depends on the input LLM decision.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.kb import KBChunk, load_chunks
from core.llm import LLM
from core.search import Searcher, load_vectors, save_vectors
from pipelines.kimjaehak.dual_index import DenseView, candidate_pool, index_path, short_chunks, valid_vectors


async def measure(run_list, output):
    cache = ROOT / 'experiments/.cache'
    datasets = {}
    for value in json.loads(run_list.read_text()):
        run = (ROOT / value).resolve()
        if not run.is_relative_to(cache):
            raise ValueError('Expected local development run')
        manifest = json.loads((run / 'manifest.json').read_text())
        provenance = json.loads((run.parent / 'derivation.json').read_text())
        dataset = run.parent / 'derived_inputs.jsonl'
        digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
        if (provenance['scope'] not in {'public_paraphrase_full_diagnostic', 'provided_kb_fact_diagnostic'}
                or provenance['dataset_sha256'] != digest or manifest['dataset_sha256'] != digest):
            raise ValueError('Expected verified public-derived dataset')
        datasets[digest] = (str(dataset.relative_to(ROOT)), list(map(json.loads, dataset.read_text().splitlines())))
    llm = LLM()
    try:
        chunks = load_chunks(os.environ.get('KB_DIR', ROOT / 'kb'))
        long_vectors = load_vectors(os.environ.get('KB_INDEX', ROOT / 'index/kb_embeddings.json'), llm.embedding_model, chunks)
        long = DenseView(chunks, long_vectors)
        short = short_chunks(chunks)
        short = DenseView(short, load_vectors(index_path(), llm.embedding_model, short), long.matrix.shape[1])
        searcher = Searcher(chunks, long_vectors)
        queries = list(dict.fromkeys('\n'.join(m['content'] for m in row['messages'])[:6000]
                    for _, rows in datasets.values() for row in rows))
        query_chunks = [KBChunk('', '', '', '', q) for q in queries]
        path = cache / 'jh_r5_public_query_vectors.json'
        vectors = load_vectors(path, llm.embedding_model, query_chunks)
        if not valid_vectors(vectors, len(queries), long.matrix.shape[1]):
            vectors = []
            for start in range(0, len(queries), 32):
                batch = await llm.embed(queries[start:start+32], timeout=60)
                if not valid_vectors(batch, len(queries[start:start+32]), long.matrix.shape[1]):
                    raise ValueError('Invalid query vector batch')
                vectors.extend(batch)
            save_vectors(path, llm.embedding_model, query_chunks, vectors)
        by_query = dict(zip(queries, vectors))
        result = {'scope':'public_derived_raw_retrieval_before_gates', 'datasets':[]}
        for digest, (source, rows) in datasets.items():
            counts = dict(normal_with_evidence=0, baseline_hits=0, dual_hits=0, short_rescues=0)
            pool_sizes, pool_chars = [], []
            for row in rows:
                query = '\n'.join(m['content'] for m in row['messages'])
                vector = by_query[query[:6000]]
                bm25 = searcher.search(query, bm25_top=10, embedding_top=0)
                dense, fine = long.search(vector), short.search(vector)
                last = searcher.search(row['messages'][-1]['content'], bm25_top=1, embedding_top=0)
                baseline = candidate_pool(None, last, (bm25, dense))
                dual = candidate_pool(None, last, (bm25, dense, fine))
                pool_sizes.append(len(dual))
                pool_chars.append(sum(len(searcher.section_text(sid)) for sid in dual))
                gold = set(row['labels'].get('evidence', []))
                if row['labels']['category'] == 'normal' and gold:
                    before, after = bool(gold.intersection(baseline)), bool(gold.intersection(dual))
                    counts['normal_with_evidence'] += 1
                    counts['baseline_hits'] += before
                    counts['dual_hits'] += after
                    counts['short_rescues'] += after and not before
            result['datasets'].append({'source':source, 'dataset_sha256':digest, 'count':len(rows), **counts,
                'raw_dual_sections':{'mean':statistics.mean(pool_sizes),'median':statistics.median(pool_sizes),'max':max(pool_sizes)},
                'raw_dual_material_chars':{'mean':statistics.mean(pool_chars),'max':max(pool_chars)}})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        await llm._chat.close()
        if llm._embed is not None:
            await llm._embed.close()


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not all(p.resolve().is_relative_to(ROOT / 'experiments/.cache') for p in (args.runs, args.output)):
        parser.error('Use local development cache paths')
    asyncio.run(measure(args.runs, args.output))
