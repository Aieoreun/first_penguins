"""No-API equivalence screen using public-fitted models and synthetic vectors.

This tests the trusted tutor route, not embedding quality, HTTP latency or score.
Input/source models are fitted once on the existing bundled public data only.
"""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = {
    'public_tutor': 'public/tutor.jsonl',
    'primary': 'experiments/.cache/jh_transfer_r6_17_primary_base/derived_inputs.jsonl',
    'alternative': 'experiments/.cache/jh_transfer_r6_25_alternative_base/derived_inputs.jsonl',
}
PUBLIC_HASHES = {
    'public/tutor.jsonl': '8f8a74a7747281d89b27cb972169f95401b1082ebfa99dce72af55e0f6dc31e0',
    'public/guard.jsonl': '4a585c174f15dec205a286ec45e8c14ff28b01bbcedde3097072ca52e3b0c426',
}
EXPECTED_COUNTS = {'public_tutor': 200, 'primary': 192, 'alternative': 181}
THRESHOLDS = {
    'learned_gate_c': 1000, 'learned_gate_threshold': .95,
    'learned_section_c': 1000, 'learned_section_threshold': .98,
    'tutor_section_threshold': .90, 'learned_section_require_retrieved': False,
    'learned_section_use_for_output': False,
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload_only(row):
    return {'spec_version': '1.0', 'session_id': 'synthetic-equivalence-session',
            'messages': deepcopy(row['messages'])}


class FakeLLM:
    can_embed = True
    embedding_model = 'synthetic-two-dimensional-not-organizer'

    def __init__(self):
        self.embedding_calls = 0
        self.chat_calls = 0

    async def embed(self, texts, timeout):
        self.embedding_calls += 1
        return [[1.0, 0.0] for _ in texts]

    async def chat(self, *args, **kwargs):
        self.chat_calls += 1
        raise AssertionError('A trusted route unexpectedly attempted chat')


def verified_sources():
    for path, expected in PUBLIC_HASHES.items():
        if digest((ROOT/path).read_bytes()) != expected:
            raise ValueError('Bundled public training data changed: '+path)
    datasets, provenance = {}, {}
    for name, relative in SOURCES.items():
        path = ROOT/relative
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        if (len(rows) != EXPECTED_COUNTS[name] or len({r['id'] for r in rows}) != len(rows)
                or any(r['endpoint'] != 'tutor' for r in rows)):
            raise ValueError('Expected complete tutor dataset: '+name)
        entry = {'path': relative, 'count': len(rows), 'dataset_sha256': digest(raw)}
        if name != 'public_tutor':
            metadata_path = path.parent/'derivation.json'
            metadata = json.loads(metadata_path.read_text())
            source = (ROOT/metadata['source']).resolve()
            if (metadata.get('scope') != 'public_paraphrase_full_diagnostic'
                    or metadata.get('score_scope') != 'tutor'
                    or metadata.get('count') != len(rows)
                    or metadata.get('paraphrase_partition') != name
                    or metadata.get('dataset_sha256') != digest(raw)
                    or not source.is_relative_to(ROOT/'experiments/.cache')
                    or digest(source.read_bytes()) != metadata.get('source_sha256')):
                raise ValueError('Derived provenance mismatch: '+name)
            entry.update(derivation_sha256=digest(metadata_path.read_bytes()),
                         generation_source_sha256=metadata['source_sha256'])
        datasets[name], provenance[name] = rows, entry
    return datasets, provenance


async def screen(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT/'experiments/.cache') or output.exists():
        raise ValueError('Use a new output file in experiments/.cache')
    datasets, sources = verified_sources()
    protected = subprocess.run(
        ['git', 'diff', '--exit-code', '6ec111d', '--', 'kb', 'public', 'core'],
        cwd=ROOT, capture_output=True, check=False)
    if protected.returncode:
        raise ValueError('Protected KB/public/core differ from reference 6ec111d')
    kb_hashes = {str(p.relative_to(ROOT)): digest(p.read_bytes())
                 for p in sorted((ROOT/'kb').rglob('*')) if p.is_file()}
    from pipelines import load
    base_class = load('jh_r4_material_budget')
    fast_class = load('jh_r6_trusted_sparse')
    from core.kb import load_chunks
    from core.search import Searcher
    for cls in (base_class, fast_class):
        for key, expected in THRESHOLDS.items():
            if getattr(cls, key) != expected:
                raise ValueError('Unexpected pinned threshold: '+key)
    chunks = load_chunks(ROOT/'kb')
    # A deliberate dense distractor: the first ten chunks outrank all others.
    vectors = [[1.0, 0.0] if i < 10 else [0.0, 1.0] for i in range(len(chunks))]
    base_llm, fast_llm = FakeLLM(), FakeLLM()
    base = base_class(Searcher(chunks, vectors), base_llm)
    fast = fast_class.__new__(fast_class)
    fast.__dict__ = base.__dict__.copy()
    fast.llm = fast_llm
    identities = {name: getattr(fast_class, name) is getattr(base_class, name)
                  for name in ('guard', '_check_output', '_classify', '_generate', '_new_state')}
    if not all(identities.values()):
        raise ValueError('Unexpected changes outside the search override')
    report = {'scope': 'public_fitted_synthetic_embedding_equivalence_not_http_or_score',
              'api_calls': 0, 'thresholds': THRESHOLDS, 'sources': sources,
              'training_public_sha256': PUBLIC_HASHES, 'protected_reference': '6ec111d',
              'kb_file_sha256': kb_hashes, 'kb_chunk_count': len(chunks),
              'synthetic_vectors': '2D: first ten chunks [1,0], all others [0,1]; query [1,0]',
              'guard_and_routing_method_identity': identities, 'datasets': {},
              'limitations': ['Only independently eligible trusted tutor routes are executed.',
                              'No HTTP, organizer API, embedding quality or score measurement.',
                              'Existing public-fitted models; derived datasets are development data.',
                              'Healthy mock calls do not establish equivalence under real deadline failures.']}
    failures = []
    for name, rows in datasets.items():
        records = []
        for row in rows:
            body = payload_only(row)
            state_fast = fast._new_state(body, False)
            if not fast._skip_query_embedding(state_fast):
                continue
            guard_state = fast._new_state(body, True)
            if fast._skip_query_embedding(guard_state):
                raise AssertionError('Guard incorrectly eligible for skipping')
            state_base = base._new_state(body, False)
            before = (base_llm.embedding_calls, fast_llm.embedding_calls,
                      base_llm.chat_calls, fast_llm.chat_calls)
            for pipeline, state in ((base, state_base), (fast, state_fast)):
                await pipeline._search(state)
                await pipeline._classify(state)
                await pipeline._generate(state)
            calls = tuple(a-b for a, b in zip(
                (base_llm.embedding_calls, fast_llm.embedding_calls,
                 base_llm.chat_calls, fast_llm.chat_calls), before))
            response_base, response_fast = state_base.to_tutor_response(), state_fast.to_tutor_response()
            equal = response_base == response_fast
            valid = equal and calls == (1, 0, 0, 0) and not state_base.errors and not state_fast.errors
            if not valid:
                failures.append({'dataset': name, 'id': row['id']})
            records.append({'id': row['id'], 'response_equal': equal,
                'base_embedding_calls': calls[0], 'fast_embedding_calls': calls[1],
                'base_chat_calls': calls[2], 'fast_chat_calls': calls[3],
                'retrieval_changed': [(c.section_id, c.text) for c in state_base.chunks]
                    != [(c.section_id, c.text) for c in state_fast.chunks],
                'label_equal': state_base.label == state_fast.label,
                'section_equal': state_base.section == state_fast.section,
                'base_errors': state_base.errors, 'fast_errors': state_fast.errors,
                'response_sha256': digest(json.dumps(response_base, ensure_ascii=False,
                                                    sort_keys=True).encode())})
        report['datasets'][name] = {'total': len(rows), 'eligible': len(records),
            'identical_responses': sum(r['response_equal'] for r in records),
            'retrieval_changed': sum(r['retrieval_changed'] for r in records), 'records': records}
        print(name, {k:v for k,v in report['datasets'][name].items() if k != 'records'}, flush=True)
    # Verify files remained unchanged during model fitting and execution.
    _, after_sources = verified_sources()
    if after_sources != sources or any(digest((ROOT/p).read_bytes()) != h for p,h in kb_hashes.items()):
        raise ValueError('Input files changed during screen')
    report.update(mismatches=failures, passed=not failures,
                  base_embedding_calls=base_llm.embedding_calls,
                  fast_embedding_calls=fast_llm.embedding_calls,
                  chat_calls=base_llm.chat_calls+fast_llm.chat_calls,
                  script_sha256=digest(Path(__file__).read_bytes()))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if failures:
        raise AssertionError('Equivalence failed; inspect saved mismatches')
    print('PASS', sum(d['eligible'] for d in report['datasets'].values()), 'eligible pairs; API=0', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'experiments/.cache/jh_r6_trusted_sparse_screen.json')
    args = parser.parse_args()
    asyncio.run(screen(args.output))
