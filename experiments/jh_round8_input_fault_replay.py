"""Capture public input checks, then replay selective failures without API calls.

Input component only: no output checks, HTTP score, or real timeout latency.
Captured checks and prepared retrieval context stay in ignored development cache.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
from copy import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.jh_environment import load_environment
from experiments.jh_guard_fusion_datasets import load_dataset
from experiments.jh_classifier_matrix import metrics
from experiments.jh_compare import evaluator_hashes
from core.kb import load_chunks
from core.search import Searcher
from core.state import Chunk
from core.schema import Decision, Label, parse_label
from pipelines import load
load('jh_r7_binary_guard')
from pipelines.kimjaehak.round8_failure_fallback import FailureInput, FailureBlock
from pipelines.kimjaehak.round7_binary_model import BLOCKED


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def call_key(name, system, user, effort):
    return digest([name, system, user, effort])


async def capture(base, items):
    sem = asyncio.Semaphore(64)

    async def one(item):
        async with sem:
            p = copy(base)
            state = p._new_state(item, True)
            await p._search(state, extra=state.candidate_answer or '')
            state.section = state.chunks[0].section_id if state.chunks else None
            prepared = {'chunks': [asdict(c) for c in state.chunks],
                        'request': state.request, 'query': state.query,
                        'errors': list(state.errors), 'section': state.section}
            calls = []
            original = p._ask

            async def ask(s, name, system, user, effort, timeout):
                record = {'name': name, 'key': call_key(name, system, user, effort)}
                calls.append(record)
                record['answer'] = await original(s, name, system, user, effort, timeout)
                return record['answer']

            p._ask = ask
            await p._classify(state)
            return {'id': item['id'], 'prepared': prepared, 'calls': calls,
                    'category': state.category.value, 'errors': state.errors}

    return await asyncio.gather(*(one(item) for item in items))


def partial_block_fallback(p, state, threshold):
    """Exploratory policy; never overwrite a valid BLOCK ballot or final BLOCK."""
    if (state.decision is Decision.BLOCK or state.debug.get('valid_block_votes') != 0
            or not any(e.startswith('block:') for e in state.errors)):
        return False
    probabilities = p.matrix_model.probabilities([state.last_user])[0]
    indexes = [i for i, c in enumerate(p.matrix_model.classes) if c in BLOCKED]
    if sum(probabilities[i] for i in indexes) < threshold:
        return False
    category = str(p.matrix_model.classes[max(indexes, key=lambda i: probabilities[i])])
    label = Label.HARMFUL_REQUEST if category == 'harmful_content' else parse_label(category)
    if label is None:
        return False
    state.label = label
    return True


async def replay(base, item, recorded, failed, policy):
    p = copy(base)
    if policy in ('complete', 'block_runtime'):
        cls = FailureInput if policy == 'complete' else FailureBlock
        p = cls.__new__(cls)
        p.__dict__.update(base.__dict__)
    state = p._new_state(item, True)
    for key, value in recorded['prepared'].items():
        setattr(state, key, [Chunk(**c) for c in value] if key == 'chunks'
                else list(value) if key == 'errors' else value)
    calls = defaultdict(list)
    used = Counter()
    for call in recorded['calls']:
        calls[call['key']].append(call['answer'])

    async def ask(s, name, system, user, effort, timeout):
        key = call_key(name, system, user, effort)
        index = used[key]
        used[key] += 1
        if name in failed:
            # Failed early ballots can trigger an extra tie-break call. It is
            # part of the same injected stage outage, not a missing GPT answer.
            if not calls[key]:
                raise ValueError('Injected call payload was never captured')
            s.errors.append(name + ':InjectedTimeout')
            return {}
        if index >= len(calls[key]):
            raise ValueError('Replay call does not match captured public context')
        answer = calls[key][index]
        if not answer:
            s.errors.append(name + ':CapturedFailure')
        return answer

    p._ask = ask
    await p._classify(state)
    changed = bool(state.debug.get('r8_failure_input') or state.debug.get('r8_failure_block'))
    if policy.startswith('partial'):
        changed = partial_block_fallback(p, state, float(policy.split(':')[1]))
    failed_keys = {c['key'] for c in recorded['calls'] if c['name'] in failed}
    if any(used[k] != len(v) for k, v in calls.items() if k not in failed_keys):
        raise ValueError('Replay skipped captured input checks')
    return state.category.value, changed


async def run(args, dest, items, provenance):
    from core.llm import LLM
    llm = LLM()
    try:
        base = load('jh_r7_binary_guard')(Searcher(load_chunks(ROOT / 'kb')), llm)
        captured_path = dest / 'captured.json'
        if args.capture:
            captured = await capture(base, items)
            captured_path.write_text(json.dumps(captured, ensure_ascii=False, indent=2) + '\n')
            (dest / 'manifest.json').write_text(json.dumps({
                'dataset': provenance, 'input_sha256': digest(items),
                'captured_sha256': digest(captured), 'evaluator_hashes': evaluator_hashes(),
                'scope': 'public_input_component_capture_not_full_guard_not_http',
                'retrieval': 'BM25-only component capture; separate from live env embedding tests',
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }, indent=2) + '\n')
        captured = json.loads(captured_path.read_text())
        manifest = json.loads((dest / 'manifest.json').read_text())
        assert manifest['input_sha256'] == digest(items)
        assert manifest['captured_sha256'] == digest(captured)
        assert [r['id'] for r in captured] == [r['id'] for r in items]
        gold = [r['labels']['category'] for r in items]
        reports = []
        for failure in ('none', 'block', 'other', 'material', 'all'):
            failed = {'block', 'other', 'material'} if failure == 'all' else {failure}
            baseline = None
            reference_partial = None
            for policy in ('baseline', 'complete', 'partial:0.5', 'partial:0.7', 'partial:0.9', 'block_runtime'):
                outputs = [await replay(base, i, c, failed, policy) for i, c in zip(items, captured)]
                pred = [x[0] for x in outputs]
                if policy == 'baseline':
                    baseline = pred
                    if failure == 'none':
                        assert pred == [r['category'] for r in captured]
                if policy == 'partial:0.7':
                    reference_partial = outputs
                if policy == 'block_runtime':
                    assert outputs == reference_partial
                report = {'failure': failure, 'policy': policy, 'metrics': metrics(gold, pred),
                          'fallback_count': sum(x[1] for x in outputs),
                          'corrected': sum(a == g and b != g for a,b,g in zip(pred,baseline,gold)),
                          'regressed': sum(a != g and b == g for a,b,g in zip(pred,baseline,gold))}
                reports.append(report)
                print(json.dumps(report), flush=True)
        (dest / 'results.json').write_text(json.dumps({'scope': 'input_only_selective_failure_replay_not_live_score',
            'replay_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'manifest': manifest, 'results': reports}, indent=2) + '\n')
        (dest / 'COMPLETE').write_text('Capture and deterministic replay complete.\n')
    finally:
        await llm._chat.close()
        if llm._embed is not None:
            await llm._embed.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--name', required=True)
    ap.add_argument('--dataset', choices=['paraphrase', 'novel', 'alternative'], required=True)
    ap.add_argument('--capture', action='store_true')
    args = ap.parse_args()
    if not args.name.replace('_', '').isalnum():
        ap.error('Simple name required')
    dest = ROOT / 'experiments/.cache' / ('jh_r8_input_fault_' + args.name)
    if args.capture:
        dest.mkdir()
    items, provenance = load_dataset(args.dataset, ROOT)
    load_environment()
    asyncio.run(run(args, dest, items, provenance))


if __name__ == '__main__':
    main()
