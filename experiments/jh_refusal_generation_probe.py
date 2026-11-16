"""Freeze own harmful-policy routing and vary only tutor refusal generation.

This is a full-tutor fixed-component diagnostic, not an end-to-end HTTP score.
Only completed ContextFast public/public-derived sources are accepted. Selection
uses source responses and current fixed-policy mechanics, never gold labels.
--resume scores a completed collection, reusing successful interrupted judgments;
completed results are never rejudged. --self-test needs no credentials or API.
"""
import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jh_tutor_generation_probe import evaluator_hashes, score_rows, source_data
from core.llm import LLM
from core.schema import ANSWER_MAX_CHARS, FALLBACK_ANSWER, Label, POLICY
from pipelines import load


BASE = 'jh_guard_fitted_contextfast'
VARIANTS = (BASE, 'jh_r3_safe_alternative')
SCOPE = 'frozen_own_harmful_generation_full_tutor_component_not_http'
GENERATION_TIMEOUT = 26
HARMFUL = {Label.HARMFUL_REQUEST, Label.HARMFUL_SPEECH}
NONHARMFUL_FIXED = frozenset(
    policy.fixed_answer.strip() for label, policy in POLICY.items()
    if label not in HARMFUL and policy.fixed_answer is not None
)


def predicted_harmful(response):
    """Recover the own policy subtype only where source mechanics identify it."""
    if not isinstance(response, dict) or response.get('citations') != []:
        return None
    answer = response.get('answer')
    if (not isinstance(answer, str) or not answer.strip()
            or len(answer) > ANSWER_MAX_CHARS or answer.strip() in NONHARMFUL_FIXED):
        return None
    return {'refuse': Label.HARMFUL_REQUEST,
            'redirect': Label.HARMFUL_SPEECH}.get(response.get('action'))


def check_source_manifest(manifest):
    if (manifest.get('variant') != BASE
            or manifest.get('effective_config', {}).get('contextual_refusal') is not True):
        raise ValueError('Source must be ContextFast with contextual_refusal=True')
    if manifest.get('scope') not in (None, 'public_paraphrase_full_diagnostic'):
        raise ValueError('Source must be bundled public or public-derived')
    if manifest.get('judge_enabled') is not True or manifest.get('category_filter'):
        raise ValueError('Source must be a fully judged, unfiltered tutor dataset')


def checked_source(source):
    source = source.resolve()
    if not source.is_relative_to(ROOT / 'experiments/.cache'):
        raise ValueError('Expected a local public diagnostic source')
    check_source_manifest(json.loads((source / 'manifest.json').read_text()))
    manifest, items, prior, result = source_data(source)
    if not items or result.get('judge_errors') != 0 or result.get('tutor_judge_skipped') is not False:
        raise ValueError('Source scoring is missing, skipped or failed; do not rejudge it')
    if not (source / 'judge.jsonl').is_file():
        raise ValueError('Source judge journal is missing')
    result_ids = [row['id'] for row in result['items']]
    if len(result_ids) != len(prior) or set(result_ids) != set(prior):
        raise ValueError('Source result is incomplete or duplicated')
    return manifest, items, prior, result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(source, manifest):
    result = 'derived_result.json' if manifest.get('scope') else 'result.json'
    files = [source / name for name in ('manifest.json', 'responses.jsonl', 'judge.jsonl', result)]
    if manifest.get('scope'):
        files += [source.parent / name for name in ('derived_inputs.jsonl', 'derivation.json')]
    else:
        files += [ROOT / 'public' / (endpoint + '.jsonl') for endpoint in manifest['endpoints']]
    return {str(path.relative_to(ROOT)): sha256(path) for path in files}


async def collect(items, prior, variant, workers, dest, pipeline=None):
    # No constructor/classification: own predictions are recovered from responses.
    if pipeline is None:
        cls = load(variant)
        pipeline = cls.__new__(cls)
        pipeline.llm = LLM()
    slots = asyncio.Semaphore(workers)
    completed = 0
    with (dest / 'responses.jsonl').open('x') as stream:
        async def one(item):
            nonlocal completed
            old = prior[item['id']]
            row = copy.deepcopy(old)
            label = None if old.get('error') else predicted_harmful(old.get('response'))
            row.update(generation_attempted=label is not None, generation_errors=[],
                       generation_label=label.value if label else None,
                       generation_seconds=0)
            if label is not None:
                async with slots:
                    # Only own conversation enters state/model; no ID or labels.
                    state = pipeline._new_state({'messages': copy.deepcopy(item['messages'])}, False)
                    state.label = label
                    start = time.monotonic()
                    try:
                        await asyncio.wait_for(pipeline._generate(state), GENERATION_TIMEOUT)
                    except TimeoutError:
                        state.answer, state.citations = FALLBACK_ANSWER, []
                        state.errors.append('refusal_probe:Deadline')
                    except Exception as exc:
                        state.answer, state.citations = FALLBACK_ANSWER, []
                        state.errors.append(f'refusal_probe:{type(exc).__name__}')
                    # Cancellation propagates; a partial collection cannot score.
                    row.update(response=state.to_tutor_response(),
                               generation_errors=list(state.errors),
                               generation_seconds=round(time.monotonic() - start, 3))
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            completed += 1
            if completed % 25 == 0 or completed == len(items):
                print(f'generated {completed}/{len(items)}', flush=True)
            return row
        return await asyncio.gather(*(one(item) for item in items))


def validate_collection(items, prior, rows):
    by_id = {row['id']: row for row in rows}
    if len(rows) != len(items) or len(by_id) != len(rows) or set(by_id) != {i['id'] for i in items}:
        raise ValueError('Incomplete or duplicate collection; generation is not resumed')
    for item in items:
        old, row = prior[item['id']], by_id[item['id']]
        label = None if old.get('error') else predicted_harmful(old.get('response'))
        if (row.get('generation_attempted') is not (label is not None)
                or row.get('generation_label') != (label.value if label else None)):
            raise ValueError('Own routing selection changed')
        if row.get('error') != old.get('error') or not isinstance(row.get('generation_errors'), list):
            raise ValueError('Error provenance changed')
        if label is None:
            if row.get('response') != old.get('response'):
                raise ValueError('A frozen response changed')
        else:
            response = row.get('response') or {}
            if (response.get('action') != old['response']['action']
                    or response.get('citations') != []
                    or not isinstance(response.get('answer'), str)
                    or not response['answer'].strip()
                    or len(response['answer']) > ANSWER_MAX_CHARS):
                raise ValueError('Generated response violates frozen routing or response shape')


def score(dest):
    dest = dest.resolve()
    if not dest.is_relative_to(ROOT / 'experiments/.cache'):
        raise ValueError('Expected a local component cache')
    with (dest / '.scoring.lock').open('a') as lock:
        # Release automatically on interruption; never run two scorers together.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _score(dest)


def _score(dest):
    if (dest / 'result.json').exists():
        raise ValueError('Completed scoring is never repeated, including judge-error results')
    meta = json.loads((dest / 'manifest.json').read_text())
    if (meta.get('scope') != SCOPE or meta.get('variant') not in VARIANTS
            or meta.get('model') != os.environ.get('OPENAI_MODEL')
            or meta.get('evaluator_hashes') != evaluator_hashes()):
        raise ValueError('Component scope/model/evaluator differs')
    source = ROOT / meta['source']
    manifest, items, prior, _ = checked_source(source)
    if source_hashes(source, manifest) != meta['source_hashes']:
        raise ValueError('Source data/manifest/result/journal changed')
    completed = json.loads((dest / 'collection.json').read_text())
    responses = dest / 'responses.jsonl'
    if completed['response_sha256'] != sha256(responses):
        raise ValueError('Collected responses changed')
    rows = [json.loads(line) for line in responses.read_text().splitlines()]
    validate_collection(items, prior, rows)
    if len(rows) != meta['count'] or completed['count'] != len(rows):
        raise ValueError('Component count changed')
    journal = dest / 'judge.jsonl'
    started = dest / 'scoring_started.json'
    if started.exists():
        if not journal.read_bytes().startswith((source / 'judge.jsonl').read_bytes()):
            raise ValueError('Original judge journal was altered during interrupted scoring')
    else:
        if sha256(journal) != meta['source_hashes'][str((source / 'judge.jsonl').relative_to(ROOT))]:
            raise ValueError('Judge journal changed before its first scoring attempt')
        with started.open('x') as stream:
            json.dump({'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}, stream)

    def stage(name, stage_rows):
        path = dest / (name + '_score.json')
        if path.exists():
            return json.loads(path.read_text())
        result = score_rows(items, stage_rows, meta['judge_workers'], journal=journal)
        # Preserve even judge-error results if the following stage is interrupted.
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
        return result

    baseline = stage('baseline', [prior[i['id']] for i in items])
    revised = stage('revised', rows)
    report = {**meta, 'baseline_tutor': baseline['tutor'], 'revised_tutor': revised['tutor'],
              'attempted': sum(row['generation_attempted'] for row in rows),
              'error_requests': sum(bool(row['generation_errors']) for row in rows),
              'baseline': baseline, 'revised': revised,
              'scoring_status': 'judge_errors_preserved' if baseline['judge_errors'] or revised['judge_errors'] else 'complete'}
    temporary = dest / 'result.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(dest / 'result.json')
    print(json.dumps({k: report[k] for k in ('variant', 'baseline_tutor', 'revised_tutor',
                                          'attempted', 'error_requests', 'scoring_status')}), flush=True)


def self_test():
    """Exercise provenance, gold isolation, cancellation and limits offline."""
    import tempfile
    import unittest
    from unittest.mock import AsyncMock, patch
    from pipelines.kimjaehak.calibrated import FIXED
    from pipelines.kimjaehak.guard_fitted import ContextFast
    from pipelines.kimjaehak.safe_alternative import Pipeline as SafeAlternative

    class Tests(unittest.IsolatedAsyncioTestCase):
        def test_policy_recovery(self):
            for label, policy in POLICY.items():
                if label not in HARMFUL and policy.fixed_answer:
                    self.assertIsNone(predicted_harmful({'answer': policy.fixed_answer,
                        'action': policy.action.value, 'citations': []}))
            for label in HARMFUL:
                self.assertEqual(predicted_harmful({'answer': FIXED[label],
                    'action': POLICY[label].action.value, 'citations': []}), label)
            self.assertIsNone(predicted_harmful({'answer': 'text', 'action': 'answer', 'citations': []}))
            self.assertIsNone(predicted_harmful({'answer': 'text', 'action': 'refuse', 'citations': ['PY-01-01']}))

        def test_source_restrictions(self):
            valid = {'variant': BASE, 'judge_enabled': True,
                     'effective_config': {'contextual_refusal': True}}
            check_source_manifest(valid)
            for change in ({'variant': 'other'}, {'effective_config': {'contextual_refusal': False}},
                           {'scope': 'provided_kb_fact_diagnostic'}, {'judge_enabled': False},
                           {'category_filter': ['harmful_content']}):
                with self.assertRaises(ValueError):
                    check_source_manifest({**valid, **change})

        async def test_generation_only_payload_and_frozen_responses(self):
            items = [{'id': 'OWN_ID_SENTINEL', 'endpoint': 'tutor',
                      'messages': [{'role': 'user', 'content': '가상 자료로 공부해요'}],
                      'labels': {'category': 'GOLD_SENTINEL', 'evidence': ['GOLD_SOURCE']}}]
            prior = {items[0]['id']: {'id': items[0]['id'], 'endpoint': 'tutor', 'seconds': 1,
                     'error': None, 'response': {'action': 'refuse', 'citations': [],
                     'answer': FIXED[Label.HARMFUL_REQUEST]}}}
            fixed = copy.deepcopy(items[0])
            fixed['id'] = 'FROZEN'
            items.append(fixed)
            prior['FROZEN'] = {**copy.deepcopy(prior[items[0]['id']]), 'id': 'FROZEN',
                              'response': {'action': 'redirect', 'citations': [],
                              'answer': POLICY[Label.OFF_TOPIC].fixed_answer}}
            for cls in (ContextFast, SafeAlternative):
                pipeline = cls.__new__(cls)
                pipeline.llm = type('FakeLLM', (), {})()
                pipeline.llm.chat = AsyncMock(return_value='가상 자료로 학습해요')
                with tempfile.TemporaryDirectory() as temp:
                    rows = await collect(items, prior, BASE, 16, Path(temp), pipeline)
                validate_collection(items, prior, rows)
                pipeline.llm.chat.assert_awaited_once()
                args, kwargs = pipeline.llm.chat.call_args
                for sentinel in ('OWN_ID_SENTINEL', 'GOLD_SENTINEL', 'GOLD_SOURCE'):
                    self.assertNotIn(sentinel, str(args))
                self.assertEqual(kwargs, {'effort': 'low', 'max_tokens': 1500, 'timeout': 8})
                self.assertEqual(rows[1]['response'], prior['FROZEN']['response'])
                self.assertEqual(rows[0]['generation_label'], Label.HARMFUL_REQUEST.value)

        async def test_cancellation_and_length(self):
            item = {'id': 'own', 'endpoint': 'tutor', 'messages': [{'role': 'user', 'content': '학습'}]}
            prior = {'own': {'id': 'own', 'endpoint': 'tutor', 'seconds': 1, 'error': None,
                            'response': {'action': 'refuse', 'citations': [],
                            'answer': FIXED[Label.HARMFUL_REQUEST]}}}
            pipeline = SafeAlternative.__new__(SafeAlternative)
            pipeline.llm = type('FakeLLM', (), {})()
            pipeline.llm.chat = AsyncMock(side_effect=asyncio.CancelledError())
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(asyncio.CancelledError):
                    await collect([item], prior, VARIANTS[1], 16, Path(temp), pipeline)
                self.assertEqual((Path(temp) / 'responses.jsonl').read_text(), '')
            pipeline.llm.chat = AsyncMock(return_value='가' * ANSWER_MAX_CHARS)
            with tempfile.TemporaryDirectory() as temp:
                rows = await collect([item], prior, VARIANTS[1], 16, Path(temp), pipeline)
            self.assertEqual(rows[0]['response']['answer'], FIXED[Label.HARMFUL_REQUEST])
            self.assertEqual(rows[0]['generation_errors'], ['safe_alternative:ValueError'])
            validate_collection([item], prior, rows)

        def test_interrupted_scoring_resumes_but_completed_error_results_do_not(self):
            module = sys.modules[__name__]
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                source = root / 'experiments/.cache/source'
                dest = root / 'experiments/.cache/component'
                source.mkdir(parents=True)
                dest.mkdir()
                original = '{"key":"original","result":{}}\n'
                (source / 'judge.jsonl').write_text(original)
                (dest / 'judge.jsonl').write_text(original)
                item = {'id': 'own', 'endpoint': 'tutor', 'messages': []}
                old = {'id': 'own', 'endpoint': 'tutor', 'error': None,
                       'response': {'action': 'refuse', 'citations': [],
                                    'answer': FIXED[Label.HARMFUL_REQUEST]}}
                row = {**old, 'generation_attempted': True, 'generation_errors': [],
                       'generation_label': Label.HARMFUL_REQUEST.value}
                (dest / 'responses.jsonl').write_text(json.dumps(row) + '\n')
                (dest / 'collection.json').write_text(json.dumps({
                    'count': 1, 'response_sha256': sha256(dest / 'responses.jsonl')}))
                hashes = {'experiments/.cache/source/judge.jsonl': sha256(source / 'judge.jsonl')}
                meta = {'scope': SCOPE, 'variant': BASE, 'model': 'mock-model',
                        'evaluator_hashes': {}, 'source': 'experiments/.cache/source',
                        'source_hashes': hashes, 'count': 1, 'judge_workers': 1}
                (dest / 'manifest.json').write_text(json.dumps(meta))
                baseline = {'tutor': 90, 'judge_errors': 1}
                revised = {'tutor': 91, 'judge_errors': 0}
                calls = []

                def scoring(*args, **kwargs):
                    calls.append(1)
                    if len(calls) == 1:
                        return baseline
                    if len(calls) == 2:
                        with kwargs['journal'].open('a') as stream:
                            stream.write('{"key":"completed-before-interruption","result":{}}\n')
                        raise RuntimeError('interrupted')
                    self.assertIn('completed-before-interruption', kwargs['journal'].read_text())
                    return revised

                with (patch.object(module, 'ROOT', root),
                      patch.object(module, 'checked_source', return_value=({}, [item], {'own': old}, {})),
                      patch.object(module, 'source_hashes', return_value=hashes),
                      patch.object(module, 'evaluator_hashes', return_value={}),
                      patch.object(module, 'score_rows', side_effect=scoring),
                      patch.dict(os.environ, {'OPENAI_MODEL': 'mock-model'})):
                    with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                        score(dest)
                    score(dest)
                    self.assertEqual(len(calls), 3)  # Baseline error result was not rerun.
                    report = json.loads((dest / 'result.json').read_text())
                    self.assertEqual(report['scoring_status'], 'judge_errors_preserved')
                    with self.assertRaises(ValueError):
                        score(dest)
                    self.assertEqual(len(calls), 3)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--name')
    ap.add_argument('--variant', choices=VARIANTS)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--judge-workers', type=int, default=32)
    ap.add_argument('--collect-only', action='store_true')
    ap.add_argument('--resume', type=Path, help='Continue scoring with the existing journal; never regenerate/rejudge completed results')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=True)
    if args.resume:
        if args.source or args.variant or args.name or args.collect_only:
            ap.error('--resume only scores an existing collection')
        score(args.resume)
        return
    if (not args.source or not args.variant or not args.name
            or not re.fullmatch(r'[A-Za-z0-9_-]+', args.name)
            or not 1 <= args.workers <= 64 or not 1 <= args.judge_workers <= 64):
        ap.error('Invalid arguments')
    source = args.source.resolve()
    manifest, items, prior, old_result = checked_source(source)
    selected = sum(not prior[item['id']].get('error')
                   and predicted_harmful(prior[item['id']].get('response')) is not None for item in items)
    if not selected:
        ap.error('No eligible own harmful-policy responses; no API calls made')
    cls = load(args.variant)
    if getattr(cls, 'contextual_refusal', None) is not True:
        ap.error('Generation variant must retain contextual refusal mechanics')
    dest = ROOT / 'experiments/.cache' / ('jh_refusal_generation_' + args.name)
    hashes = source_hashes(source, manifest)
    metadata = {'scope': SCOPE, 'score_scope': 'full_tutor_fixed_component_not_http',
                'variant': args.variant, 'source': str(source.relative_to(ROOT)),
                'source_manifest': manifest, 'source_hashes': hashes,
                'model': manifest['model'], 'evaluator_hashes': evaluator_hashes(),
                'source_tutor_score': old_result['tutor'], 'count': len(items),
                'selected_count': selected, 'workers': args.workers, 'judge_workers': args.judge_workers,
                'runtime_limit': 'generation_only_26s; policy_call_low_1500_tokens_8s',
                'runtime_labels': 'own_action_empty_citations_excluding_fixed_nonharmful_never_gold',
                'code_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in (
                    Path(__file__).resolve(), ROOT / 'pipelines/kimjaehak/context_policy.py',
                    ROOT / 'pipelines/kimjaehak/safe_alternative.py', ROOT / 'pipelines/kimjaehak/calibrated.py')},
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())}
    dest.mkdir(exist_ok=False)
    shutil.copyfile(source / 'judge.jsonl', dest / 'judge.jsonl')
    (dest / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    rows = asyncio.run(collect(items, prior, args.variant, args.workers, dest))
    validate_collection(items, prior, rows)
    (dest / 'collection.json').write_text(json.dumps({
        'count': len(rows), 'response_sha256': sha256(dest / 'responses.jsonl')}) + '\n')
    if args.collect_only:
        print('SCORING_PENDING', dest, flush=True)
    else:
        score(dest)


if __name__ == '__main__':
    main()
