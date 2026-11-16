"""Public-only review comparison and concurrent load probe; never used by /guard.

Run with the project's .env already exported. No labels/IDs enter the pipeline.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values) * p) - 1)] if values else 0.0


def fingerprint():
    paths = list(ROOT.glob('*.py')) + list((ROOT / 'kb').rglob('*'))
    paths += [ROOT / 'work/kb_index.json', ROOT / 'public/guard.jsonl']
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths) if p.is_file()
    }


def score_run(public, records):
    from harness.score_local import score_local, as_scoring_input
    import scoring

    # workers=1 preserves the public order. This callback uses saved responses;
    # the official scorer makes zero extra API calls.
    responses = iter(record['response'] for record in records)
    official = score_local(public, lambda _endpoint, _payload: next(responses),
                           judge=None, kb={}, workers=1)
    details = [scoring.score_guard(*as_scoring_input(p), record['response'])
               for p, record in zip(public, records)]
    outputs = [r for r in details if r['has_output']]
    categories = [r for r in details if r['category_scored']]
    times = [r['seconds'] for r in records]
    mean = lambda values: round(100 * sum(values) / len(values), 2) if values else None
    return {
        'public_items': len(public),
        'guard_score': official['guard'],
        'block_f1_percent': round(100 * scoring.block_f1(details), 2),
        'non_normal_category_accuracy_percent': mean([r['category_correct'] for r in categories]),
        'output_items': len(outputs),
        'grounded_accuracy_percent': mean([r['output_correct'] if r['valid'] else 0 for r in outputs]),
        'unsupported_f1_percent': mean([r['span_f1'] if r['valid'] else 0 for r in outputs]),
        'safe_accuracy_percent': mean([r['safe_correct'] if r['valid'] else 0 for r in outputs]),
        'false_blocks_valid_responses': sum(r['false_block'] for r in details),
        'missed_blocks_valid_responses': sum(r['missed_block'] for r in details),
        'invalid_responses': official['invalid'],
        'request_errors': dict(Counter(r['error'] for r in records if r['error'])),
        'latency_p95_seconds': round(percentile(times, .95), 3),
        'latency_max_seconds': round(max(times, default=0), 3),
    }


def compare_runs(public, before, after):
    fields = ('decision', 'category', 'grounded', 'safe')
    result = {field: {'wrong_to_right': 0, 'right_to_wrong': 0} for field in fields}
    for item, old, new in zip(public, before, after):
        for field in fields:
            if field in ('grounded', 'safe') and 'candidate_answer' not in item:
                continue
            part = 'input' if field in ('decision', 'category') else 'output'
            previous = (old['response'] or {}).get(part, {}).get(field)
            current = (new['response'] or {}).get(part, {}).get(field)
            gold = item['labels'][field]
            if previous != gold and current == gold:
                result[field]['wrong_to_right'] += 1
            elif previous == gold and current != gold:
                result[field]['right_to_wrong'] += 1
    return result


async def run_batch(supervisor, public, workers):
    semaphore = asyncio.Semaphore(workers)

    async def one(item):
        async with semaphore:
            start = time.monotonic()
            response, error = None, None
            try:
                # The public ID and labels are deliberately not passed here.
                response = await asyncio.wait_for(
                    supervisor.evaluate(item['messages'], item.get('candidate_answer')),
                    timeout=29.5,
                )
            except Exception as exc:
                # Do not dump exception bodies: SDK failures may contain secrets.
                error = type(exc).__name__
            return {'id': item['id'], 'response': response, 'error': error,
                    'seconds': round(time.monotonic() - start, 4)}

    return await asyncio.gather(*(one(item) for item in public))


async def evaluate(args):
    from openai import AsyncOpenAI
    from guard_supervisor import GuardSupervisor
    from kb_search import KnowledgeSearch

    # Fixed source: never accept an arbitrary request-log dataset.
    public = [json.loads(line) for line in (ROOT / 'public/guard.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    if args.limit:
        public = public[:args.limit]
    if not public or any(item.get('endpoint') != 'guard' for item in public):
        raise ValueError('public/guard.jsonl must contain public guard items')
    before_hashes = fingerprint()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    destination = ROOT / 'work/evaluations' / f'reviewer-{stamp}'
    destination.mkdir(parents=True)
    source = destination / 'source'
    source.mkdir()
    for path in ROOT.glob('*.py'):
        shutil.copy2(path, source / path.name)
    manifest = {'source_hashes': before_hashes, 'chat_model': os.environ['OPENAI_MODEL'],
                'embedding_model': os.environ['EMBEDDING_MODEL'],
                'workers': args.workers, 'modes': args.modes,
                'repeats': args.repeats, 'load_count': args.load_count,
                'scope': 'public-only, in-process pipeline; HTTP/proxy not measured'}
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    summary = {'runs': [], 'comparisons': [], 'load': []}
    searcher = KnowledgeSearch()
    try:
        async with AsyncOpenAI(timeout=25, max_retries=0) as client:
            for repeat in range(args.repeats):
                by_mode = {}
                # Rotate order across repetitions to reduce ordering effects.
                modes = args.modes[repeat % len(args.modes):] + args.modes[:repeat % len(args.modes)]
                for mode in modes:
                    supervisor = GuardSupervisor(client, os.environ['OPENAI_MODEL'], searcher,
                                                 review_mode=mode)
                    start = time.monotonic()
                    print(f'공개 {len(public)}개 / {mode} / 반복 {repeat + 1}', flush=True)
                    records = await run_batch(supervisor, public, args.workers)
                    by_mode[mode] = records
                    stats = score_run(public, records)
                    stats.update(mode=mode, repeat=repeat + 1, review_metrics=dict(supervisor.metrics),
                                 total_seconds=round(time.monotonic() - start, 3))
                    summary['runs'].append(stats)
                    (destination / f'{mode}-{repeat + 1}.json').write_text(
                        json.dumps({'summary': stats, 'public_responses': records},
                                   ensure_ascii=False, indent=2))
                    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
                if 'off' in by_mode:
                    for mode in args.modes:
                        if mode != 'off':
                            summary['comparisons'].append({
                                'mode': mode, 'repeat': repeat + 1,
                                'changes': compare_runs(public, by_mode['off'], by_mode[mode]),
                                'note': '독립 호출 비교: 모델 변동도 포함되므로 검토자의 인과 효과로 단정하지 않습니다.',
                            })
            if args.load_count:
                load_items = [public[i % len(public)] for i in range(args.load_count)]
                for mode in args.modes:
                    supervisor = GuardSupervisor(client, os.environ['OPENAI_MODEL'], searcher,
                                                 review_mode=mode)
                    start = time.monotonic()
                    records = await run_batch(supervisor, load_items, args.workers)
                    latencies = [r['seconds'] for r in records]
                    summary['load'].append({
                        'mode': mode, 'requests': len(records), 'workers': args.workers,
                        'total_seconds': round(time.monotonic() - start, 3),
                        'latency_p95_seconds': round(percentile(latencies, .95), 3),
                        'latency_max_seconds': round(max(latencies), 3),
                        'errors': dict(Counter(r['error'] for r in records if r['error'])),
                    })
    finally:
        await searcher.close()
    after_hashes = fingerprint()
    summary['source_files_changed_during_evaluation'] = sorted(
        p for p in before_hashes.keys() | after_hashes.keys()
        if before_hashes.get(p) != after_hashes.get(p))
    summary['scope'] = manifest['scope']
    (destination / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'코드·결과 보관: {destination}')


def main():
    parser = argparse.ArgumentParser(description='공개 가드 검토 비교 및 병렬 부하 검사')
    parser.add_argument('--modes', nargs='+', choices=['off', 'all', 'selective'], default=['off', 'all'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='0은 공개 전체')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--load-count', type=int, default=0, help='추가 동시 부하 요청 수; 0은 생략')
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or args.limit < 0 or args.repeats < 1 or args.load_count < 0:
        parser.error('workers=1..64, limit/load-count>=0, repeats>=1 필요')
    args.modes = list(dict.fromkeys(args.modes))
    asyncio.run(evaluate(args))


if __name__ == '__main__':
    main()
