"""Public-only, serial endpoint evaluation using the unchanged reference scorer.

Each endpoint/repetition gets a fresh reference server. HTTP has one absolute
30-second deadline, and tutor judging starts only after that server has stopped.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import jh_compare


def fingerprints():
    paths = {ROOT / 'server.py', Path(__file__).resolve()}
    for folder in ('core', 'pipelines', 'grader', 'harness'):
        paths.update((ROOT / folder).rglob('*.py'))
    for folder in ('kb', 'index', 'public'):
        paths.update(p for p in (ROOT / folder).rglob('*') if p.is_file())
    paths.add(ROOT / 'experiments/jh_server.py')
    paths.add(ROOT / 'experiments/jh_compare.py')
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}


async def read_body(reader, headers, limit=256 * 1024):
    body = bytearray()
    if 'chunked' in headers.get('transfer-encoding', '').lower():
        while True:
            line = await reader.readline()
            size = int(line.split(b';', 1)[0].strip(), 16)
            if size < 0 or len(body) + size > limit:
                raise ValueError('invalid_body_size')
            if size == 0:
                while await reader.readline() not in (b'\r\n', b'\n', b''):
                    pass
                return bytes(body)
            body.extend(await reader.readexactly(size))
            if await reader.readexactly(2) != b'\r\n':
                raise ValueError('invalid_chunk_end')
    elif 'content-length' in headers:
        size = int(headers['content-length'])
        if size < 0 or size > limit:
            raise ValueError('invalid_body_size')
        return await reader.readexactly(size)
    else:
        while True:
            chunk = await reader.read(min(65536, limit + 1 - len(body)))
            if not chunk:
                return bytes(body)
            body.extend(chunk)
            if len(body) > limit:
                raise ValueError('invalid_body_size')


async def request(item, port, token, deadline=30):
    request_id = uuid.uuid4().hex
    payload = {'spec_version': '1.0', 'session_id': 'public-' + request_id,
               'messages': item['messages']}
    if 'candidate_answer' in item:
        payload['candidate_answer'] = item['candidate_answer']
    row = {'id': item['id'], 'endpoint': item['endpoint'], 'request_id': request_id,
           'response': None, 'status': None, 'error': None, 'deadline_seconds': deadline}
    started = time.monotonic()
    writer = None
    try:
        # Includes connect, send, headers, and the entire response body; not a
        # per-read socket timeout. Semaphore queueing happens before this clock.
        async with asyncio.timeout(deadline):
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            head = (f'POST /{item["endpoint"]} HTTP/1.1\r\n'
                    f'Host: 127.0.0.1:{port}\r\nAuthorization: Bearer {token}\r\n'
                    'Content-Type: application/json\r\nConnection: close\r\n'
                    f'Content-Length: {len(data)}\r\nX-Public-Eval-ID: {request_id}\r\n\r\n')
            writer.write(head.encode() + data)
            await writer.drain()
            raw = await reader.readuntil(b'\r\n\r\n')
            lines = raw.decode('iso-8859-1').split('\r\n')
            row['status'] = int(lines[0].split()[1])
            headers = dict((k.strip().lower(), v.strip()) for k, v in
                           (line.split(':', 1) for line in lines[1:] if line))
            body = await read_body(reader, headers)
            row['body_received_seconds'] = round(time.monotonic() - started, 6)
        if row['status'] != 200:
            row['error'] = f'HTTPError:{row["status"]}'
        else:
            row['response'] = json.loads(body)
    except Exception as exc:
        row['error'] = type(exc).__name__
    finally:
        if writer is not None:
            writer.close()
    row['seconds'] = round(time.monotonic() - started, 6)
    return row


async def collect(items, port, token, workers, dest):
    gate = asyncio.Semaphore(workers)
    before = fingerprints()
    (dest.parent / 'fingerprints-before.json').write_text(json.dumps(before, indent=2))
    completed = 0
    with dest.open('w', encoding='utf-8') as stream:
        async def bounded(item):
            nonlocal completed
            async with gate:
                row = await request(item, port, token)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            completed += 1
            if completed % 25 == 0:
                print(f'collected {completed}/{len(items)}', flush=True)
            return row
        tasks = [asyncio.create_task(bounded(item)) for item in items]
        try:
            rows = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    after = fingerprints()
    (dest.parent / 'fingerprints-after-http.json').write_text(json.dumps(after, indent=2))
    if before != after:
        raise RuntimeError('Source or data changed during HTTP evaluation')
    return rows


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='jh_r4_material_budget')
    parser.add_argument('--endpoints', nargs='+', choices=('tutor', 'guard'), default=['tutor', 'guard'])
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--request-slots', type=int, default=64)
    parser.add_argument('--judge-workers', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.endpoints)) != len(args.endpoints):
        parser.error('Duplicate endpoint')
    if not 1 <= args.workers <= 64 or not 1 <= args.request_slots <= 64 or args.repeat < 1:
        parser.error('workers/request-slots must be 1..64, repeat must be positive')
    for key in ('OPENAI_BASE_URL', 'OPENAI_MODEL', 'OPENAI_API_KEY', 'TEAM_TOKEN'):
        if not os.environ.get(key):
            parser.error(f'Missing setting: {key}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.embedding_mode = 'env'
    args.collect_only = False
    args.no_judge = False  # Tutor must use the real judge; guard does not call it.
    args.category = None
    endpoints = list(args.endpoints)
    jh_compare.collect = collect
    summary = []
    initial = fingerprints()
    (args.output_dir / 'fingerprints-initial.json').write_text(json.dumps(initial, indent=2))
    for endpoint in endpoints:
        args.endpoints = [endpoint]
        items = jh_compare.load_public(endpoints=(endpoint,))
        if len(items) != 200:
            raise ValueError(f'Expected exactly 200 public {endpoint} items')
        endpoint_dir = args.output_dir / endpoint
        endpoint_dir.mkdir()
        for repetition in range(1, args.repeat + 1):
            result = jh_compare.run_variant(args, args.variant, repetition, items, endpoint_dir)
            run = endpoint_dir / f'{args.variant}_{repetition}'
            rows = [json.loads(line) for line in (run / 'responses.jsonl').read_text().splitlines()]
            runtime = result['runtime']
            runtime.update(mean=sum(row['seconds'] for row in rows) / len(rows),
                           client_deadline_timeouts=sum(row['error'] == 'TimeoutError' for row in rows),
                           request_error_types=dict(Counter(row['error'] for row in rows if row['error'])),
                           successful_bodies_over_30s=sum(row.get('body_received_seconds', 0) > 30
                                                         for row in rows if row['response'] is not None))
            after = fingerprints()
            before = json.loads((run / 'fingerprints-before.json').read_text())
            changed = [p for p in sorted(before.keys() | after.keys()) if before.get(p) != after.get(p)]
            (run / 'fingerprints-after.json').write_text(json.dumps(after, indent=2))
            full = json.loads((run / 'result.json').read_text())
            full.update(runtime=runtime, source_files_changed_during_evaluation=changed)
            (run / 'result.json').write_text(json.dumps(full, ensure_ascii=False, indent=2))
            (run / 'runtime.json').write_text(json.dumps(runtime, indent=2))
            manifest = json.loads((run / 'manifest.json').read_text())
            manifest.update(http_transport='asyncio_streams_absolute_deadline', deadline_seconds=30)
            (run / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            result.update(endpoint=endpoint, source_files_changed_during_evaluation=changed)
            summary.append(result)
            (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            if changed:
                raise RuntimeError('Source or data changed during scoring')
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print('RESULT_DIR', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
