"""Trace fresh tutor execution of validated local public diagnostics only.

This is NOT an HTTP benchmark or a score. Hooks observe existing calls without
extra retrieval/model calls. Snapshots add overhead; elapsed time is diagnostic.
Raw public dialogue/material/query data stay in the local development cache.
_ask entry fields precede parent overrides; they are not final wire prompts.
_source_response records the outer result (e.g. VoteThree consensus), not
otherwise bypassed internal vote calls. No extra hooks are invoked to fill gaps.
No server/production import, scoring, index creation, or automatic retry/resume.
"""
import argparse
import asyncio
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from jh_tutor_generation_probe import source_data
from core.kb import load_chunks
from core.llm import LLM
from core.search import Searcher, load_vectors
from pipelines import load
from pipelines.kimjaehak.config import VARIANTS

SCOPE = 'public_inprocess_trace_not_http'
_ACTIVE = ContextVar('public_trace_request', default=None)


def snapshot(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return snapshot(asdict(value))
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Do not serialize arbitrary objects/reprs (clients may contain credentials).
    return {'unserialized_type': type(value).__name__}


def event(hook, phase, **values):
    recorder = _ACTIVE.get()
    if recorder is None:
        return
    try:
        recorder['events'].append({'hook': hook, 'phase': phase, **snapshot(values)})
    except Exception as exc:
        # Observation failure must not change the original hook's behavior.
        recorder['capture_errors'].append(type(exc).__name__)


class TraceMixin:
    def _new_state(self, body, guard):
        state = super(self._trace_hook_class, self)._new_state(body, guard)
        recorder = _ACTIVE.get()
        if recorder is not None:
            recorder['state'] = state
        return state

    async def _ask(self, state, name, system, user, effort, timeout):
        event('_ask', 'enter', name=name, entry_system=system, entry_user=user,
              entry_effort=effort, entry_timeout=timeout)
        try:
            result = await super(self._trace_hook_class, self)._ask(state, name, system, user, effort, timeout)
        except BaseException as exc:
            event('_ask', 'raise', name=name, exception=type(exc).__name__)
            raise
        event('_ask', 'return', name=name, result=result)
        return result

    async def _search(self, state, extra=''):
        event('_search', 'enter', messages=state.messages, extra=extra)
        try:
            result = await super(self._trace_hook_class, self)._search(state, extra)
        except BaseException as exc:
            event('_search', 'raise', exception=type(exc).__name__)
            raise
        event('_search', 'return', chunks=state.chunks, section=state.section,
              debug=state.debug, errors=state.errors)
        return result

    async def _resolve_query(self, state):
        event('_resolve_query', 'enter', messages=state.messages)
        try:
            result = await super(self._trace_hook_class, self)._resolve_query(state)
        except BaseException as exc:
            event('_resolve_query', 'raise', exception=type(exc).__name__)
            raise
        event('_resolve_query', 'return', query=result)
        return result

    def _source_payload(self, state, ids):
        try:
            result = super(self._trace_hook_class, self)._source_payload(state, ids)
        except BaseException as exc:
            event('_source_payload', 'raise', exception=type(exc).__name__)
            raise
        event('_source_payload', 'return', ids=ids, payload=result)
        return result

    async def _source_response(self, state, ids, timeout):
        event('_source_response', 'enter', ids=ids, timeout=timeout)
        try:
            result = await super(self._trace_hook_class, self)._source_response(state, ids, timeout)
        except BaseException as exc:
            event('_source_response', 'raise', exception=type(exc).__name__)
            raise
        event('_source_response', 'return', raw=result)
        return result


def traced_class(base):
    # Only wrap hooks the selected variant already exposes. The explicit super
    # anchor lets a per-base mixin preserve optional-method availability.
    hooks = {name: method for name, method in vars(TraceMixin).items()
             if name.startswith('_') and callable(method) and hasattr(base, name)}
    mixin = type('ExistingTraceHooks', (), hooks)
    mixin._trace_hook_class = mixin
    return type('PublicTrace_' + base.__name__, (mixin, base), {'request_slots': 64})


def tutor_body(item):
    return {'spec_version': '1.0', 'session_id': 'trace_' + uuid.uuid4().hex,
            'messages': [{'role': m['role'], 'content': m['content']}
                         for m in item['messages']]}


def checked_source(source):
    source = Path(source).resolve()
    cache = (ROOT / 'experiments/.cache').resolve()
    if source == cache or not source.is_relative_to(cache):
        raise ValueError('Source must be a completed run inside experiments/.cache')
    manifest_path = source / 'manifest.json'
    if not manifest_path.resolve().is_relative_to(cache):
        raise ValueError('Source file escapes cache')
    manifest = json.loads(manifest_path.read_text())
    result_name = 'derived_result.json' if manifest.get('scope') else 'result.json'
    files = [manifest_path, source/'responses.jsonl', source/result_name]
    if manifest.get('scope'):
        files += [source.parent/'derived_inputs.jsonl', source.parent/'derivation.json']
    for path in files:
        if not path.resolve().is_relative_to(cache) or not path.is_file():
            raise ValueError('Missing or escaped completed-source file')
    # Existing validator checks public scope, model/evaluator and dataset hash.
    manifest, items, prior, result = source_data(source)
    result_ids = [r['id'] for r in result['items']]
    if (not items or len(result_ids) != len(prior) or set(result_ids) != set(prior)
            or result.get('tutor_judge_skipped') is True):
        raise ValueError('Source result is incomplete or unscored')
    if not manifest.get('scope'):
        files += [ROOT/'public'/f'{endpoint}.jsonl'
                  for endpoint in manifest.get('endpoints', ['tutor', 'guard'])]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return source, manifest, items, hashes


def select_items(items, ids=None):
    if not ids:
        return items
    if len(set(ids)) != len(ids) or set(ids) - {i['id'] for i in items}:
        raise ValueError('IDs must be distinct tutor IDs from the verified source')
    selected = set(ids)
    return [i for i in items if i['id'] in selected]


def initialize(variant):
    cls = traced_class(load(variant))
    llm = LLM()
    chunks = load_chunks(ROOT/'kb')
    index = Path(os.environ.get('KB_INDEX', ROOT/'index/kb_embeddings.json')).resolve()
    vectors = load_vectors(index, llm.embedding_model, chunks) if llm.can_embed else None
    # Deliberately no save_vectors, _kb_vectors, or embedding warmup call.
    pipeline = cls(Searcher(chunks, vectors), llm)
    return pipeline, {'model': llm.model, 'embedding_model': llm.embedding_model,
                      'embedding_configured': llm.can_embed,
                      'embedding_index_loaded': vectors is not None,
                      'embedding_index_sha256': hashlib.sha256(index.read_bytes()).hexdigest()
                      if index.is_file() else None}


async def collect(items, pipeline, workers, dest):
    slots = asyncio.Semaphore(workers)
    with (dest/'observations.jsonl').open('x') as stream:
        async def one(item):
            async with slots:
                recorder = {'events': [], 'capture_errors': [], 'state': None}
                token = _ACTIVE.set(recorder)
                started = time.monotonic()
                row = {'id': item['id'], 'endpoint': 'tutor', 'events': recorder['events'],
                       'capture_errors': recorder['capture_errors'], 'response': None, 'exception': None}
                try:
                    state = await pipeline.tutor(tutor_body(item))
                    recorder['state'] = state
                    row['response'] = state.to_tutor_response()
                    return row
                except BaseException as exc:
                    row['exception'] = type(exc).__name__
                    raise
                finally:
                    try:
                        state = recorder['state']
                        row['final'] = snapshot({key: getattr(state, key, None) for key in
                                                 ('label', 'section', 'errors', 'debug', 'citations')})
                        row['elapsed_inprocess_seconds'] = time.monotonic() - started
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        stream.flush()
                    finally:
                        _ACTIVE.reset(token)
        tasks = [asyncio.create_task(one(item)) for item in items]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--variant', required=True, choices=VARIANTS)
    ap.add_argument('--name', required=True)
    ap.add_argument('--ids', nargs='+')
    ap.add_argument('--workers', type=int, default=64)
    args = ap.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.name) or not 1 <= args.workers <= 64:
        ap.error('Use a simple name and workers in 1..64')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=True)  # source_data validates OPENAI_MODEL.
    source, manifest, items, hashes = checked_source(args.source)
    items = select_items(items, args.ids)
    dest = ROOT/'experiments/.cache'/('jh_public_trace_' + args.name)
    dest.mkdir(exist_ok=False)
    metadata = {'scope': SCOPE, 'status': 'initializing', 'source': str(source.relative_to(ROOT)),
                'source_hashes': hashes, 'variant': args.variant, 'workers': args.workers,
                'request_slots': 64, 'count': len(items), 'selected_ids': [i['id'] for i in items],
                'source_variant': manifest['variant'], 'evaluator_hashes': manifest['evaluator_hashes'],
                'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                'dirty': bool(subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip()),
                'limits': 'Inherited pipeline deadlines/cancellation; observation overhead included',
                'trace_contract': 'Existing hook dispatch only; _ask entry fields precede parent prompt/budget overrides; _source_response is outer result, not internal VoteThree raw votes; no extra candidate calls',
                'no_http_or_judge': True, 'runtime_gold': False}
    def save():
        (dest/'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    save()
    try:
        pipeline, info = initialize(args.variant)
        metadata.update(info, status='collecting')
        save()
        asyncio.run(collect(items, pipeline, args.workers, dest))
        metadata['status'] = 'complete'
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
                        else 'failed', exception=type(exc).__name__)
        raise
    finally:
        save()
    print(dest)


if __name__ == '__main__':
    main()
