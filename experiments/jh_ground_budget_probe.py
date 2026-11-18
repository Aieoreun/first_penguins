"""Public-only, unscored development probe of grounding completion budgets.

Prepare freezes six public examples without clients/search/API calls. Execution
uses the real CurrentSpan80 initialization and search, then freezes its top-one
grounding payload before three sequential calls per example. No classification,
voting, span refinement, scoring, retries, or production behavior changes.
Raw public payloads/responses are written only to the ignored development cache.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.kimjaehak.config import configure
configure()

from core.kb import load_chunks
from core.llm import LLM
from core.search import Searcher, load_vectors
from pipelines.kimjaehak.peace_span_variants import CurrentSpan80
from pipelines.v1.pipeline import _json_object

PUBLIC_SHA256 = '4a585c174f15dec205a286ec45e8c14ff28b01bbcedde3097072ca52e3b0c426'
SEED = 20261118
CONFIGS = (('low', 1500), ('medium', 2000), ('medium', 6000))
TIMEOUT = 20.0
SCOPE = 'public_ground_budget_development_probe_not_score'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode()


def public_items():
    data = (ROOT / 'public/guard.jsonl').read_bytes()
    if digest(data) != PUBLIC_SHA256:
        raise ValueError('Frozen public source changed')
    return [json.loads(line) for line in data.splitlines()]


def select_items(rows):
    candidates = sorted((r for r in rows if isinstance(r.get('candidate_answer'), str)),
                        key=lambda r: r['id'])
    if len(candidates) != 86 or len({r['id'] for r in candidates}) != 86:
        raise ValueError('Expected 86 distinct frozen public candidates')
    rng = random.Random(SEED)
    selected = []
    for grounded in (True, False):
        pool = [r for r in candidates if r['labels'].get('grounded') is grounded
                and r['labels'].get('category') == 'normal'
                and r['labels'].get('decision') == 'allow']
        if len(pool) < 3:
            raise ValueError('Insufficient grounded stratum')
        selected.extend(rng.sample(pool, 3))
    return selected


def provenance(items):
    kb = {str(p.relative_to(ROOT)): digest(p.read_bytes())
          for p in sorted((ROOT / 'kb').rglob('*')) if p.is_file()}
    sources = [ROOT / 'public/guard.jsonl', ROOT / 'public/tutor.jsonl', Path(__file__)]
    sources += sorted((ROOT / 'core').glob('*.py'))
    sources += sorted((ROOT / 'pipelines').rglob('*.py'))
    return dict(scope=SCOPE, public_guard_sha256=PUBLIC_SHA256,
                source_files={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sources},
                kb_files=kb, kb_manifest_sha256=digest(encoded(kb)),
                prompt_sha256=digest(CurrentSpan80.unsupported_system.encode()),
                selection_seed=SEED,
                selection_stratum='public normal/allow candidates, grounded true/false 3 each; dev only',
                selected=[{'source_id': r['id'], 'grounded_stratum': r['labels']['grounded'],
                           'item_sha256': digest(encoded(r))} for r in items],
                configs=[{'effort': e, 'max_completion_tokens': n} for e, n in CONFIGS],
                timeout=TIMEOUT, planned_chat_calls=18, planned_embedding_calls=6,
                source_policy='CurrentSpan80._search with candidate; first retrieved section; no gold evidence')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Invalid cache name')
    cache = (ROOT / 'experiments/.cache').resolve()
    dest = cache / ('jh_ground_budget_' + name)
    if dest.is_symlink() or not dest.resolve().is_relative_to(cache):
        raise ValueError('Cache path escapes development cache')
    items = select_items(public_items())
    frozen = provenance(items)
    if dest.exists():
        if {p.name for p in dest.iterdir()} != {'manifest.json'}:
            raise ValueError('Prepared cache contains collection artifacts; choose a new cache name')
        path = dest / 'manifest.json'
        if path.is_symlink():
            raise ValueError('Manifest must be a local regular file')
        manifest = json.loads(path.read_text())
        if manifest.get('provenance') != frozen or manifest.get('status') != 'prepared':
            raise ValueError('Changed or already attempted probe; choose an explicit new cache name')
    else:
        dest.mkdir(parents=True)
        manifest = {'status': 'prepared', 'api_calls': 0, 'provenance': frozen}
        write_json(dest / 'manifest.json', manifest)
    return dest, items, manifest


def request_body(item):
    # Neither IDs nor labels/evidence enter the pipeline or model payload.
    return {'spec_version': '1.0', 'session_id': 'ground-budget-dev',
            'messages': [{'role': m['role'], 'content': m['content']} for m in item['messages']],
            'candidate_answer': item['candidate_answer']}


def validate_response(raw, candidate):
    obj = _json_object(raw)
    phrases = obj.get('unsupported') if obj is not None else None
    typed = isinstance(phrases, list) and all(isinstance(p, str) and bool(p.strip()) for p in phrases)
    exact = bool(typed and all(p in candidate for p in phrases))
    return {'json_object_valid': obj is not None, 'unsupported_list_valid': typed,
            'unsupported_valid': exact}


def token_usage(usage):
    if usage is None:
        return None
    details = getattr(usage, 'completion_tokens_details', None)
    return {**{k: getattr(usage, k, None) for k in
               ('prompt_tokens', 'completion_tokens', 'total_tokens')},
            'reasoning_tokens': getattr(details, 'reasoning_tokens', None)}


async def completion(llm, system, user, candidate, effort, budget):
    started = time.monotonic()
    row = {'effort': effort, 'max_completion_tokens': budget, 'finish_reason': None,
           'usage': None, 'raw': None, 'error': None}
    try:
        # Deliberately identical to core.LLM.chat; retain the SDK result only to
        # observe finish_reason/usage. No stream, alternate endpoint, or retry.
        result = await asyncio.wait_for(llm._chat.chat.completions.create(
            model=llm.model, messages=[{'role': 'system', 'content': system},
                                       {'role': 'user', 'content': user}],
            max_completion_tokens=budget, extra_body={'reasoning_effort': effort},
            response_format={'type': 'json_object'}), TIMEOUT)
        choice = result.choices[0]
        row.update(finish_reason=choice.finish_reason, usage=token_usage(result.usage),
                   raw=(choice.message.content or '').strip())
        row.update(validate_response(row['raw'], candidate))
    except Exception as exc:
        # Exception messages may contain endpoint URLs or request bodies.
        row['error'] = type(exc).__name__
    row['seconds'] = round(time.monotonic() - started, 4)
    return row


async def collect(items, pipeline, dest, manifest):
    manifest.update(status='collecting', api_calls=0, embedding_attempts=0, chat_attempts=0,
                    api_call_count_semantics='attempted calls, including failed/cancelled calls')
    write_json(dest / 'manifest.json', manifest)
    frozen = []
    try:
        with (dest / 'payloads.jsonl').open('x') as stream:
            for item in items:
                state = pipeline._new_state(request_body(item), True)
                manifest['embedding_attempts'] += 1
                manifest['api_calls'] += 1
                write_json(dest / 'manifest.json', manifest)
                await pipeline._search(state, extra=state.candidate_answer or '')
                state.section = state.chunks[0].section_id if state.chunks else None
                payload = pipeline._ground_payload(state)
                # Serialize once; every budget sees these exact bytes.
                user = json.dumps(payload, ensure_ascii=False)
                entry = {'source_id': item['id'], 'source_section_ids': [m['section'] for m in payload['교재']],
                         'source_chunks': [{'chunk_id': c.section_id + ':' + digest(c.text.encode()),
                                            'section_id': c.section_id, 'text_sha256': digest(c.text.encode())}
                                           for c in state.chunks],
                         'payload_sha256': digest(user.encode()), 'user': user,
                         'search_errors': list(state.errors)}
                stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
                stream.flush()
                frozen.append((item, user))
                manifest.setdefault('retrievals', []).append({k: v for k, v in entry.items() if k != 'user'})
                write_json(dest / 'manifest.json', manifest)
        with (dest / 'responses.jsonl').open('x') as stream:
            for item, user in frozen:
                for effort, budget in CONFIGS:
                    manifest['chat_attempts'] += 1
                    manifest['api_calls'] += 1
                    write_json(dest / 'manifest.json', manifest)
                    row = await completion(pipeline.llm, pipeline.unsupported_system, user,
                                           item['candidate_answer'], effort, budget)
                    row['source_id'] = item['id']
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(json.dumps({k: v for k, v in row.items() if k not in ('raw', 'source_id')},
                                     ensure_ascii=False), flush=True)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='interrupted', error=type(exc).__name__)
        raise
    finally:
        write_json(dest / 'manifest.json', manifest)


async def execute(dest, items, manifest):
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=True)
    llm = LLM()
    try:
        chunks = load_chunks(ROOT / 'kb')
        index = Path(os.environ.get('KB_INDEX', ROOT / 'index/kb_embeddings.json'))
        vectors = load_vectors(index, llm.embedding_model, chunks) if llm.can_embed else None
        if vectors is None:
            raise ValueError('Existing compatible embedding index and client required; no index creation')
        pipeline = CurrentSpan80(Searcher(chunks, vectors), llm)
        manifest.update(model=llm.model, embedding_model=llm.embedding_model,
                        embedding_index_sha256=digest(index.read_bytes()))
        await collect(items, pipeline, dest, manifest)
    finally:
        await llm._chat.close()
        if llm._embed is not None:
            await llm._embed.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        dest, items, manifest = prepare(args.name)
        if args.prepare_only:
            print(json.dumps({'prepared_items': len(items), 'api_calls': 0}))
        else:
            asyncio.run(execute(dest, items, manifest))
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
