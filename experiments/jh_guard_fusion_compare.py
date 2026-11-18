"""Compare pinned public guard diagnostics and an exact Peace snapshot.

No production module imports this development recorder. Snapshot source files,
the official evaluator and both servers remain unmodified. --prepare-only never
starts a server; --collect-only checkpoints HTTP before separate scoring.
"""
import argparse
import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import jh_compare as shared
from experiments import jh_guard_fusion_datasets as datasets

PEACE_COMMIT = '3c64079eb1af14170b3bee4578ac6f916bec433e'
CANDIDATES = {
    'current': ('jh_r4_material_budget', 'pipelines.kimjaehak.material_budget.Pipeline'),
    'fusion': ('jh_pg_fusion', 'pipelines.kimjaehak.peace_guard_fusion.Pipeline'),
    'peace': ('jh_parallel_guard', 'pipelines.kimjaehak.parallel_guard.Pipeline'),
    'requested': ('jh_pg_requested', 'pipelines.kimjaehak.peace_guard_variants.RequestedAction'),
    'action': ('jh_pg_action', 'pipelines.kimjaehak.peace_safety_variants.ActionSafety'),
    'context': ('jh_pg_context', 'pipelines.kimjaehak.peace_safety_variants.ContextSafety'),
    'conditioned': ('jh_pg_conditioned', 'pipelines.kimjaehak.peace_conditioned_safety.ConditionedSafety'),
    'requested_conditioned': ('jh_pg_requested_conditioned', 'pipelines.kimjaehak.peace_conditioned_safety.RequestedConditioned'),
    'relation': ('jh_pg_relation', 'pipelines.kimjaehak.peace_ground_variants.RelationGrounding'),
    'provenance': ('jh_pg_provenance', 'pipelines.kimjaehak.peace_ground_variants.ProvenanceGrounding'),
}

# A fresh interpreter imports exclusively from the selected source tree. The
# wrapper only observes original lifespan startup; it does not replace routing.
# Peace's server ignores PIPELINE, so require its actual hard-coded route too.
BOOTSTRAP = r'''
import asyncio, contextlib, hashlib, inspect, json, logging, os, pathlib, sys
logging.disable(logging.CRITICAL)
try:
    root, variant, expected, port, identity_path, reference = sys.argv[1:]
    root = pathlib.Path(root).resolve()
    sys.path.insert(0, str(root))
    os.environ['PIPELINE'] = variant
    from pipelines import load
    selected = load(variant)
    if selected.__module__ + '.' + selected.__name__ != expected:
        raise RuntimeError('Selected pipeline class mismatch')
    if reference == '1' and getattr(selected, 'learned_models_enabled', None) is not False:
        raise RuntimeError('Peace reference must disable trained models before initialization')
    selected.request_slots = 64
    import server
    if pathlib.Path(server.__file__).resolve() != root / 'server.py' or server.PIPELINE != variant:
        raise RuntimeError('Actual server route mismatch')
    original = server.app.router.lifespan_context
    @contextlib.asynccontextmanager
    async def observed(app):
        async with original(app):
            instance = server.state['pipeline']
            actual = type(instance)
            if actual is not selected or instance.request_slots != 64 or instance._slots._value != 64:
                raise RuntimeError('Actual pipeline instance mismatch')
            if not pathlib.Path(inspect.getfile(actual)).resolve().is_relative_to(root):
                raise RuntimeError('Pipeline imported outside selected source tree')
            if reference == '1' and getattr(instance, 'learned_models_enabled', None) is not False:
                raise RuntimeError('Peace reference must disable trained models')
            if variant == 'jh_pg_requested' and getattr(instance, 'guard_requested_action', None) is not True:
                raise RuntimeError('Requested-action guard is not enabled')
            data = {'requested_variant': variant, 'server_variant': server.PIPELINE,
                    'class': actual.__module__ + '.' + actual.__name__, 'request_slots': 64,
                    'source_root': str(root), 'embedding_index_loaded': instance.searcher.has_embeddings,
                    'learned_models_enabled': getattr(instance, 'learned_models_enabled', None)}
            for key in ('guard_deadline','guard_prepare_limit','guard_material_timeout',
                        'guard_ground_timeout','guard_safety_timeout','guard_requested_action',
                        'guard_safety_action_review','guard_safety_curriculum_context',
                        'guard_safety_after_input',
                        'guard_restore_span_refinement','deadline','prepare_limit','output_votes',
                        'learned_gate_threshold'):
                data[key] = getattr(instance, key, None)
            prompts = sys.modules.get('pipelines.v2.prompts')
            block = getattr(instance, 'guard_block_system', getattr(prompts, 'BLOCK_SYSTEM', None))
            if hasattr(instance, '_block_system'):
                guard_state = instance._new_state({'messages': [
                    {'role': 'user', 'content': 'startup configuration check'}]}, True)
                block = instance._block_system(guard_state)
            safety = getattr(instance, 'guard_safety_system', None)
            if safety is None:
                method = getattr(instance, '_parallel_safety', None)
                safety = getattr(method, '__globals__', {}).get('SAFETY_SYSTEM')
            for key, text in (('guard_safety_system_sha256', safety), ('guard_block_system_sha256', block)):
                data[key] = hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None
            ground = getattr(instance, 'unsupported_system', None)
            data['unsupported_system_sha256'] = hashlib.sha256(ground.encode()).hexdigest() if isinstance(ground, str) else None
            data['guard_safety_hints_sha256'] = {k: hashlib.sha256(v.encode()).hexdigest()
                for k,v in getattr(instance, 'guard_safety_hint_systems', {}).items()}
            pathlib.Path(identity_path).write_text(json.dumps(data))
            yield
    server.app.router.lifespan_context = observed
    import uvicorn
    uvicorn.run(server.app, host='127.0.0.1', port=int(port), access_log=False, log_config=None)
except BaseException as exc:
    print('SERVER_FAILURE ' + type(exc).__name__, flush=True)
    sys.exit(1)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def archive_files(data):
    """Reject links/path traversal rather than allowing archive extraction escape."""
    result = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise ValueError('Unsafe snapshot member')
            if member.isdir():
                continue
            if not member.isfile() or member.name in result or name.name == '.env':
                raise ValueError('Invalid snapshot member')
            result[member.name] = archive.extractfile(member).read()
    return result


def verify_snapshot(dest, expected):
    actual = {str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}
    if actual != set(expected):
        raise ValueError('Snapshot file inventory changed')
    for name, checksum in expected.items():
        path = dest / name
        if path.is_symlink() or not path.resolve().is_relative_to(dest.resolve()) or digest(path.read_bytes()) != checksum:
            raise ValueError('Snapshot file changed')


def prepare_snapshot(root, dest):
    if not dest.resolve().is_relative_to((root / 'experiments/.cache').resolve()):
        raise ValueError('Snapshot must stay in ignored experiment cache')
    commit = subprocess.check_output(['git', 'rev-parse', PEACE_COMMIT + '^{commit}'], cwd=root, text=True).strip()
    if commit != PEACE_COMMIT:
        raise ValueError('Unexpected reference commit')
    archive = subprocess.check_output(['git', 'archive', '--format=tar', commit], cwd=root)
    files = archive_files(archive)
    expected = {name: digest(data) for name, data in sorted(files.items())}
    if not dest.exists():
        dest.mkdir(parents=True)
        for name, data in files.items():
            path = dest / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    verify_snapshot(dest, expected)
    return {'commit': commit, 'archive_sha256': digest(archive), 'files': expected}


def protected_hashes(root):
    paths = [root / 'SPEC.md', root / 'server.py', root / 'harness/score_local.py']
    for folder in ('core', 'grader', 'kb', 'public'):
        paths += [p for p in (root / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in sorted(paths)}


def pipeline_hashes(root):
    return {str(p.relative_to(root)): digest(p.read_bytes())
            for p in sorted((root / 'pipelines').rglob('*.py'))}


def verify_common_inputs(root, reference, snapshot):
    """The reference may have a different server, never different KB/gold/grader."""
    current = protected_hashes(root)
    for name, checksum in current.items():
        if name != 'server.py' and snapshot['files'].get(name) != checksum:
            raise ValueError('Reference core/evaluator/KB/public differs: ' + name)
    return current


def validate_identity(value, label, source_root):
    variant, expected = CANDIDATES[label]
    if (value.get('requested_variant') != variant or value.get('server_variant') != variant
            or value.get('class') != expected or value.get('request_slots') != 64
            or value.get('source_root') != str(source_root.resolve())
            or (label == 'peace' and value.get('learned_models_enabled') is not False)
            or (label == 'requested' and value.get('guard_requested_action') is not True)):
        raise ValueError('Recorded startup identity does not match requested candidate')


def server_env(env, root, embedding_mode):
    result = dict(env)
    result.pop('PYTHONPATH', None)
    result['PYTHONDONTWRITEBYTECODE'] = '1'
    result['KB_DIR'] = str(root / 'kb')
    result['KB_INDEX'] = str(root / 'index/kb_embeddings.json')
    if embedding_mode == 'off':
        for key in ('EMBEDDING_BASE_URL', 'EMBEDDING_MODEL', 'EMBEDDING_API_KEY'):
            result[key] = ''
    return result


def collect_candidate(label, source_root, items, dest, env, judge_workers, embedding_mode, dataset_metadata=None):
    """Small process adapter; requests and official scoring are shared, not copied."""
    dest.mkdir()
    variant, expected = CANDIDATES[label]
    identity_path = dest / 'startup_identity.json'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    manifest = {'variant': variant, 'candidate': label, 'expected_class': expected,
                'workers': 64, 'request_slots_override': 64, 'judge_workers': judge_workers,
                'count': len(items), 'endpoints': ['guard'], 'score_scope': 'guard',
                'evaluator_hashes': shared.evaluator_hashes(), 'embedding_mode': embedding_mode,
                'scoring_deferred': True, 'judge_enabled': True}
    if dataset_metadata is not None:
        manifest['dataset_provenance'] = dataset_metadata
        manifest['scope'] = dataset_metadata['scope']
    write_json(dest / 'manifest.json', manifest)
    command = [sys.executable, '-B', '-c', BOOTSTRAP, str(source_root), variant, expected,
               str(port), str(identity_path), '1' if label == 'peace' else '0']
    with (dest / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=source_root, env=env, stdout=log, stderr=log)
        try:
            shared.wait_ready(proc, port)
            identity = json.loads(identity_path.read_text())
            validate_identity(identity, label, source_root)
            manifest['effective_config'] = identity
            write_json(dest / 'manifest.json', manifest)
            started = time.monotonic()
            rows = asyncio.run(shared.collect(items, port, env['TEAM_TOKEN'], 64, dest / 'responses.jsonl'))
            seconds = round(time.monotonic() - started, 3)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    durations = sorted(row['seconds'] for row in rows)
    errors = re.findall(r'errors=([^\s]+)', (dest / 'server.log').read_text())
    runtime = {'collect_seconds': seconds, 'p50': statistics.median(durations),
               'p95': durations[min(len(durations)-1, int(.95*len(durations)))], 'max': max(durations),
               'http_errors': sum(bool(row['error']) for row in rows),
               'stage_failure_requests': sum(error != '-' for error in errors)}
    write_json(dest / 'runtime.json', runtime)
    if shared.evaluator_hashes() != manifest['evaluator_hashes']:
        raise RuntimeError('Official evaluator changed during collection')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--dataset', choices=datasets.DATASETS, default='public')
    parser.add_argument('--order', nargs='+', choices=CANDIDATES, default=['current','fusion','peace'])
    parser.add_argument('--embedding-mode', choices=('env', 'off'), default='env')
    parser.add_argument('--judge-workers', type=int, default=32)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument('--prepare-only', action='store_true')
    phase.add_argument('--collect-only', action='store_true')
    phase.add_argument('--score-only', action='store_true')
    args = parser.parse_args(argv)
    if (not re.fullmatch(r'[A-Za-z0-9_-]+', args.name) or len(set(args.order)) != len(args.order)
            or not 1 <= args.judge_workers <= 64):
        parser.error('Invalid name, order or judge workers')
    dest = ROOT / 'experiments/.cache' / ('jh_guard_fusion_' + args.name)
    if not args.score_only:
        dest.mkdir(exist_ok=False)
        snapshot = prepare_snapshot(ROOT, dest / 'peace_snapshot')
        protected = verify_common_inputs(ROOT, dest / 'peace_snapshot', snapshot)
        items, dataset_metadata = datasets.load_dataset(args.dataset, ROOT)
        data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in items)
        (dest / 'inputs.jsonl').write_text(data)
        plan = {'scope': dataset_metadata['scope'], 'dataset_provenance': dataset_metadata, 'order': args.order,
                'peace_snapshot': snapshot, 'protected_hashes': protected,
                'pipeline_hashes': pipeline_hashes(ROOT), 'bootstrap_sha256': digest(BOOTSTRAP.encode()),
                'runner_sha256': digest(Path(__file__).read_bytes()),
                'dataset_module_sha256': digest(Path(datasets.__file__).read_bytes()),
                'dataset_sha256': digest(data.encode()), 'embedding_mode': args.embedding_mode,
                'current_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'environment_policy': 'One inherited environment for all children; no secrets recorded'}
        write_json(dest / 'plan.json', plan)
    else:
        plan = json.loads((dest / 'plan.json').read_text())
        if plan['order'] != args.order or plan['embedding_mode'] != args.embedding_mode:
            raise ValueError('Resume settings differ from frozen plan')
        data = (dest / 'inputs.jsonl').read_bytes()
        if digest(data) != plan['dataset_sha256']:
            raise ValueError('Frozen public dataset changed')
        items = [json.loads(line) for line in data.splitlines()]
        expected_items, dataset_metadata = datasets.load_dataset(args.dataset, ROOT)
        if (items != expected_items
                or plan.get('dataset_provenance', {}).get('dataset', 'public') != args.dataset):
            raise ValueError('Resume dataset differs from pinned inputs')
    current_protected = protected_hashes(ROOT)
    if args.score_only:
        # Cached responses no longer execute pipeline/server/recorder code.
        # Retain their recorded code hashes, but permit newer experimental code.
        scoring_files = lambda rows: {k: v for k, v in rows.items()
                                     if k.startswith(('grader/', 'harness/', 'kb/', 'public/'))}
        protected_equal = scoring_files(current_protected) == scoring_files(plan['protected_hashes'])
    else:
        protected_equal = current_protected == plan['protected_hashes']
    if not protected_equal:
        raise ValueError('Protected current files changed')
    # Older public plans did not list SPEC separately; their exact reference
    # snapshot still pins its original content for resumed scoring.
    expected_spec = plan['protected_hashes'].get('SPEC.md', plan['peace_snapshot']['files'].get('SPEC.md'))
    if expected_spec is not None and digest((ROOT / 'SPEC.md').read_bytes()) != expected_spec:
        raise ValueError('Original competition specification changed')
    if not args.score_only and (pipeline_hashes(ROOT) != plan['pipeline_hashes']
            or digest(BOOTSTRAP.encode()) != plan['bootstrap_sha256']
            or digest(Path(__file__).read_bytes()) != plan['runner_sha256']
            or digest(Path(datasets.__file__).read_bytes()) != plan['dataset_module_sha256']):
        raise ValueError('Experiment implementation changed after preparation')
    verify_snapshot(dest / 'peace_snapshot', plan['peace_snapshot']['files'])
    if args.prepare_only:
        print(json.dumps({'prepared': str(dest), 'count': len(items), 'api_calls': 0}))
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=True)
    for key in ('TEAM_TOKEN', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL'):
        if not os.environ.get(key):
            raise ValueError('Required environment setting missing: ' + key)
    env = server_env(os.environ, ROOT, args.embedding_mode)
    if not args.score_only:
        for label in args.order:
            source = dest / 'peace_snapshot' if label == 'peace' else ROOT
            collect_candidate(label, source, items, dest / label, env, args.judge_workers,
                              args.embedding_mode, dataset_metadata)
            verify_snapshot(dest / 'peace_snapshot', plan['peace_snapshot']['files'])
            if protected_hashes(ROOT) != plan['protected_hashes']:
                raise ValueError('Protected files changed during collection')
            if pipeline_hashes(ROOT) != plan['pipeline_hashes']:
                raise ValueError('Candidate implementation changed during collection')
    if args.collect_only:
        print('SCORING_PENDING', dest)
        return
    summary = []
    for label in args.order:
        run = dest / label
        manifest = json.loads((run / 'manifest.json').read_text())
        source = dest / 'peace_snapshot' if label == 'peace' else ROOT
        validate_identity(manifest['effective_config'], label, source)
        if manifest['evaluator_hashes'] != shared.evaluator_hashes():
            raise ValueError('Scorer differs from collection')
        rows = [json.loads(line) for line in (run / 'responses.jsonl').read_text().splitlines()]
        result = shared.score_rows(items, rows, args.judge_workers, False, run / 'judge.jsonl')
        if shared.evaluator_hashes() != manifest['evaluator_hashes']:
            raise ValueError('Official evaluator changed during scoring')
        result['score_scope'] = 'guard'
        result['diagnostic_scope'] = dataset_metadata['scope']
        result['dataset_provenance'] = dataset_metadata
        result['runtime'] = json.loads((run / 'runtime.json').read_text())
        write_json(run / 'result.json', result)
        summary.append(datasets.summarize_result(label, result, dataset_metadata))
    write_json(dest / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('COMPARISON_FAILURE ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
