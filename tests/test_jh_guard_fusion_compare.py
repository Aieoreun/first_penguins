import asyncio
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from experiments import jh_guard_fusion_compare as probe


def archive(entries, symlink=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as output:
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            if symlink:
                member.type, member.linkname = tarfile.SYMTYPE, '/tmp/outside'
            else:
                member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    return stream.getvalue()


class GuardFusionCompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'experiments/.cache').mkdir(parents=True)

    def identity(self, label, root=None):
        variant, cls = probe.CANDIDATES[label]
        return {'requested_variant': variant, 'server_variant': variant, 'class': cls,
                'request_slots': 64, 'source_root': str(root or self.root),
                'learned_models_enabled': False if label == 'peace' else None,
                'guard_requested_action': label == 'requested'}

    def test_snapshot_archive_hashes_exact_content_and_detects_edits_or_extras(self):
        payload = archive({'server.py': b'original server', 'core/llm.py': b'no edits'})
        dest = self.root / 'experiments/.cache/reference'
        with patch.object(probe.subprocess, 'check_output', side_effect=[probe.PEACE_COMMIT+'\n', payload]):
            meta = probe.prepare_snapshot(self.root, dest)
        self.assertEqual(meta['commit'], probe.PEACE_COMMIT)
        self.assertEqual(meta['archive_sha256'], probe.digest(payload))
        self.assertEqual((dest / 'server.py').read_bytes(), b'original server')
        probe.verify_snapshot(dest, meta['files'])
        (dest / 'server.py').write_bytes(b'edited')
        with self.assertRaises(ValueError):
            probe.verify_snapshot(dest, meta['files'])
        (dest / 'server.py').write_bytes(b'original server')
        (dest / 'injected.py').write_text('unexpected')
        with self.assertRaises(ValueError):
            probe.verify_snapshot(dest, meta['files'])

    def test_archive_traversal_links_and_env_are_rejected(self):
        for name, link in (('../escape', False), ('/absolute', False), ('.env', False), ('link', True)):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    probe.archive_files(archive({name: b'value'}, symlink=link))

    def test_snapshot_only_writes_ignored_cache_and_pins_full_commit(self):
        with self.assertRaises(ValueError):
            probe.prepare_snapshot(self.root, self.root / 'production')
        with patch.object(probe.subprocess, 'check_output', return_value='anothercommit'):
            with self.assertRaises(ValueError):
                probe.prepare_snapshot(self.root, self.root / 'experiments/.cache/reference')

    def test_identity_refuses_cli_class_mismatch_training_and_wrong_semaphore(self):
        valid = self.identity('peace')
        probe.validate_identity(valid, 'peace', self.root)
        for key, value in (('requested_variant', 'jh_r4_material_budget'),
                           ('server_variant', 'jh_r4_material_budget'),
                           ('class', 'pipelines.kimjaehak.material_budget.Pipeline'),
                           ('request_slots', 32), ('learned_models_enabled', True),
                           ('source_root', '/different')):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    probe.validate_identity({**valid, key: value}, 'peace', self.root)

    def test_server_env_preserves_credentials_without_mutating_parent(self):
        original = {'OPENAI_API_KEY': 'fixture-secret', 'TEAM_TOKEN': 'fixture-token',
                    'PYTHONPATH': '/unrelated', 'EMBEDDING_MODEL': 'fixture-embedding',
                    'EMBEDDING_BASE_URL': 'https://fixture.invalid'}
        env = probe.server_env(original, self.root, 'off')
        self.assertEqual(env['OPENAI_API_KEY'], original['OPENAI_API_KEY'])
        self.assertEqual(env['TEAM_TOKEN'], original['TEAM_TOKEN'])
        self.assertEqual(env['EMBEDDING_MODEL'], '')
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')
        self.assertEqual(original['EMBEDDING_MODEL'], 'fixture-embedding')
        self.assertEqual(env['KB_DIR'], str(self.root / 'kb'))

    def test_requested_candidate_identity_requires_enabled_action_rubric(self):
        identity = self.identity('requested')
        probe.validate_identity(identity, 'requested', self.root)
        identity['guard_requested_action'] = False
        with self.assertRaises(ValueError):
            probe.validate_identity(identity, 'requested', self.root)

    def fake_server_tree(self, variant='jh_parallel_guard', trained=False):
        package = self.root / 'pipelines'
        package.mkdir()
        (package / '__init__.py').write_text('from .fake import Pipeline\ndef load(name): return Pipeline\n')
        (package / 'fake.py').write_text(
            'import asyncio\nfrom types import SimpleNamespace\n'
            'class Pipeline:\n request_slots=32\n learned_models_enabled='+repr(trained)+'\n'
            ' guard_safety_system="synthetic safety"\n guard_block_system="synthetic block"\n'
            ' def _new_state(self, body, guard): return SimpleNamespace(guard=guard)\n'
            ' def _block_system(self, state):\n  assert state.guard\n  return "resolved synthetic block"\n'
            ' def __init__(self):\n  self._slots=asyncio.Semaphore(self.request_slots)\n'
            '  self.searcher=SimpleNamespace(has_embeddings=False)\n')
        (self.root / 'server.py').write_text(
            'from contextlib import asynccontextmanager\nfrom types import SimpleNamespace\n'
            'from pipelines import load\nPIPELINE='+repr(variant)+'\nstate={}\n'
            '@asynccontextmanager\nasync def original(app):\n'
            ' state["pipeline"]=load(PIPELINE)()\n yield\n'
            'app=SimpleNamespace(router=SimpleNamespace(lifespan_context=original))\n')
        (self.root / 'uvicorn.py').write_text(
            'import asyncio\ndef run(app, **kwargs):\n'
            ' async def work():\n  async with app.router.lifespan_context(app): pass\n'
            ' asyncio.run(work())\n')

    def run_fake_bootstrap(self, reference='1'):
        return subprocess.run([sys.executable, '-B', '-c', probe.BOOTSTRAP, str(self.root),
            'jh_parallel_guard', 'pipelines.fake.Pipeline', '12345',
            str(self.root / 'identity.json'), reference], cwd=self.root,
            capture_output=True, text=True, timeout=10)

    def test_bootstrap_records_actual_instance_from_original_lifespan_without_api(self):
        self.fake_server_tree()
        before = (self.root / 'server.py').read_bytes()
        result = self.run_fake_bootstrap()
        self.assertEqual(result.returncode, 0, result.stdout)
        identity = json.loads((self.root / 'identity.json').read_text())
        self.assertEqual(identity['class'], 'pipelines.fake.Pipeline')
        self.assertEqual(identity['server_variant'], 'jh_parallel_guard')
        self.assertEqual(identity['request_slots'], 64)
        self.assertIs(identity['learned_models_enabled'], False)
        self.assertEqual(identity['guard_safety_system_sha256'], probe.digest(b'synthetic safety'))
        self.assertEqual(identity['guard_block_system_sha256'], probe.digest(b'resolved synthetic block'))
        self.assertEqual((self.root / 'server.py').read_bytes(), before)

    def test_bootstrap_refuses_server_ignoring_selection_before_initialization(self):
        self.fake_server_tree(variant='ignored_cli')
        result = self.run_fake_bootstrap()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), 'SERVER_FAILURE RuntimeError')
        self.assertFalse((self.root / 'identity.json').exists())

    def test_bootstrap_refuses_trained_peace_reference(self):
        self.fake_server_tree(trained=True)
        result = self.run_fake_bootstrap()
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root / 'identity.json').exists())

    def test_wrong_startup_identity_never_reaches_collector_and_process_stops(self):
        dest = self.root / 'run'
        process = Mock()
        socket_context = Mock()
        socket_context.__enter__ = Mock(return_value=SimpleNamespace(
            bind=Mock(), getsockname=Mock(return_value=('127.0.0.1', 12345))))
        socket_context.__exit__ = Mock(return_value=False)
        def ready(proc, port):
            probe.write_json(dest / 'startup_identity.json', {**self.identity('peace'), 'class': 'wrong'})
        with patch.object(probe.socket, 'socket', return_value=socket_context), \
             patch.object(probe.subprocess, 'Popen', return_value=process), \
             patch.object(probe.shared, 'wait_ready', side_effect=ready), \
             patch.object(probe.shared, 'evaluator_hashes', return_value={}), \
             patch.object(probe.shared, 'collect', new=AsyncMock()) as collect:
            with self.assertRaises(ValueError):
                probe.collect_candidate('peace', self.root, [], dest, {'TEAM_TOKEN':'fixture'}, 32, 'off')
        collect.assert_not_awaited()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10)

    def test_prepare_only_builds_full_guard_plan_without_server_or_scoring(self):
        rows = [{'id':str(i), 'endpoint':'guard', 'messages':[{'role':'user','content':'fixture'}],
                 'labels':{'category':'normal'}} for i in range(200)]
        def snapshot(root, dest):
            dest.mkdir()
            return {'commit':probe.PEACE_COMMIT, 'files':{}, 'archive_sha256':'fixture'}
        with patch.object(probe, 'ROOT', self.root), \
             patch.object(probe, 'prepare_snapshot', side_effect=snapshot), \
             patch.object(probe, 'verify_common_inputs', return_value={}), \
             patch.object(probe, 'protected_hashes', return_value={}), \
             patch.object(probe.datasets, 'load_dataset', return_value=(rows, {
                 'dataset':'public', 'scope':'public_guard200_comparison_not_submission_total'})), \
             patch.object(probe.subprocess, 'check_output', return_value='fixture-head\n'), \
             patch.object(probe, 'collect_candidate', side_effect=AssertionError('No HTTP')) as collect, \
             patch.object(probe.shared, 'score_rows', side_effect=AssertionError('No API')) as score, \
             redirect_stdout(io.StringIO()):
            probe.main(['--name','synthetic','--prepare-only'])
        collect.assert_not_called()
        score.assert_not_called()
        dest = self.root / 'experiments/.cache/jh_guard_fusion_synthetic'
        plan = json.loads((dest / 'plan.json').read_text())
        self.assertEqual(plan['dataset_sha256'], probe.digest((dest / 'inputs.jsonl').read_bytes()))
        self.assertEqual(len((dest / 'inputs.jsonl').read_text().splitlines()), 200)
        self.assertEqual(plan['order'], ['current','fusion','peace'])

    def test_score_only_allows_new_pipeline_code_but_uses_frozen_responses_and_grader(self):
        dest = self.root / 'experiments/.cache/jh_guard_fusion_old'
        (dest / 'peace_snapshot').mkdir(parents=True)
        run = dest / 'current'
        run.mkdir()
        items = [{'id':'a','endpoint':'guard','messages':[{'role':'user','content':'fixture'}]}]
        data = json.dumps(items[0])+'\n'
        (dest / 'inputs.jsonl').write_text(data)
        probe.write_json(dest / 'plan.json', {
            'order':['current'],'embedding_mode':'env','dataset_sha256':probe.digest(data.encode()),
            'protected_hashes':{'server.py':'old','core/llm.py':'old'},
            'pipeline_hashes':{'old':'implementation'},'runner_sha256':'old',
            'peace_snapshot':{'files':{}},
        })
        probe.write_json(run / 'manifest.json', {'effective_config':self.identity('current'), 'evaluator_hashes':{}})
        probe.write_json(run / 'runtime.json', {'collect_seconds':1})
        (run / 'responses.jsonl').write_text(json.dumps({'id':'a','response':{}})+'\n')
        metadata = {'dataset':'public','scope':'public diagnostic','primary_metric':'guard'}
        result = {'guard':90,'guard_details':{'safe_accuracy':1}}
        env = {name:'synthetic' for name in ('TEAM_TOKEN','OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL')}
        with patch.object(probe, 'ROOT', self.root), \
             patch.object(probe, 'protected_hashes', return_value={'server.py':'new','core/llm.py':'new'}), \
             patch.object(probe.datasets, 'load_dataset', return_value=(items,metadata)), \
             patch.object(probe.shared, 'evaluator_hashes', return_value={}), \
             patch.object(probe.shared, 'score_rows', return_value=result) as score, \
             patch.object(probe, 'collect_candidate', side_effect=AssertionError('Must not recollect')) as collect, \
             patch.dict(os.environ, env), redirect_stdout(io.StringIO()):
            probe.main(['--name','old','--score-only','--order','current'])
        collect.assert_not_called()
        score.assert_called_once()
        saved = json.loads((run / 'result.json').read_text())
        self.assertEqual(saved['guard'],90)
        self.assertEqual(saved['diagnostic_scope'],'public diagnostic')


if __name__ == '__main__':
    unittest.main()
