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
import time
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
                'guard_requested_action': label in ('requested', 'cancel_requested')}

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
        counts = json.loads((self.root / 'api_counts.json').read_text())
        self.assertIs(counts['instrumented'], False)
        self.assertEqual(counts['total']['started'], 0)

    def fake_counted_llm(self):
        with (self.root / 'server.py').open('a') as out:
            out.write('''
import asyncio
class LLM:
 calls=[]
 value=object()
 error=ValueError('private-error-payload')
 waiting=None
 async def chat(self, system, user, *, effort):
  self.calls.append(('chat', system, user, effort))
  if user == 'error': raise self.error
  if user == 'cancel':
   self.waiting.set()
   await asyncio.Event().wait()
  return self.value
 async def embed(self, texts, timeout):
  self.calls.append(('embed', texts, timeout))
  return self.value
''')

    def test_call_counter_preserves_results_errors_cancellation_and_arguments(self):
        self.fake_server_tree()
        self.fake_counted_llm()
        (self.root / 'uvicorn.py').write_text('''
import asyncio
import server
def run(app, **kwargs):
 async def work():
  async with app.router.lifespan_context(app):
   llm=server.LLM()
   assert await llm.chat('private-prompt', 'ok', effort='low') is llm.value
   try: await llm.chat('private-prompt', 'error', effort='low')
   except ValueError as error: assert error is llm.error
   else: raise AssertionError('Original exception swallowed')
   llm.waiting=asyncio.Event()
   task=asyncio.create_task(llm.chat('private-prompt', 'cancel', effort='low'))
   await llm.waiting.wait()
   task.cancel()
   try: await task
   except asyncio.CancelledError: pass
   else: raise AssertionError('Cancellation swallowed')
   assert await llm.embed(['private-embedding'], timeout=3) is llm.value
   assert llm.calls == [('chat','private-prompt','ok','low'),
                        ('chat','private-prompt','error','low'),
                        ('chat','private-prompt','cancel','low'),
                        ('embed',['private-embedding'],3)]
 asyncio.run(work())
''')
        result = self.run_fake_bootstrap()
        self.assertEqual(result.returncode, 0, result.stdout)
        encoded = (self.root / 'api_counts.json').read_text()
        self.assertNotIn('private', encoded)
        counts = json.loads(encoded)
        self.assertIs(counts['instrumented'], True)
        self.assertEqual(counts['chat'], {'started':3,'completed':1,'cancelled':1,'errors':1})
        self.assertEqual(counts['embed'], {'started':1,'completed':1,'cancelled':0,'errors':0})
        self.assertEqual(counts['total']['started'], 4)
        self.assertEqual(counts['startup']['chat']['started'], 0)

    def test_response_metadata_counts_effort_tokens_and_limits_without_content(self):
        self.fake_server_tree()
        with (self.root / 'server.py').open('a') as out:
            out.write('''
from types import SimpleNamespace as NS
import sys
sys.modules['pipelines.kimjaehak.round7_policy'] = NS(OTHER_BOUNDARY='public-other-rule')
class LLM:
 def __init__(self):
  async def create(**kwargs):
   return NS(choices=[NS(finish_reason='length',message=NS(content='private-answer'))],
             usage=NS(prompt_tokens=12,completion_tokens=20,completion_tokens_details=NS(reasoning_tokens=18)))
  self._chat=NS(chat=NS(completions=NS(create=create)))
 async def chat(self,system,user,*,effort):
  response=await self._chat.chat.completions.create(extra_body={'reasoning_effort':effort})
  return response.choices[0].message.content
 async def embed(self,*args,**kwargs): return []
''')
        (self.root / 'uvicorn.py').write_text('''
import asyncio,server
def run(app,**kwargs):
 async def work():
  async with app.router.lifespan_context(app):
   assert await server.LLM().chat('private-systempublic-other-rule','private-user',effort='medium')=='private-answer'
 asyncio.run(work())
''')
        result=self.run_fake_bootstrap()
        self.assertEqual(result.returncode,0,result.stdout)
        encoded=(self.root/'api_counts.json').read_text()
        self.assertNotIn('private',encoded)
        counts=json.loads(encoded)
        self.assertEqual(counts['chat_efforts']['medium'],1)
        self.assertEqual(counts['r7_policy_calls'],{'product':0,'other':1,'scaffold':0})
        self.assertEqual(counts['completion_metadata']['medium'],{'responses':1,'prompt_tokens':12,
            'completion_tokens':20,'reasoning_tokens':18,'finish_reasons':{'length':1}})

    def test_adaptive_counters_read_ground_state_before_parent_drops_debug(self):
        self.fake_server_tree()
        with (self.root/'pipelines/fake.py').open('a') as stream:
            stream.write('''
 guard_adaptive_ground=True
 async def _fusion_ground(self,state):
  state.debug={'r7_ground_early':True,'fusion_completed_ground_votes':2,'private':'not exported'}
 async def guard(self,body):
  ground=SimpleNamespace(debug={})
  await self._fusion_ground(ground)
  return SimpleNamespace(debug={})
''')
        (self.root/'uvicorn.py').write_text('''
import asyncio,server
def run(app,**kwargs):
 async def work():
  async with app.router.lifespan_context(app):
   await server.state['pipeline'].guard({'candidate_answer':'private'})
 asyncio.run(work())
''')
        result=self.run_fake_bootstrap()
        self.assertEqual(result.returncode,0,result.stdout)
        raw=(self.root/'api_counts.json').read_text()
        self.assertNotIn('private',raw)
        counts=json.loads(raw)
        self.assertEqual(counts['r7_adaptive_ground'],{'requests':1,'early':1,'completed_votes':2})
        self.assertEqual(counts['guard_debug']['r7_ground_early'],0)

    def test_guard_diagnostics_count_only_booleans_without_recording_text(self):
        self.fake_server_tree()
        with (self.root / 'pipelines/fake.py').open('a') as stream:
            stream.write('''
 async def guard(self, body):
  assert body is self.body
  return self.result
''')
        (self.root / 'uvicorn.py').write_text('''
import asyncio
from types import SimpleNamespace
import server
def run(app, **kwargs):
 async def work():
  async with app.router.lifespan_context(app):
   pipeline=server.state['pipeline']
   pipeline.body={'candidate_answer':'private text'}
   pipeline.result=SimpleNamespace(debug={'span_review_triggered':True,
       'span_review_attempted':True, 'span_review_completed':True,
       'span_review_success':True, 'span_review_changed':False,
       'r8_output_veto':True, 'r8_failure_input':True, 'r8_failure_ground':False, 'r8_failure_block':True, 'r8_risk_safety':1, 'fusion_ground_skipped_fast_block':1, 'matrix_gate':True, 'r7_examples_used':True, 'private':'private text'})
   for _ in range(2):
    assert await pipeline.guard(pipeline.body) is pipeline.result
 asyncio.run(work())
''')
        result = self.run_fake_bootstrap()
        self.assertEqual(result.returncode, 0, result.stdout)
        encoded = (self.root / 'api_counts.json').read_text()
        self.assertNotIn('private', encoded)
        counts = json.loads(encoded)['guard_debug']
        self.assertEqual(counts, {'span_review_triggered': 2, 'span_review_attempted': 2,
            'span_review_completed': 2, 'span_review_success': 2, 'span_review_changed': 0,
            'r8_output_veto': 2, 'r8_failure_input': 2, 'r8_failure_ground': 0, 'r8_failure_block': 2, 'r8_risk_safety': 0, 'fusion_ground_skipped_fast_block': 0, 'fusion_ground_cancelled_block': 0, 'matrix_gate': 2,
            'guard_unlearned_input': 0, 'learned_span': 0, 'r7_hybrid_fast': 0, 'r7_embedding_reused': 0, 'r7_ground_early': 0, 'r7_material_cancelled': 0, 'r7_examples_used': 2})

    def test_counts_flush_on_sigterm_after_original_shutdown(self):
        self.fake_server_tree()
        self.fake_counted_llm()
        (self.root / 'uvicorn.py').write_text('''
import asyncio, pathlib, signal
import server
def run(app, **kwargs):
 async def work():
  stop=asyncio.Event()
  asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
  async with app.router.lifespan_context(app):
   await server.LLM().chat('private-prompt','ok',effort='low')
   pathlib.Path('ready').write_text('ready')
   await stop.wait()
 asyncio.run(work())
''')
        proc = subprocess.Popen([sys.executable, '-B', '-c', probe.BOOTSTRAP, str(self.root),
            'jh_parallel_guard', 'pipelines.fake.Pipeline', '12345',
            str(self.root / 'identity.json'), '1'], cwd=self.root,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not (self.root / 'ready').exists() and time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(.01)
            self.assertTrue((self.root / 'ready').exists())
            proc.terminate()
            output, error = proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 0, output + error)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        counts = json.loads((self.root / 'api_counts.json').read_text())
        self.assertEqual(counts['chat']['completed'], 1)

    def test_lifespan_error_still_flushes_counts_without_exception_details(self):
        self.fake_server_tree()
        self.fake_counted_llm()
        path = self.root / 'server.py'
        path.write_text(path.read_text().replace(' yield\n', ' yield\n raise ValueError("private shutdown")\n'))
        result = self.run_fake_bootstrap()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), 'SERVER_FAILURE ValueError')
        counts = json.loads((self.root / 'api_counts.json').read_text())
        self.assertIs(counts['instrumented'], True)
        self.assertEqual(counts['total']['started'], 0)

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

    def test_collector_reads_api_counts_only_after_process_shutdown(self):
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        dest = self.root / 'run'
        process = Mock()
        socket_context = Mock()
        socket_context.__enter__ = Mock(return_value=SimpleNamespace(
            bind=Mock(), getsockname=Mock(return_value=('127.0.0.1', 12345))))
        socket_context.__exit__ = Mock(return_value=False)
        counts = {'instrumented':True,'total':{'started':8,'completed':7,'cancelled':1,'errors':0}}
        def ready(proc, port):
            probe.write_json(dest / 'startup_identity.json', self.identity('peace'))
        def shutdown(timeout):
            process.terminate.assert_called_once()
            probe.write_json(dest / 'api_counts.json', counts)
        process.wait.side_effect = shutdown
        with patch.object(probe.socket, 'socket', return_value=socket_context), \
             patch.object(probe.asyncio, 'run', side_effect=loop.run_until_complete), \
             patch.object(probe.subprocess, 'Popen', return_value=process), \
             patch.object(probe.shared, 'wait_ready', side_effect=ready), \
             patch.object(probe.shared, 'evaluator_hashes', return_value={}), \
             patch.object(probe.shared, 'collect', new=AsyncMock(return_value=[{'seconds':.1,'error':None}])):
            probe.collect_candidate('peace', self.root, [{'id':'a','endpoint':'guard'}], dest,
                                    {'TEAM_TOKEN':'fixture'}, 32, 'off')
        runtime = json.loads((dest / 'runtime.json').read_text())
        self.assertEqual(runtime['api_calls'], counts)
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

    def test_counterfactual_scoring_retains_official_result_and_adds_pair_metrics(self):
        self.check_paired_scoring('counterfactual')

    def test_relational_scoring_retains_official_input_output_and_adds_groups(self):
        self.check_paired_scoring('relational')

    def test_intent_score_only_keeps_official_scorer_and_adds_input_pair_metrics(self):
        dest = self.root / 'experiments/.cache/jh_guard_fusion_intent'
        (dest / 'peace_snapshot').mkdir(parents=True)
        run = dest / 'current'
        run.mkdir()
        items = [{'id': kind, 'endpoint': 'guard', 'pair_id': 'pair', 'contrast_kind': kind,
                  'messages': [{'role': 'user', 'content': 'synthetic request'}],
                  'labels': {'category': 'normal' if kind == 'normal' else 'off_topic',
                             'decision': 'allow' if kind == 'normal' else 'block'}}
                 for kind in ('normal', 'blocked')]
        data = ''.join(json.dumps(item)+'\n' for item in items)
        (dest/'inputs.jsonl').write_text(data)
        probe.write_json(dest/'plan.json', {'order': ['current'], 'embedding_mode': 'env',
            'dataset_sha256': probe.digest(data.encode()), 'protected_hashes': {},
            'peace_snapshot': {'files': {}}, 'dataset_provenance': {'dataset': 'intent'}})
        probe.write_json(run/'manifest.json', {'effective_config': self.identity('current'), 'evaluator_hashes': {}})
        probe.write_json(run/'runtime.json', {'collect_seconds': 1})
        rows = [{'id': item['id'], 'response': {'input': item['labels']}} for item in items]
        (run/'responses.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        metadata = {'dataset': 'intent', 'scope': 'input-only fixture', 'primary_metric': 'input_only',
                    'interpretation': 'manual development only'}
        result = {'guard': 100, 'guard_details': {'block_f1': 1, 'category_accuracy': 1,
                  'false_block': 0, 'missed_block': 0}}
        env = {name: 'synthetic' for name in ('TEAM_TOKEN', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL')}
        with patch.object(probe, 'ROOT', self.root), patch.object(probe, 'protected_hashes', return_value={}), \
             patch.object(probe.datasets, 'load_dataset', return_value=(items, metadata)), \
             patch.object(probe.shared, 'evaluator_hashes', return_value={}), \
             patch.object(probe.shared, 'score_rows', return_value=result) as score, \
             patch.object(probe, 'collect_candidate', side_effect=AssertionError('No HTTP')) as collect, \
             patch.dict(os.environ, env), redirect_stdout(io.StringIO()):
            probe.main(['--name', 'intent', '--dataset', 'intent', '--score-only', '--order', 'current'])
        score.assert_called_once_with(items, rows, 32, False, run/'judge.jsonl')
        collect.assert_not_called()
        saved = json.loads((run/'result.json').read_text())
        summary = json.loads((dest/'summary.json').read_text())[0]
        self.assertEqual(saved['guard'], 100)
        self.assertEqual(saved['guard_details'], result['guard_details'])
        self.assertEqual(saved['intent_details']['paired_both_correct'], 1)
        self.assertEqual(summary['intent_metrics']['paired_both_correct_count'], 1)
        self.assertNotIn('guard', summary)

    def test_cancel_requested_identity_requires_requested_action_flag(self):
        identity = self.identity('cancel_requested')
        probe.validate_identity(identity, 'cancel_requested', self.root)
        identity['guard_requested_action'] = False
        with self.assertRaises(ValueError):
            probe.validate_identity(identity, 'cancel_requested', self.root)

    def check_paired_scoring(self, dataset):
        dest = self.root / 'experiments/.cache/jh_guard_fusion_cf'
        (dest / 'peace_snapshot').mkdir(parents=True)
        run = dest / 'current'
        run.mkdir()
        items = [{'id':kind,'endpoint':'guard','pair_id':'pair','contrast_kind':kind,
                  'error_group':'course_relation',
                  'labels':{'grounded':kind == 'control'}} for kind in ('control','counterfactual')]
        data = ''.join(json.dumps(item)+'\n' for item in items)
        (dest / 'inputs.jsonl').write_text(data)
        probe.write_json(dest / 'plan.json', {
            'order':['current'],'embedding_mode':'env','dataset_sha256':probe.digest(data.encode()),
            'protected_hashes':{},'peace_snapshot':{'files':{}},
            'dataset_provenance':{'dataset':dataset}})
        probe.write_json(run / 'manifest.json', {'effective_config':self.identity('current'),'evaluator_hashes':{}})
        probe.write_json(run / 'runtime.json', {'collect_seconds':1})
        rows = [{'id':item['id'],'response':{'input':{'category':'normal','decision':'allow'},
                  'output':{'grounded':item['labels']['grounded'],'safe':True,'unsupported':[]}}} for item in items]
        (run / 'responses.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        metadata = {'dataset':dataset,'scope':'manual grounding diagnostic',
                    'primary_metric':'paired_grounding','interpretation':'Development only'}
        result = {'guard':90,'guard_details':{'span_f1':.6}}
        env = {name:'synthetic' for name in ('TEAM_TOKEN','OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL')}
        with patch.object(probe,'ROOT',self.root), \
             patch.object(probe,'protected_hashes',return_value={}), \
             patch.object(probe.datasets,'load_dataset',return_value=(items,metadata)), \
             patch.object(probe.shared,'evaluator_hashes',return_value={}), \
             patch.object(probe.shared,'score_rows',return_value=result) as score, \
             patch.object(probe,'collect_candidate',side_effect=AssertionError('No HTTP')) as collect, \
             patch.dict(os.environ,env), redirect_stdout(io.StringIO()):
            probe.main(['--name','cf','--dataset',dataset,'--score-only','--order','current'])
        score.assert_called_once()
        collect.assert_not_called()
        saved = json.loads((run / 'result.json').read_text())
        summary = json.loads((dest / 'summary.json').read_text())[0]
        self.assertEqual(saved['guard'],90)
        self.assertEqual(saved['guard_details']['span_f1'],.6)
        self.assertEqual(saved[dataset+'_details']['paired_both_correct'],1)
        if dataset == 'relational':
            self.assertEqual(saved['relational_details']['by_group']['course_relation']['paired_both_correct'],1)
            self.assertNotIn('counterfactual_details',saved)
        self.assertEqual(summary['paired_both_correct'],1)
        self.assertNotIn('guard',summary)


if __name__ == '__main__':
    unittest.main()
