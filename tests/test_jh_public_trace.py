import asyncio
from copy import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from core.schema import Label
from core.state import Chunk, State
from experiments import jh_public_trace as trace


class FakePipeline:
    request_slots = 3

    def __init__(self):
        self.calls = []
        self.quote = {'quote': 'original material', 'section': 'S'}

    def _new_state(self, body, guard):
        self.calls.append(('new', body, guard))
        return State.from_request(body)

    async def _ask(self, state, name, system, user, effort, timeout):
        self.calls.append(('ask', name, system, user, effort, timeout))
        await asyncio.sleep(0)
        return self.quote

    async def _resolve_query(self, state):
        self.calls.append(('resolve', state.last_user))
        await asyncio.sleep(0)
        return state.last_user + ' resolved'

    async def _search(self, state, extra=''):
        self.calls.append(('search', state.last_user, extra))
        await self._resolve_query(state)
        state.chunks = [Chunk('S', state.last_user + ' original material')]

    def _source_payload(self, state, ids):
        self.calls.append(('payload', list(ids)))
        return {'conversation': state.messages, 'materials': [{'section': 'S', 'text': state.chunks[0].text}]}

    async def _candidate_sections(self, state):
        self.calls.append(('candidates',))
        return ['S']

    async def _source_response(self, state, ids, timeout):
        self.calls.append(('source', list(ids), timeout))
        self._source_payload(state, ids)
        return '{"section":"S"}'

    async def tutor(self, body):
        state = self._new_state(body, False)
        await self._search(state, extra='same extra')
        await self._ask(state, 'material', 'system', 'user', 'low', 13)
        ids = await self._candidate_sections(state)
        await self._source_response(state, ids, timeout=8)
        state.label, state.section = Label.NORMAL, 'S'
        state.answer, state.citations = state.chunks[0].text, ['S']
        return state


def item(name='public1'):
    return {'id': name, 'endpoint': 'tutor', 'messages': [{'role': 'user', 'content': name}],
            'labels': {'category': 'normal', 'gold_facts': ['GOLD_SENTINEL'], 'evidence': ['GOLD_ID']},
            'candidate_answer': 'not allowed', 'session_id': 'not allowed'}


class TraceHooksTests(unittest.IsolatedAsyncioTestCase):
    def test_joint_adapter_trace_preserves_identity_calls_and_exceptions(self):
        class Adapter:
            def _adapt_joint_result(self, result, ids):
                self.calls += 1
                if 'error' in result:
                    raise result['error']
                return result
        p = trace.traced_class(Adapter)()
        p.calls = 0
        recorder = {'events': [], 'capture_errors': []}
        token = trace._ACTIVE.set(recorder)
        try:
            raw = {'supported': True, 'section': 'A', 'quote': 'public quote'}
            self.assertIs(p._adapt_joint_result(raw, ['A']), raw)
            exc = ValueError('original')
            with self.assertRaises(ValueError) as raised:
                p._adapt_joint_result({'error': exc}, ['A'])
            self.assertIs(raised.exception, exc)
        finally:
            trace._ACTIVE.reset(token)
        self.assertEqual(p.calls, 2)
        self.assertEqual(recorder['events'][0]['result'], raw)
        self.assertEqual(recorder['events'][-1]['phase'], 'raise')
        self.assertEqual(recorder['capture_errors'], [])
        self.assertFalse(hasattr(trace.traced_class(FakePipeline), '_adapt_joint_result'))

    async def test_same_calls_arguments_response_and_no_extra_candidate_call(self):
        plain, observed = FakePipeline(), trace.traced_class(FakePipeline)()
        body = trace.tutor_body(item())
        expected = await plain.tutor(body)
        recorder = {'events': [], 'capture_errors': [], 'state': None}
        token = trace._ACTIVE.set(recorder)
        try:
            actual = await observed.tutor(body)
        finally:
            trace._ACTIVE.reset(token)
        self.assertEqual(observed.calls, plain.calls)
        self.assertEqual(actual.to_tutor_response(), expected.to_tutor_response())
        self.assertEqual(observed.calls.count(('candidates',)), 1)
        self.assertIs(recorder['state'], actual)
        self.assertEqual(recorder['capture_errors'], [])
        events = recorder['events']
        self.assertEqual(next(e for e in events if e['hook']=='_ask' and e['phase']=='return')['result'], plain.quote)
        self.assertEqual(next(e for e in events if e['hook']=='_resolve_query' and e['phase']=='return')['query'], 'public1 resolved')
        actual.chunks[0].text = 'mutated later'
        self.assertIn('original material', next(e for e in events if e['hook']=='_search' and e['phase']=='return')['chunks'][0]['text'])
        self.assertEqual(FakePipeline.request_slots, 3)
        self.assertEqual(observed.request_slots, 64)

    async def test_return_identity_and_request_copy_super_dispatch(self):
        pipeline = trace.traced_class(FakePipeline)()
        result = await copy(pipeline)._ask(State(messages=[]), 'material', 's', 'u', 'low', 1)
        self.assertIs(result, pipeline.quote)

    async def test_real_source_hint_copy_mro_preserves_llm_call(self):
        from pipelines.kimjaehak.round6_query_hint import SourceHint
        calls = []
        for cls in (SourceHint, trace.traced_class(SourceHint)):
            p = cls.__new__(cls)
            p.llm = SimpleNamespace(chat=AsyncMock(return_value='{"section":"S"}'))
            p.searcher = SimpleNamespace(section_text=lambda sid: 'original material')
            state = p._new_state({'messages':[{'role':'user','content':'original question'}]}, False)
            state._resolved_source_hint = 'non-authoritative retrieval question'
            raw = await p._source_response(state, ['S'], 3.5)
            self.assertEqual(raw, '{"section":"S"}')
            p.llm.chat.assert_awaited_once()
            calls.append(p.llm.chat.await_args)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[0].kwargs,
                         {'effort':'low','max_tokens':2000,'timeout':3.5,'json_mode':True})

    async def test_real_guard_generation_bypass_stays_unchanged(self):
        from pipelines.kimjaehak.focused_excerpt import Selected
        from pipelines.kimjaehak.guard_fitted import ContextFast
        cls = trace.traced_class(Selected)
        p = cls.__new__(cls)
        p.llm = SimpleNamespace(chat=AsyncMock())
        state = p._new_state({'messages':[{'role':'user','content':'question'}]}, True)
        state.label, state.section = Label.NORMAL, 'S'
        with patch.object(ContextFast, '_generate', new_callable=AsyncMock) as parent:
            await p._generate(state)
            parent.assert_awaited_once_with(state)
        p.llm.chat.assert_not_called()

    async def test_each_async_hook_propagates_identical_exception_and_cancel(self):
        invocations = {
            '_ask': (State(messages=[]), 'material', 's', 'u', 'low', 1),
            '_search': (State(messages=[]), 'extra'), '_resolve_query': (State(messages=[]),),
            '_source_response': (State(messages=[]), ['S'], 8),
        }
        for hook, args in invocations.items():
            for exc in (RuntimeError('original'), TimeoutError('original'), asyncio.CancelledError()):
                calls = []
                async def fail(self, *args, **kwargs):
                    calls.append((args, kwargs))
                    raise exc
                base = type('Failure', (), {hook: fail})
                pipeline = trace.traced_class(base)()
                with self.assertRaises(type(exc)) as raised:
                    await getattr(pipeline, hook)(*args)
                self.assertIs(raised.exception, exc)
                self.assertEqual(len(calls), 1)

    async def test_external_cancel_reaches_pending_original(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        class Pending:
            async def _resolve_query(self, state):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
        task = asyncio.create_task(trace.traced_class(Pending)()._resolve_query(State(messages=[])))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancelled.is_set())

    def test_optional_hook_absence_and_sync_exception_preserved(self):
        self.assertFalse(hasattr(trace.traced_class(object), '_resolve_query'))
        exc = ValueError('original')
        class Failure:
            def _source_payload(self, state, ids):
                raise exc
        with self.assertRaises(ValueError) as raised:
            trace.traced_class(Failure)()._source_payload(State(messages=[]), [])
        self.assertIs(raised.exception, exc)

    async def test_parallel_requests_no_gold_in_body_and_separate_events(self):
        pipeline = trace.traced_class(FakePipeline)()
        with tempfile.TemporaryDirectory() as tmp:
            rows = await trace.collect([item('publicA'), item('publicB')], pipeline, 2, Path(tmp))
            saved = [json.loads(s) for s in (Path(tmp)/'observations.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        for row in saved:
            self.assertEqual(row['final']['label'], Label.NORMAL.value)
            self.assertEqual(row['final']['section'], 'S')
            self.assertEqual(row['final']['errors'], [])
            queries = [e['query'] for e in row['events'] if e['hook']=='_resolve_query' and e['phase']=='return']
            self.assertEqual(queries, [row['id']+' resolved'])
            self.assertNotIn('GOLD_SENTINEL', json.dumps(row))
        bodies = [c[1] for c in pipeline.calls if c[0]=='new']
        for body in bodies:
            self.assertEqual(set(body), {'messages', 'session_id', 'spec_version'})
            self.assertTrue(body['session_id'].startswith('trace_'))
            self.assertNotIn('GOLD', json.dumps(body))
        self.assertNotEqual(bodies[0]['session_id'], bodies[1]['session_id'])
        self.assertIsNone(trace._ACTIVE.get())

    async def test_collection_failure_records_partial_and_propagates(self):
        exc = RuntimeError('original')
        class Failed(FakePipeline):
            async def _search(self, state, extra=''):
                state.errors.append('original:error')
                raise exc
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as raised:
                await trace.collect([item()], trace.traced_class(Failed)(), 1, Path(tmp))
            self.assertIs(raised.exception, exc)
            row = json.loads((Path(tmp)/'observations.jsonl').read_text())
            self.assertEqual(row['exception'], 'RuntimeError')
            self.assertEqual(row['final']['errors'], ['original:error'])
            with self.assertRaises(FileExistsError):
                await trace.collect([item()], FakePipeline(), 1, Path(tmp))
        self.assertIsNone(trace._ACTIVE.get())


class PublicSourceTests(unittest.TestCase):
    def fixture(self, root):
        parent = root/'experiments/.cache/derived'
        run = parent/'variant_1'
        run.mkdir(parents=True)
        dataset = (json.dumps(item())+'\n').encode()
        (parent/'derived_inputs.jsonl').write_bytes(dataset)
        (parent/'derivation.json').write_text(json.dumps({'dataset_sha256': hashlib.sha256(dataset).hexdigest()}))
        (run/'manifest.json').write_text(json.dumps({'scope': 'provided_kb_fact_diagnostic', 'model':'test',
            'evaluator_hashes':{}, 'count':1, 'variant':'fake'}))
        (run/'responses.jsonl').write_text(json.dumps({'id':'public1', 'response':{}, 'error':None})+'\n')
        (run/'derived_result.json').write_text(json.dumps({'items':[{'id':'public1'}], 'tutor_judge_skipped':False}))
        return run

    def test_real_source_data_hash_model_completion_and_confinement(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENAI_MODEL':'test'}), \
                patch('jh_tutor_generation_probe.evaluator_hashes', return_value={}):
            root = Path(tmp).resolve()
            run = self.fixture(root)
            with patch.object(trace, 'ROOT', root):
                _, _, items, hashes = trace.checked_source(run)
                self.assertEqual(len(items), 1)
                self.assertIn('experiments/.cache/derived/derived_inputs.jsonl', hashes)
                with self.assertRaises(ValueError): trace.checked_source(root)
                with patch.dict(os.environ, {'OPENAI_MODEL':'different'}):
                    with self.assertRaises(ValueError): trace.checked_source(run)
                data = run.parent/'derived_inputs.jsonl'
                data.write_text(data.read_text()+'\n')
                with self.assertRaises(ValueError): trace.checked_source(run)

    def test_incomplete_and_escaping_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'OPENAI_MODEL':'test'}), \
                patch('jh_tutor_generation_probe.evaluator_hashes', return_value={}):
            root = Path(tmp).resolve(); run = self.fixture(root)
            with patch.object(trace, 'ROOT', root):
                (run/'derived_result.json').write_text('{"items":[]}')
                with self.assertRaises(ValueError): trace.checked_source(run)
                (run/'derived_result.json').unlink()
                outside=root/'outside.json';outside.write_text('{}')
                (run/'derived_result.json').symlink_to(outside)
                with self.assertRaises(ValueError): trace.checked_source(run)

    def test_selection_and_body_drop_gold_and_message_extras(self):
        x=item();x['messages'][0]['labels']='HIDDEN'
        body=trace.tutor_body(x)
        self.assertEqual(body['messages'],[{'role':'user','content':'public1'}])
        self.assertEqual(trace.select_items([x], ['public1']), [x])
        for ids in [['unknown'], ['public1','public1']]:
            with self.assertRaises(ValueError):trace.select_items([x], ids)

    def test_initialize_loads_existing_index_without_embedding_call(self):
        fake_llm=Mock(model='provided',embedding_model='embed',can_embed=True)
        base=Mock()
        traced=Mock(return_value='pipeline')
        with patch.object(trace,'load',return_value=base) as load, \
                patch.object(trace,'traced_class',return_value=traced), \
                patch.object(trace,'LLM',return_value=fake_llm), \
                patch.object(trace,'load_chunks',return_value=[]), \
                patch.object(trace,'load_vectors',return_value=None) as vectors, \
                patch.dict(os.environ, {'KB_INDEX':'/tmp/nonexistent-public-trace-index.json'}):
            p, meta=trace.initialize('test')
        self.assertEqual(p,'pipeline')
        load.assert_called_once_with('test')
        vectors.assert_called_once()
        fake_llm.embed.assert_not_called()
        fake_llm.chat.assert_not_called()
        self.assertFalse(meta['embedding_index_loaded'])

    def test_main_loads_dotenv_before_source_validation_no_api(self):
        marker=[]
        def env(*args,**kwargs):marker.append('env')
        def source(*args):
            self.assertEqual(marker,['env'])
            raise ValueError('stop before initialization')
        with patch('dotenv.load_dotenv',side_effect=env), \
                patch.object(trace,'checked_source',side_effect=source), \
                patch.object(trace,'initialize') as init:
            with self.assertRaises(ValueError):
                trace.main(['--source','missing','--variant','jh_r3_selected_focused','--name','offline'])
            init.assert_not_called()


if __name__ == '__main__':
    unittest.main()
