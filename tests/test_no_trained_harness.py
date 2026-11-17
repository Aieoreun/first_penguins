"""The selected port skips learned layers without replacing its LLM fallbacks."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from core.schema import Label
from core.state import Chunk
from pipelines.kimjaehak.material_budget import Pipeline
from pipelines.kimjaehak.focused_excerpt import Selected
from pipelines.kimjaehak.learned_gate import Pipeline as LearnedGate

TRAINERS = (
    'pipelines.kimjaehak.learned_gate.train_public_classifier',
    'pipelines.kimjaehak.guard_fitted.train_public_classifier',
    'pipelines.kimjaehak.learned_sections.train_section_classifier',
    'pipelines.kimjaehak.learned_harmful.train_harmful_classifier',
    'pipelines.kimjaehak.learned_output.SpanTagger',
)


def make_pipeline():
    searcher = SimpleNamespace(section_text=lambda sid: 'source text')
    with ExitStack() as stack:
        fits = [stack.enter_context(patch(name, side_effect=AssertionError('training forbidden')))
                for name in TRAINERS]
        pipe = Pipeline(searcher, SimpleNamespace())
    for fit in fits:
        fit.assert_not_called()
    return pipe


class NoTrainedHarnessTests(unittest.IsolatedAsyncioTestCase):
    def test_selected_variant_only_disables_training(self):
        self.assertFalse(Pipeline.learned_models_enabled)
        self.assertTrue(Selected.learned_models_enabled)
        self.assertTrue(LearnedGate.learned_models_enabled)
        pipe = make_pipeline()
        for attr in ('classifier', 'vectorizer', 'guard_classifier', 'section_classifier',
                     'harmful_classifier', 'span_tagger'):
            self.assertFalse(hasattr(pipe, attr), attr)
        self.assertEqual(pipe.deadline, 26)
        self.assertEqual(pipe.output_votes, 5)
        self.assertEqual(pipe.request_slots, 32)

    async def test_both_endpoints_fall_through_to_original_adaptive_llm_roles(self):
        for guard in (False, True):
            pipe = make_pipeline()
            state = pipe._new_state({'messages': [{'role': 'user', 'content': 'Explain the concept'}]}, guard)
            state.chunks = [Chunk('S', 'source text')]
            calls = []
            async def ask(state, name, system, user, effort, timeout):
                calls.append(name)
                return {'block': {'category': 'none'},
                        'other': {'category': 'none', 'self_info': True},
                        'material': {'quote': 'source text', 'section': 'S'}}[name]
            pipe._ask = ask
            await pipe._classify(state)
            self.assertEqual(state.label, Label.PII_DISCLOSURE)
            self.assertEqual(state.section, 'S')
            self.assertEqual(calls.count('block'), 2)
            self.assertEqual(calls.count('other'), 2)
            self.assertEqual(calls.count('material'), 1)
            self.assertFalse(any(k.startswith('learned') for k in state.debug))

    async def test_disagreement_still_calls_third_vote_and_preserves_priority(self):
        pipe = make_pipeline()
        state = pipe._new_state({'messages': [{'role': 'user', 'content': 'Explain the concept'}]}, True)
        state.chunks = [Chunk('S', 'source text')]
        block = iter(['none', 'harmful_request', 'harmful_request'])
        calls = []
        async def ask(state, name, system, user, effort, timeout):
            calls.append(name)
            if name == 'block': return {'category': next(block)}
            if name == 'other': return {'category': 'emotional_distress', 'self_info': True}
            return {'quote': '', 'section': 'S'}
        pipe._ask = ask
        await pipe._classify(state)
        self.assertEqual(state.label, Label.HARMFUL_REQUEST)
        self.assertEqual(calls.count('block'), 3)
        self.assertEqual(calls.count('other'), 2)
        self.assertEqual(calls.count('material'), 1)

    async def test_original_all_failed_fallback_is_preserved_not_replaced(self):
        pipe = make_pipeline()
        state = pipe._new_state({'messages': [{'role': 'user', 'content': 'Explain the concept'}]}, True)
        state.chunks = [Chunk('S', 'source text')]
        pipe._ask = AsyncMock(return_value={})
        await pipe._classify(state)
        self.assertEqual(state.label, Label.NORMAL)
        self.assertEqual(pipe._ask.await_count, 7)

    async def test_output_consensus_runs_without_trained_span_refinement(self):
        pipe = make_pipeline()
        state = pipe._new_state({'messages': [{'role': 'user', 'content': 'Explain the concept'}],
                                 'candidate_answer': 'unsupported claim'}, True)
        state.label = Label.NORMAL
        state.chunks = [Chunk('S', 'source text')]
        state.section = 'S'
        pipe._ground_response = AsyncMock(return_value=json.dumps({'unsupported': ['unsupported claim']}))
        await pipe._check_output(state)
        self.assertEqual(pipe._ground_response.await_count, 5)
        self.assertFalse(state.grounded)
        self.assertEqual(state.unsupported, ['unsupported claim'])
        self.assertNotIn('learned_span', state.debug)

    def test_import_and_initialization_do_not_import_training_dependencies(self):
        script = '''
import importlib.abc, sys
class DenyTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'threadpoolctl'}:
            raise AssertionError('training dependency imported: ' + fullname)
sys.meta_path.insert(0, DenyTrainingImports())
from pipelines.kimjaehak.material_budget import Pipeline
from types import SimpleNamespace
p = Pipeline(SimpleNamespace(), SimpleNamespace())
assert not hasattr(p, 'classifier')
assert not hasattr(p, 'span_tagger')
'''
        result = subprocess.run([sys.executable, '-c', script],
                                cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
