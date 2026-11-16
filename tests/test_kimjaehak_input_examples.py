import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from core.state import State
from pipelines.kimjaehak.fewshot import normalized
from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.input_examples import PolicyExamples, Lexical, Dense
from tests.test_pipeline_v2 import SEARCHER


class InputExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = PolicyExamples(True)

    def test_exclusion_and_stage_labels_for_both_retrievers(self):
        examples = self.examples
        index = next(i for i,r in enumerate(examples.rows) if r["labels"]["category"]=="harmful_content")
        query = examples.texts[index]
        messages = [{"role":"user","content":query+"!!"}]
        before = json.dumps(examples.rows,sort_keys=True)
        for vector in (None, examples.vectors[index].tolist()):
            for kind in ("block","other"):
                selected = examples.select(messages,kind,vector)
                self.assertTrue(selected)
                self.assertTrue(all(normalized(r["last_user"]) != normalized(query) for r in selected))
                allowed = ({"none","harmful_request","harmful_speech","jailbreak_injection","academic_dishonesty","off_topic"}
                           if kind == "block" else {"none","emotional_distress","answer_seeking","low_signal_input"})
                self.assertTrue(all(r["decision"]["category"] in allowed for r in selected))
        self.assertEqual(json.dumps(examples.rows,sort_keys=True),before)

    def test_prompt_examples_are_cached_only_in_request_state(self):
        pipe = Lexical.__new__(Lexical)
        pipe.policy_examples = self.examples
        state = State(messages=[{"role":"user","content":"파이썬 코드를 설명해 주세요"}])
        with patch.object(self.examples,"select",wraps=self.examples.select) as select, \
             patch.object(ContextFast,"_ask",new_callable=AsyncMock) as inherited:
            for _ in range(2):
                asyncio.run(pipe._ask(state,"block","sys","user","low",10))
            self.assertEqual(select.call_count,1)
            self.assertEqual(inherited.await_count,2)
            self.assertNotIn("policy_prompt_examples",pipe.__dict__)
            other = State(messages=state.messages)
            asyncio.run(pipe._ask(other,"block","sys","user","low",10))
            self.assertEqual(select.call_count,2)

    def test_embedding_failure_leaves_lexical_retrieval_usable(self):
        pipe = Dense.__new__(Dense)
        pipe.searcher,pipe.policy_examples = SEARCHER,self.examples
        pipe.llm = type("LLM",(),{"can_embed":True,"embedding_model":"text-embedding-3-small",
                                "embed":AsyncMock(side_effect=TimeoutError)})()
        state = State(messages=[{"role":"user","content":"파이썬 문자열"}])
        asyncio.run(pipe._search(state))
        self.assertTrue(state.chunks)
        self.assertFalse(hasattr(state,"policy_example_vector"))
        self.assertEqual(state.errors,["embed:TimeoutError"])


if __name__ == "__main__":
    unittest.main()
