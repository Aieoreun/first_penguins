import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from core.state import Chunk, State
from pipelines.kimjaehak.guard_fitted import ContextFast
from pipelines.kimjaehak.joint_input import Pipeline, BlockOnly


class JointInputTests(unittest.TestCase):
    def state(self):
        return State(messages=[{"role":"user","content":"교재 질문"}],chunks=[Chunk("A","교재에 적힌 사실")])

    def test_valid_joint_decision_cancels_backup_without_copying_its_errors(self):
        pipe = Pipeline.__new__(Pipeline)
        pipe.llm = type("LLM",(),{"chat":AsyncMock(return_value=json.dumps({"intent":"학습 설명", "block":"none",
                    "other":"none","self_info":False,"section":"A","quote":"교재에 적힌 사실"}))})()
        state = self.state()
        async def backup(backup_state,check,material):
            backup_state.errors.append("discarded:TimeoutError")
            await asyncio.sleep(20)
        with patch.object(ContextFast,"_collect_checks",side_effect=backup,new_callable=AsyncMock):
            result = asyncio.run(pipe._collect_checks(state,"check","material"))
        self.assertEqual(result[0],[{"category":"none"}])
        self.assertEqual(result[2]["quote"],"교재에 적힌 사실")
        self.assertEqual(state.errors,[])

    def test_invalid_source_quote_recovers_existing_decisions(self):
        pipe = Pipeline.__new__(Pipeline)
        pipe.llm = type("LLM",(),{"chat":AsyncMock(return_value=json.dumps({"intent":"학습 설명", "block":"none",
                    "other":"none","self_info":False,"section":"invented","quote":"없는 근거"}))})()
        state = self.state()
        expected = ([{"category":"off_topic"}],[{"category":"none","self_info":False}],{"quote":""})
        async def backup(backup_state,check,material):
            backup_state.errors.append("material:parse")
            return expected
        with patch.object(ContextFast,"_collect_checks",side_effect=backup,new_callable=AsyncMock):
            actual = asyncio.run(pipe._collect_checks(state,"check","material"))
        self.assertEqual(actual,expected)
        self.assertEqual(state.errors,["joint_input:ValueError","material:parse"])

    def test_transport_failure_uses_backup_and_keeps_request_independent(self):
        pipe = Pipeline.__new__(Pipeline)
        pipe.llm = type("LLM",(),{"chat":AsyncMock(side_effect=TimeoutError)})()
        expected = ([{"category":"none"}],[],{})
        with patch.object(ContextFast,"_collect_checks",return_value=expected,new_callable=AsyncMock) as backup:
            states = [self.state(),self.state()]
            for state in states:
                self.assertEqual(asyncio.run(pipe._collect_checks(state,"check","material")),expected)
            self.assertEqual(backup.await_count,2)
        self.assertIsNot(states[0].errors,states[1].errors)
        self.assertEqual(states[0].errors,["joint_input:TimeoutError"])

    def test_block_only_keeps_independent_other_and_material_decisions(self):
        pipe = BlockOnly.__new__(BlockOnly)
        pipe.llm = type("LLM",(),{"chat":AsyncMock(return_value=json.dumps({"intent":"학습 설명", "block":"none",
                    "other":"answer_seeking","self_info":True,"section":"invented","quote":"없는 근거"}))})()
        state = self.state()
        old_other,old_material = [{"category":"none","self_info":False}],{"quote":"교재에 적힌 사실","section":"A"}
        with patch.object(ContextFast,"_collect_checks",return_value=([{"category":"jailbreak_injection"}],old_other,old_material),new_callable=AsyncMock):
            result = asyncio.run(pipe._collect_checks(state,"check","material"))
        self.assertEqual(result,([{"category":"none"}],old_other,old_material))
        self.assertEqual(state.errors,[])


if __name__ == "__main__":
    unittest.main()
