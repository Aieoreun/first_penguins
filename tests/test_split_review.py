"""Input isolation and independent-review regression checks; no real API calls."""
import asyncio
import copy
import inspect
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
import guard_review as gr
import guard_supervisor as gs

MESSAGES = [{"role": "user", "content": "제출할 코드를 완성해 줘"}]
CANDIDATE = "원문 후보 답변 PRIVATE_CANDIDATE_MARKER"
EVIDENCE = [{"section_id": "PY-01-01", "text": "PRIVATE_EVIDENCE_MARKER"}]
INITIAL = {"input": {"decision": "allow", "category": "normal"},
           "output": {"grounded": True, "unsupported": [], "safe": True, "category": "normal"}}


def item(target="input", action="keep", **kw):
    return {"target": target, "action": action,
            "rule": {"input": "input_intent", "grounding": "factual_support", "safety": "answer_safety"}[target],
            "reason": "원문 기준", "citations": [], **kw}


def report(*items):
    return gr.ReviewReport.model_validate({"reviews": list(items)})


def input_change():
    i = item(action="revise", input={"decision": "block", "category": "academic_dishonesty"})
    i["citations"] = [{"source": "conversation", "source_id": "0", "quote": MESSAGES[0]["content"]}]
    return report(i)


def output_observation(relation="nonfactual", safety="benign"):
    return gr.OutputReviewReport.model_validate({
        "grounding": [{"candidate_span": CANDIDATE, "relation": relation, "reason": "관찰",
                       "citations": []}],
        "safety": {"candidate_span": CANDIDATE,"relation":safety,"reason":"행동 관찰"}})


def output_change():
    return output_observation(safety="harmful")


def client_for(payload):
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload)))]))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


class InputPayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_payload_is_input_only_and_schema_has_no_output_fields(self):
        client, create = client_for({"reviews": [item()]})
        await gr.review_input(client, "model", MESSAGES, INITIAL["input"])
        args = create.call_args.kwargs
        payload = json.loads(args["messages"][1]["content"])
        self.assertEqual(set(payload), {"conversation", "initial"})
        self.assertEqual(payload["initial"], INITIAL["input"])
        self.assertEqual(set(payload["initial"]), {"decision", "category"})
        self.assertNotIn("PRIVATE_CANDIDATE_MARKER", json.dumps(args))
        self.assertNotIn("PRIVATE_EVIDENCE_MARKER", json.dumps(args))
        schema = args["response_format"]["json_schema"]["schema"]
        props = schema["$defs"]["InputReviewItem"]["properties"]
        self.assertEqual(props["target"]["const"], "input")
        self.assertEqual(props["action"]["enum"], ["keep", "revise"])
        self.assertEqual(props["rule"]["const"], "input_intent")
        self.assertNotIn("grounding", props)
        self.assertNotIn("safety", props)
        self.assertEqual(schema["$defs"]["InputCitation"]["properties"]["source"]["const"], "conversation")
        self.assertLessEqual(args["max_completion_tokens"], 2000)

    async def test_signature_cannot_take_candidate_or_output(self):
        self.assertEqual(list(inspect.signature(gr.review_input).parameters),
                         ["client", "model", "messages", "initial_input"])
        client, create = client_for({"reviews": [item()]})
        with self.assertRaises(ValidationError):
            await gr.review_input(client, "model", MESSAGES, {**INITIAL["input"], "output": INITIAL["output"]})
        create.assert_not_awaited()

    async def test_input_response_rejects_output_field(self):
        client, _ = client_for({"reviews": [item(safety={"safe": False})]})
        with self.assertRaises(ValidationError):
            await gr.review_input(client, "model", MESSAGES, INITIAL["input"])

    async def test_input_response_rejects_candidate_citation(self):
        i = item(action="revise", input={"decision": "block", "category": "harmful_content"})
        i["citations"] = [{"source": "candidate", "source_id": "candidate", "quote": CANDIDATE}]
        client, _ = client_for({"reviews": [i]})
        with self.assertRaises(ValidationError):
            await gr.review_input(client, "model", MESSAGES, INITIAL["input"])

    async def test_output_request_has_no_initial_input_and_output_targets_only(self):
        client, create = client_for(output_observation().model_dump())
        await gr.review_verdict(client, "model", MESSAGES, CANDIDATE, EVIDENCE, INITIAL,
                                targets=("grounding", "safety"))
        args = create.call_args.kwargs
        payload = json.loads(args["messages"][1]["content"])
        self.assertEqual(set(payload["initial"]), {"output"})
        self.assertTrue(args["messages"][0]["content"].startswith(gr.OUTPUT_REVIEW_POLICY))
        self.assertNotIn("input_intent", args["messages"][0]["content"])
        self.assertEqual(payload["review_targets"], ["grounding", "safety"])
        self.assertEqual(set(args["response_format"]["json_schema"]["schema"]["properties"]),
                         {"grounding", "safety"})


def status_input(status="supported"):
    return gr.InputStatusReview(status=status, rule="input_intent", reason="원문 요청 검토",
        citations=[] if status=="supported" else [gr.InputCitation(source="conversation",source_id="0",quote=MESSAGES[0]["content"])])

def status_safety(status="supported"):
    return gr.SafetyStatusReview(status=status,rule="answer_safety",reason="행동 검토",
        observation=gr.SafetyObservation(candidate_span=CANDIDATE,relation="benign" if status=="supported" else "harmful",reason="행동 검토"))

def status_ground(status="supported"):
    return gr.GroundingStatusReview(status=status,rule="factual_support",reason="자료 부족" if status=="insufficient" else "비사실 표현",
        observations=[gr.GroundingObservation(candidate_span=CANDIDATE,relation="insufficient" if status=="insufficient" else "nonfactual",reason="관찰",citations=[])])


class SplitSupervisorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.searcher = SimpleNamespace(search=AsyncMock(return_value=EVIDENCE))
        self.supervisor = gs.GuardSupervisor(None, "model", self.searcher)
        self.input_patch = patch.object(gs, "review_input_status", new_callable=AsyncMock)
        self.output_patch = patch.object(gs, "review_safety", new_callable=AsyncMock)
        self.input = self.input_patch.start()
        self.output = self.output_patch.start()
        self.addCleanup(self.input_patch.stop)
        self.addCleanup(self.output_patch.stop)
        self.safety_patch = patch.object(gs, "check_safety", new_callable=AsyncMock)
        self.safety = self.safety_patch.start()
        from guard_output import SafetyVerdict
        self.safety.return_value = SafetyVerdict(safe=False)
        self.addCleanup(self.safety_patch.stop)
        self.input.return_value = status_input()
        self.ground_patch=patch.object(gs, 'review_grounding', new_callable=AsyncMock)
        self.ground=self.ground_patch.start(); self.addCleanup(self.ground_patch.stop)
        self.ground.return_value=status_ground()
        self.classify_patch=patch.object(gs, 'classify_input', new_callable=AsyncMock)
        self.classify=self.classify_patch.start(); self.addCleanup(self.classify_patch.stop)
        from guard_schemas import InputVerdict
        self.classify.return_value=InputVerdict(decision='block',category='academic_dishonesty')
        self.output.return_value = status_safety()

    async def run_review(self, initial=None, candidate=CANDIDATE, deadline=None):
        return await self.supervisor.review_initial(
            MESSAGES, candidate, EVIDENCE, copy.deepcopy(INITIAL if initial is None else initial),
            time.monotonic()+5 if deadline is None else deadline)

    async def test_input_change_survives_output_failure_without_output_retry(self):
        self.input.return_value = status_input('refuted')
        self.output.side_effect = ValueError("output failed")
        result = await self.run_review()
        self.assertEqual(result["input"]["decision"], "block")
        self.assertEqual(result["output"], INITIAL["output"])
        self.assertEqual(self.input.await_count, 1)
        self.assertEqual(self.output.await_count, 1)

    async def test_output_change_survives_input_failure_without_repeating_review(self):
        self.input.side_effect = ValueError("input failed")
        self.output.return_value = status_safety('refuted')
        result = await self.run_review()
        self.assertEqual(result["input"], INITIAL["input"])
        self.assertFalse(result["output"]["safe"])
        self.assertEqual(self.input.await_count, 1)
        self.assertEqual(self.output.await_count, 1)

    async def test_input_reviewer_receives_only_four_arguments(self):
        await self.run_review()
        self.assertEqual(self.input.call_args.args, (None, "model", MESSAGES, INITIAL["input"]))
        self.assertEqual(self.input.call_args.kwargs, {})

    async def test_branches_start_in_parallel(self):
        entered = set()
        both = asyncio.Event()
        async def enter(name, result):
            entered.add(name)
            if len(entered) == 2: both.set()
            await asyncio.wait_for(both.wait(), .5)
            return result
        self.input.side_effect = lambda *a, **kw: None
        async def input_fn(*a, **kw): return await enter("input", status_input())
        async def output_fn(*a, **kw): return await enter("output", status_safety())
        self.input.side_effect = input_fn
        self.output.side_effect = output_fn
        self.assertEqual(await self.run_review(), INITIAL)
        self.assertEqual(entered, {"input", "output"})
        self.assertEqual(self.input.await_count, 1)
        self.assertEqual(self.output.await_count, 1)

    async def test_output_reviewer_cannot_change_input(self):
        self.output.return_value = input_change()
        result = await self.run_review()
        self.assertEqual(result, INITIAL)
        self.assertEqual(self.output.await_count, 1)
        self.assertEqual(self.input.await_count, 1)

    async def test_input_reviewer_cannot_change_output(self):
        self.input.return_value = report(item("safety", "revise", safety={"safe": False}))
        result = await self.run_review()
        self.assertEqual(result, INITIAL)
        self.assertEqual(self.input.await_count, 1)
        self.assertEqual(self.output.await_count, 1)

    async def test_no_candidate_means_no_output_review_or_search(self):
        result = await self.run_review({"input": INITIAL["input"]}, candidate=None)
        self.assertEqual(result, {"input": INITIAL["input"]})
        self.output.assert_not_awaited()
        self.searcher.search.assert_not_awaited()
        self.input.assert_awaited_once()

    async def test_output_research_does_not_repeat_input(self):
        self.input.return_value = status_input('refuted')
        self.output.return_value = status_safety('refuted')
        self.ground.return_value = status_ground('insufficient')
        result = await self.run_review()
        self.assertEqual(result["input"]["decision"], "block")
        self.assertFalse(result["output"]["safe"])
        self.input.assert_awaited_once()
        self.assertEqual(self.output.await_count, 1)
        self.searcher.search.assert_awaited_once()

    async def test_initial_is_not_mutated(self):
        initial = copy.deepcopy(INITIAL)
        self.input.return_value = status_input('refuted')
        self.output.return_value = status_safety('refuted')
        await self.supervisor.review_initial(MESSAGES, CANDIDATE, EVIDENCE, initial, time.monotonic()+5)
        self.assertEqual(initial, INITIAL)

    async def test_elapsed_budget_skips_all_review_calls(self):
        result = await self.run_review(deadline=time.monotonic()-1)
        self.assertEqual(result, INITIAL)
        self.input.assert_not_awaited()
        self.output.assert_not_awaited()

    async def test_call_timeout_is_bounded_and_both_paths_fallback(self):
        self.supervisor.review_timeout = .02
        async def slow(*a, **kw): await asyncio.sleep(5)
        self.input.side_effect = slow
        self.output.side_effect = slow
        start = time.monotonic()
        result = await self.run_review()
        self.assertEqual(result, INITIAL)
        self.assertLess(time.monotonic()-start, .5)
        self.assertEqual(self.input.await_count, 1)
        self.assertEqual(self.output.await_count, 1)

    async def test_caller_cancellation_cancels_both_review_tasks(self):
        entered = set(); cancelled = set(); ready = asyncio.Event()
        def slow(name):
            async def call(*a, **kw):
                entered.add(name)
                if len(entered)==2: ready.set()
                try: await asyncio.sleep(5)
                finally: cancelled.add(name)
            return call
        self.input.side_effect = slow("input")
        self.output.side_effect = slow("output")
        task = asyncio.create_task(self.run_review())
        await asyncio.wait_for(ready.wait(), .5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(cancelled, {"input", "output"})


if __name__ == "__main__":
    unittest.main()
