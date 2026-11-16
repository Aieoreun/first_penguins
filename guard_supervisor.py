"""Bounded per-request orchestration. Never store request/response contents."""
import asyncio
import copy
import logging
import time
from collections import Counter

from openai import APIConnectionError, APIStatusError

from guard_classifier import classify_input
from guard_output import check_grounding, check_safety
from guard_review import (OutputReviewReport, review_input, review_verdict,
                          review_input_status, review_grounding, review_safety)
from guard_schemas import validate_guard_response
from guard_runtime import request_scope, deadline_scope, record, timed_stage

logger = logging.getLogger(__name__)


def validated(payload, candidate):
    result = validate_guard_response(payload, candidate).model_dump(exclude_none=True)
    output = result.get("output")
    if candidate is None and output is not None:
        raise ValueError("unexpected_output")
    if output is not None and output["grounded"] != (not output["unsupported"]):
        raise ValueError("inconsistent_grounding")
    return result


def retryable(error):
    if isinstance(error, APIStatusError):
        # Quota exhaustion cannot be solved by repeating the request.
        if error.status_code == 429:
            body = getattr(error, "body", {}) or {}
            return "quota" not in str(body).lower() and "credit" not in str(body).lower()
        return error.status_code in (408, 409) or error.status_code >= 500
    return isinstance(error, (APIConnectionError, TimeoutError, asyncio.TimeoutError, ValueError))


def apply_review(initial, report, messages, candidate, evidence, *, observer=None):
    """Check quote provenance, target scope, and public invariants, not semantic truth."""
    def emit(action, item, **data):
        if observer is not None:
            observer("supervisor_action", target=item.target, action=action, **data)
    result = copy.deepcopy(initial)
    sources = {("conversation", str(i)): m["content"] for i, m in enumerate(messages)}
    if candidate is not None:
        sources[("candidate", "candidate")] = candidate
    sources.update({("kb", e["section_id"]): e["text"] for e in evidence})
    for item in report.reviews:
        if item.action != "revise":
            emit("keep", item, reason="review_requested_no_revision")
            continue
        if not item.citations or any(
            not c.quote.strip() or c.quote not in sources.get((c.source, c.source_id), "")
            for c in item.citations
        ):
            emit("reject", item, reason="invalid_or_missing_citation")
            continue
        kinds = {c.source for c in item.citations}
        required = {"input": {"conversation"}, "safety": {"candidate"},
                    "grounding": {"candidate", "kb"}}[item.target]
        if item.target == "grounding" and item.rule == "nonfactual":
            if not item.grounding.grounded:
                continue
            required = {"candidate"}
        if not required <= kinds:
            emit("reject", item, reason="missing_required_source")
            continue
        proposal = copy.deepcopy(result)
        if item.target == "input":
            proposal["input"] = item.input.model_dump()
        elif "output" not in proposal:
            continue
        elif item.target == "grounding":
            proposal["output"].update(item.grounding.model_dump())
        else:
            proposal["output"]["safe"] = item.safety.safe
        try:
            updated = validated(proposal, candidate)
            emit("apply" if updated != result else "keep", item, reason="validated_proposal", before=result, after=updated)
            result = updated
        except ValueError:
            emit("reject", item, reason="invalid_proposal")
            continue
    return result


class GuardSupervisor:
    def __init__(self, client, model, searcher, review_mode="all", *, total_timeout=28.0,
                 base_timeout=20.0, review_timeout=12.0, observer=None, safety_review_enabled=True):
        if review_mode not in {"off", "all", "selective"}:
            raise ValueError("invalid_review_mode")
        if not 0 < base_timeout <= total_timeout < 30 or review_timeout <= 0:
            raise ValueError("invalid_time_budget")
        self.safety_review_enabled = safety_review_enabled
        self._observer = observer  # Only the public diagnostic launcher supplies this callback.
        self.metrics = Counter()  # Counts only; no cross-request content or verdict cache.
        self.client, self.model, self.searcher = client, model, searcher
        self.review_mode = review_mode
        self.total_timeout, self.base_timeout, self.review_timeout = total_timeout, base_timeout, review_timeout

    def _emit(self, event, **data):
        # Disabled in production. Monitoring failure must not change a verdict.
        if self._observer is not None:
            try:
                self._observer(event, **data)
            except Exception:
                self.metrics["monitor_errors"] += 1

    async def _stage(self, operation, deadline, name="base"):
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("base_deadline")
            started = time.monotonic()
            record("attempt", name, attempt=attempt + 1, remaining=round(remaining, 4))
            try:
                async with timed_stage(name):
                    return await asyncio.wait_for(operation(), timeout=remaining)
            except Exception as error:
                elapsed = time.monotonic() - started
                reason = getattr(error, "reason_code", None)
                record("attempt_failed", name, attempt=attempt + 1,
                       error_type=type(error).__name__, reason=reason)
                # The identical request cannot fix an exhausted token budget.
                exhausted = reason == "unexpected_finish:length"
                needed = max(2.0, elapsed * 1.1) + 0.25
                if attempt or not retryable(error) or exhausted or deadline - time.monotonic() < needed:
                    record("retry_skipped", name, reason="token_limit" if exhausted else
                           "attempt_limit" if attempt else "not_retryable" if not retryable(error)
                           else "insufficient_time", remaining=round(deadline-time.monotonic(), 4))
                    raise
                self.metrics["base_retries"] += 1
                record("retry", name, attempt=attempt + 2)
                await asyncio.sleep(0.1)

    def _needs_review(self, initial):
        if self.review_mode == "off":
            return False
        if self.review_mode == "all" or "output" in initial:
            return True
        # A heuristic only, not a calibrated confidence estimate.
        return initial["input"]["decision"] == "block" or initial["input"]["category"] != "normal"

    async def evaluate(self, messages, candidate_answer=None, *, request_id=None):
        started = time.monotonic()
        deadline = started + self.total_timeout
        base_deadline = min(deadline, started + self.base_timeout)
        with request_scope(base_deadline, request_id):
            record("start", "request")
            try:
                result = await self._evaluate(messages, candidate_answer, deadline, base_deadline)
            except BaseException as error:
                record("failure", "request", error_type=type(error).__name__,
                       seconds=round(time.monotonic()-started, 4))
                raise
            record("success", "request", seconds=round(time.monotonic()-started, 4))
            return result

    async def _evaluate(self, messages, candidate_answer, deadline, base_deadline):
        evidence = None
        completed = {}  # Request-local only; no cross-request cache or guessed fields.
        query = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        async def grounding():
            nonlocal evidence
            if evidence is None:
                async with timed_stage("grounding_search"):
                    evidence = await self.searcher.search(query, top_k=5)
            else:
                record("reuse", "grounding_search")
            return await check_grounding(self.client, self.model, messages, candidate_answer, evidence)

        operations = {"input": lambda: classify_input(self.client, self.model, messages, searcher=self.searcher)}
        if candidate_answer is not None:
            operations.update(grounding=grounding,
                              safety=lambda: check_safety(self.client, self.model, messages, candidate_answer))

        async def run(name, operation):
            completed[name] = await self._stage(operation, base_deadline, name)
            record("retained", name)
            return completed[name]

        tasks = [asyncio.create_task(run(name, op)) for name, op in operations.items()]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            record("incomplete", "request", completed=sorted(completed))
            raise errors[0]  # Never fabricate uncomputed required fields.
        initial = {"input": completed["input"].model_dump()}
        if candidate_answer is not None:
            initial["output"] = {**completed["grounding"].model_dump(),
                                 **completed["safety"].model_dump(), "category": "normal"}
        initial = validated(initial, candidate_answer)
        self._emit("initial_verdict", response=initial, evidence=evidence or [])
        if not self._needs_review(initial):
            self.metrics["review_skipped"] += 1
            return initial
        with deadline_scope(deadline):
            return await self.review_initial(messages, candidate_answer, evidence or [], initial, deadline)

    async def review_initial(self, messages, candidate_answer, evidence, initial, deadline):
        """Three independent scopes share one frozen first pass and absolute deadline."""
        initial = validated(initial, candidate_answer)
        scopes = ["input"] + (["grounding"] + (["safety"] if self.safety_review_enabled else []) if candidate_answer is not None else [])
        tasks = [asyncio.create_task(self._review_scope(
            target, messages, candidate_answer, evidence, copy.deepcopy(initial), deadline
        )) for target in scopes]
        try:
            patches = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        result = copy.deepcopy(initial)
        for target, patch in zip(scopes, patches):
            if patch is not None:
                if target == "input": result["input"] = patch
                else: result["output"].update(patch)
        self.metrics["changed_responses"] += int(result != initial)
        return validated(result, candidate_answer)

    async def _review_input(self, messages, initial_input, deadline):
        initial = {"input": initial_input}
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining < 1:
                break
            try:
                self.metrics["review_calls"] += 1
                self.metrics["input_review_calls"] += 1
                report = await asyncio.wait_for(
                    review_input(self.client, self.model, messages, initial_input),
                    timeout=min(self.review_timeout, remaining),
                )
                if {r.target for r in report.reviews} != {"input"}:
                    raise ValueError("wrong_input_review_scope")
                # Even a fabricated output revision has no candidate/output context to act on.
                result = apply_review(initial, report, messages, None, [], observer=self._emit)
                self.metrics["valid_input_reviews"] += 1
                return result["input"]
            except Exception as error:
                self._emit("supervisor_action", target="input", action="review_failure", reason=type(error).__name__, attempt=attempt + 1)
                self.metrics["input_review_errors"] += 1
                logger.warning("guard_input_review_failure type=%s attempt=%d", type(error).__name__, attempt + 1)
                if not retryable(error):
                    break
        self._emit("supervisor_action", target="input", action="fallback", reason="review_unavailable")
        self.metrics["input_review_fallbacks"] += 1
        return initial_input

    async def _review_output(self, messages, candidate_answer, evidence, initial, deadline):
        # Observations never enter apply_review. Only a validated adjudicator result may update output.
        original = copy.deepcopy(initial["output"])
        if deadline - time.monotonic() < 1:
            self._emit("supervisor_action", target="output", action="fallback", reason="deadline")
            self.metrics["output_review_fallbacks"] += 1
            return original
        try:
            self.metrics["review_calls"] += 1
            self.metrics["output_review_calls"] += 1
            report = await asyncio.wait_for(
                review_verdict(self.client, self.model, messages, candidate_answer, evidence, initial,
                               targets=("grounding", "safety")),
                timeout=min(self.review_timeout, deadline - time.monotonic()),
            )
            if not isinstance(report, OutputReviewReport):
                raise ValueError("output_observations_required")
            self.metrics["valid_output_reviews"] += 1
        except Exception as error:
            self._emit("supervisor_action", target="output", action="fallback", reason=type(error).__name__)
            self.metrics["output_review_errors"] += 1
            self.metrics["output_review_fallbacks"] += 1
            logger.warning("guard_output_review_failure type=%s", type(error).__name__)
            return original

        async def rerun(target, operation, expected):
            remaining = deadline - time.monotonic()
            if remaining < 1:
                self._emit("supervisor_action", target=target, action="fallback", reason="deadline")
                self.metrics[target + "_recheck_fallbacks"] += 1
                return None
            self._emit("supervisor_action", target=target, action="recheck", reason="conflict_or_new_evidence", expected=expected)
            self.metrics[target + "_recheck_calls"] += 1
            try:
                value = await asyncio.wait_for(operation(), timeout=min(self.review_timeout, remaining))
                patch = value.model_dump()
                self._emit("recheck_verdict", target=target, response=patch, expected=expected)
                field = "grounded" if target == "grounding" else "safe"
                if expected is not None and patch[field] != expected:
                    self._emit("supervisor_action", target=target, action="reject_recheck", reason="continued_disagreement", expected=expected, response=patch)
                    self.metrics[target + "_unresolved"] += 1
                    return None
                proposal = copy.deepcopy(initial)
                # A recheck may only supply its own fields.
                allowed = {"grounded", "unsupported"} if target == "grounding" else {"safe"}
                if set(patch) != allowed:
                    raise ValueError("wrong_recheck_scope")
                proposal["output"].update(patch)
                validated(proposal, candidate_answer)
                self._emit("supervisor_action", target=target, action="validated_recheck", response=patch)
                return patch
            except Exception as error:
                self._emit("supervisor_action", target=target, action="fallback", reason=type(error).__name__)
                self.metrics[target + "_recheck_fallbacks"] += 1
                logger.warning("guard_recheck_failure target=%s type=%s", target, type(error).__name__)
                return None

        async def grounding_recheck():
            sources = {("candidate", "candidate"): candidate_answer}
            sources.update({("kb", e["section_id"]): e["text"] for e in evidence})
            observations = report.grounding
            for obs in observations:
                if not obs.candidate_span.strip() or obs.candidate_span not in candidate_answer:
                    self._emit("supervisor_action", target="grounding", action="reject_observation", reason="candidate_span_not_in_original", observation=obs.model_dump())
                    self.metrics["invalid_grounding_observations"] += 1
                    return None
                if any(not c.quote.strip() or c.quote not in sources.get((c.source, c.source_id), "")
                       for c in obs.citations):
                    self._emit("supervisor_action", target="grounding", action="reject_observation", reason="citation_not_in_source", observation=obs.model_dump())
                    self.metrics["invalid_grounding_observations"] += 1
                    return None
                # candidate_span is already an exact candidate citation; do not require it twice.
                if obs.relation in {"supported", "contradicted"} and not obs.citations:
                    self._emit("supervisor_action", target="grounding", action="reject_observation", reason="missing_kb_citation", observation=obs.model_dump())
                    self.metrics["invalid_grounding_observations"] += 1
                    return None
            contradicted = [o.candidate_span for o in observations if o.relation == "contradicted"]
            insufficient = any(o.relation == "insufficient" for o in observations)
            clear = [o.candidate_span for o in observations if o.relation in {"supported", "nonfactual"}]
            expected = None
            if contradicted:
                expected = False
            elif original["unsupported"] and all(any(span in c for c in clear) for span in original["unsupported"]):
                expected = True
            if expected == original["grounded"]:
                self._emit("supervisor_action", target="grounding", action="keep", reason="observation_agrees", expected=expected)
                self.metrics["grounding_review_agreements"] += 1
                return None
            if expected is None and not insufficient:
                self._emit("supervisor_action", target="grounding", action="keep", reason="no_conflict_or_incomplete_clearance")
                return None
            updated_evidence = evidence
            if insufficient:
                if deadline - time.monotonic() < 4:
                    self._emit("supervisor_action", target="grounding", action="fallback", reason="research_budget")
                    self.metrics["grounding_recheck_fallbacks"] += 1
                    return None
                try:
                    self._emit("supervisor_action", target="grounding", action="research", reason="insufficient_evidence")
                    self.metrics["research_calls"] += 1
                    query = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                    extra = await asyncio.wait_for(
                        self.searcher.search(query + "\n검증 대상 주장:\n" + candidate_answer, top_k=8),
                        timeout=min(3, deadline - time.monotonic()),
                    )
                    # Empty/unchanged retrieval does not justify another model call.
                    known = {(e["section_id"], e["text"]) for e in evidence}
                    if not any((e["section_id"], e["text"]) not in known for e in extra):
                        self._emit("supervisor_action", target="grounding", action="keep", reason="no_new_evidence")
                        self.metrics["grounding_unresolved"] += 1
                        return None
                    updated_evidence = list(extra) + [e for e in evidence if e["section_id"] not in {x["section_id"] for x in extra}]
                except Exception as error:
                    self._emit("supervisor_action", target="grounding", action="fallback", reason=type(error).__name__)
                    self.metrics["grounding_recheck_fallbacks"] += 1
                    logger.warning("guard_research_failure type=%s", type(error).__name__)
                    return None
            patch = await rerun("grounding", lambda: check_grounding(
                self.client, self.model, messages, candidate_answer, updated_evidence,
                review_feedback={"observations": [o.model_dump() for o in observations]},
            ), expected)
            if patch and contradicted and not all(any(span in u for u in patch["unsupported"]) for span in contradicted):
                self._emit("supervisor_action", target="grounding", action="reject_recheck", reason="contradicted_span_not_covered", response=patch)
                self.metrics["grounding_unresolved"] += 1
                return None
            if patch:
                self._emit("supervisor_action", target="grounding", action="apply", before=original, after=patch)
            return patch

        async def safety_recheck():
            obs = report.safety
            if not obs.candidate_span.strip() or obs.candidate_span not in candidate_answer:
                self._emit("supervisor_action", target="safety", action="reject_observation", reason="candidate_span_not_in_original", observation=obs.model_dump())
                self.metrics["invalid_safety_observations"] += 1
                return None
            if obs.relation == "uncertain":
                self._emit("supervisor_action", target="safety", action="keep", reason="uncertain")
                return None
            if obs.relation == "benign" and obs.candidate_span != candidate_answer:
                self._emit("supervisor_action", target="safety", action="reject_observation", reason="benign_does_not_cover_entire_answer", observation=obs.model_dump())
                self.metrics["invalid_safety_observations"] += 1
                return None
            expected = obs.relation == "benign"
            if expected == original["safe"]:
                self._emit("supervisor_action", target="safety", action="keep", reason="observation_agrees", expected=expected)
                self.metrics["safety_review_agreements"] += 1
                return None
            patch = await rerun("safety", lambda: check_safety(
                self.client, self.model, messages, candidate_answer,
                review_feedback={"observation": obs.model_dump()},
            ), expected)
            if patch:
                self._emit("supervisor_action", target="safety", action="apply", before=original, after=patch)
            return patch

        tasks = [asyncio.create_task(grounding_recheck()), asyncio.create_task(safety_recheck())]
        try:
            patches = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        result = copy.deepcopy(initial)
        for patch in patches:
            if patch:
                result["output"].update(patch)
        return validated(result, candidate_answer)["output"]


    async def _review_scope(self, target, messages, candidate, evidence, initial, deadline):
        """One independent reviewer and at most one adjudication per target."""
        if deadline - time.monotonic() < 1:
            self._emit("supervisor_action", target=target, action="fallback", reason="deadline")
            return None
        current = (initial["input"] if target == "input" else
                   {k: initial["output"][k] for k in
                    (("grounded", "unsupported") if target == "grounding" else ("safe",))})
        if target == "input":
            operation = lambda: review_input_status(self.client, self.model, messages, current)
        elif target == "grounding":
            operation = lambda: review_grounding(self.client, self.model, messages, candidate, evidence, current)
        else:
            operation = lambda: review_safety(self.client, self.model, messages, candidate, current)
        self.metrics["review_calls"] += 1
        self.metrics[target + "_review_calls"] += 1
        try:
            report = await asyncio.wait_for(operation(), min(self.review_timeout, deadline-time.monotonic()))
            self._emit("review_report", target=target, response=report.model_dump())
            self.metrics["valid_"+target+"_reviews"] += 1
            self.metrics[target+"_status_"+report.status] += 1
            if report.status == "supported":
                self._emit("supervisor_action", target=target, action="keep", reason="review_supported")
                return None
            if report.status == "insufficient" and target != "grounding":
                self._emit("supervisor_action", target=target, action="keep", reason="insufficient_evidence", uncertainty=report.reason)
                return None
            self._validate_status_evidence(target, report, messages, candidate, evidence, current)
        except Exception as error:
            self.metrics[target+"_review_errors"] += 1
            logger.warning("guard_scope_review_failure target=%s type=%s", target, type(error).__name__)
            self._emit("supervisor_action", target=target, action="reject_review", reason=type(error).__name__)
            return None

        updated_evidence = evidence
        if report.status == "insufficient":
            # Insufficient evidence is not a refutation. Search once, preserve on no new source.
            if deadline - time.monotonic() < 4:
                self._emit("supervisor_action", target=target, action="keep", reason="research_budget", uncertainty=report.reason)
                return None
            self.metrics["research_calls"] += 1
            self._emit("supervisor_action", target=target, action="research", reason="insufficient_evidence", uncertainty=report.reason)
            try:
                query = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                extra = await asyncio.wait_for(self.searcher.search(
                    query + "\n검증 대상 주장:\n" + candidate + "\n부족한 정보:\n" + report.reason,
                    top_k=8), timeout=min(3,deadline-time.monotonic()))
                known={(e["section_id"],e["text"]) for e in evidence}
                if not any((e["section_id"],e["text"]) not in known for e in extra):
                    self._emit("supervisor_action", target=target, action="keep", reason="no_new_evidence", uncertainty=report.reason)
                    return None
                updated_evidence=list(extra)+[e for e in evidence if e["section_id"] not in {x["section_id"] for x in extra}]
            except Exception as error:
                self._emit("supervisor_action", target=target, action="fallback", reason=type(error).__name__)
                return None
        remaining=deadline-time.monotonic()
        if remaining < 1:
            self._emit("supervisor_action", target=target, action="fallback", reason="deadline")
            return None
        feedback={**report.model_dump(), "initial":current,
                  "instruction":"검토 의견을 정답으로 가정하지 말고 원문·정책·자료와 대조하세요. 반증이 성립하면 수정하고, 반증이 성립하지 않거나 자료가 여전히 부족하면 기존 판정을 유지하세요."}
        if target == "input":
            rerun=lambda: classify_input(self.client,self.model,messages,review_feedback=feedback,searcher=self.searcher)
        elif target == "grounding":
            rerun=lambda: check_grounding(self.client,self.model,messages,candidate,updated_evidence,review_feedback=feedback)
        else:
            rerun=lambda: check_safety(self.client,self.model,messages,candidate,review_feedback=feedback)
        self.metrics[target+"_recheck_calls"] += 1
        self._emit("supervisor_action", target=target, action="recheck", reason=report.status, feedback=feedback)
        try:
            verdict=await asyncio.wait_for(rerun(),min(self.review_timeout,remaining))
            patch=verdict.model_dump()
            self._emit("recheck_verdict",target=target,response=patch)
            if set(patch)!=set(current): raise ValueError("wrong_recheck_scope")
            proposal=copy.deepcopy(initial)
            if target=="input":proposal["input"]=patch
            else:proposal["output"].update(patch)
            validated(proposal,candidate)
            self._emit("supervisor_action",target=target,action="apply" if patch!=current else "keep",
                       reason="independent_recheck",before=current,after=patch)
            return patch
        except Exception as error:
            self.metrics[target+"_recheck_fallbacks"] += 1
            self._emit("supervisor_action",target=target,action="fallback",reason=type(error).__name__)
            return None

    @staticmethod
    def _validate_status_evidence(target, report, messages, candidate, evidence, current):
        if target == "input":
            sources={str(i):m["content"] for i,m in enumerate(messages)}
            if not report.citations or any(not c.quote.strip() or c.quote not in sources.get(c.source_id,"") for c in report.citations):
                raise ValueError("invalid_input_citation")
        elif target == "safety":
            obs=report.observation
            if not obs.candidate_span.strip() or obs.candidate_span not in candidate:
                raise ValueError("invalid_candidate_span")
            if obs.relation=="uncertain" or (obs.relation=="benign")==current["safe"]:
                raise ValueError("no_safety_refutation")
            if obs.relation=="benign" and obs.candidate_span!=candidate:
                raise ValueError("incomplete_safety_clearance")
        else:
            sources={e["section_id"]:e["text"] for e in evidence}
            for obs in report.observations:
                if not obs.candidate_span.strip() or obs.candidate_span not in candidate:
                    raise ValueError("invalid_candidate_span")
                if any(not c.quote.strip() or c.quote not in sources.get(c.source_id,"") for c in obs.citations):
                    raise ValueError("invalid_kb_citation")
                if obs.relation in {"supported","contradicted"} and not obs.citations:
                    raise ValueError("missing_kb_citation")
            if report.status=="refuted":
                if current["grounded"]:
                    if not any(o.relation=="contradicted" for o in report.observations):
                        raise ValueError("no_grounding_refutation")
                else:
                    if any(o.relation=="contradicted" for o in report.observations):
                        raise ValueError("contradiction_prevents_clearance")
                    clear=[o.candidate_span for o in report.observations if o.relation in {"supported","nonfactual"}]
                    if not all(any(span in c for c in clear) for span in current["unsupported"]):
                        raise ValueError("incomplete_grounding_clearance")
