"""Compare fewer low-effort votes with bounded non-streaming medium reasoning."""
import json

from .peace_span_variants import RoutedSpan80


class LowThree(RoutedSpan80):
    output_votes = 3


class MediumThree(LowThree):
    guard_ground_timeout = 20.0
    guard_ground_effort = 'medium'
    guard_ground_max_tokens = 2000

    async def _ground_response(self, payload):
        return await self.llm.chat(self.unsupported_system, json.dumps(payload, ensure_ascii=False),
            effort=self.guard_ground_effort, max_tokens=self.guard_ground_max_tokens,
            timeout=self.guard_ground_timeout, json_mode=True)
