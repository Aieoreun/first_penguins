"""Distinguish exact comparison conditions from user-described scenarios."""
from .learned_output import Consensus


class Pipeline(Consensus):
    unsupported_system = Consensus.unsupported_system + """

# 대화 맥락과 조건을 대조하는 추가 원칙
- 비교 실습은 교재에서 비교한 두 조건을 정확히 대조한다. 어떤 조건을 쓰지 않은 경우와 쓴 경우를
  비교하는 것과, 서로 다른 조건을 각각 쓴 두 경우를 비교하는 것은 같은 뜻이 아니다.
  후보가 비교 대상의 조건이나 개입 여부를 바꿨으면 그 부분은 근거 없는 사실이다.
- 후보가 사용자가 말한 상황을 받아서 일반적인 예방·보호 방법을 안내하는 부분은, 교재에서 정한
  상품 규칙이나 과정 고유 사실을 새로 주장하는 것과 구별한다. 대화에서 주어진 위험 상황을
  가정형으로 언급하고 안전한 행동을 권하는 것만으로 근거 없는 구절로 잡지 않는다.
- 이 원칙이 교재의 실제 숫자·사용 조건·순서를 바꾸는 것을 허용하지는 않는다. 그러한 고유 사실은
  반드시 교재와 일치해야 한다. 오류 구절은 여전히 후보의 실제 글자만 짧게 출력한다.
"""


class Reasoned(Pipeline):
    # Unlike the earlier 13-second medium check, fast learned input leaves
    # room for one longer contextual check inside the 26-second deadline.
    output_votes = 1
    output_effort = "medium"
    output_max_tokens = 4000
    output_timeout = 20

    async def _ground_response(self, payload):
        import json
        from .reasoned import stream_text
        return await stream_text(self.llm, self.unsupported_system, json.dumps(payload, ensure_ascii=False),
                                 effort=self.output_effort, max_tokens=self.output_max_tokens,
                                 timeout=self.output_timeout)
