from typing import Literal

from pydantic import BaseModel, ConfigDict


Category = Literal[
    "normal",
    "answer_seeking",
    "academic_dishonesty",
    "off_topic",
    "out_of_material",
    "jailbreak_injection",
    "pii_disclosure",
    "harmful_content",
    "emotional_distress",
    "low_signal_input",
]


class InputVerdict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    decision: Literal["allow", "block"]
    category: Category


class OutputVerdict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    grounded: bool
    unsupported: list[str]
    safe: bool
    category: str = "normal"


class GuardResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    input: InputVerdict
    output: OutputVerdict | None = None


def validate_guard_response(
    payload: dict,
    candidate_answer: str | None = None,
) -> GuardResponse:
    result = GuardResponse.model_validate(payload)

    if candidate_answer is not None:
        if result.output is None:
            raise ValueError("후보 답변이 있으면 output이 필요합니다.")

        for span in result.output.unsupported:
            if not span.strip():
                raise ValueError("unsupported에 빈 구간을 넣을 수 없습니다.")

            if span not in candidate_answer:
                raise ValueError("unsupported 구간이 후보 답변 원문에 없습니다.")

    return result
