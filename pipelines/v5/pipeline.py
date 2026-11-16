"""v5 흐름. v4와 구조가 같고, 정답만 요구(answer_seeking)의 힌트 답변 프롬프트만 다르다.

v4는 힌트 답변에서 교재의 값을 말하지 않았다. v5는 교재에 적힌 규칙·값을 말하고 최종 답만 학생에게 남긴다.
2026-11-16 주최 측 답변에 맞춘 것이다 (pipelines/v5/prompts.py, docs/sw/02_설계.md 결정 12-2).
"""
from pipelines.v4.pipeline import Pipeline as V4Pipeline
from pipelines.v5 import prompts


class Pipeline(V4Pipeline):
    answer_systems = prompts.ANSWER_SYSTEMS
