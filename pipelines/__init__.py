"""요청 처리 흐름. 구조가 다른 흐름을 폴더로 나눠 두고 환경변수 PIPELINE으로 하나를 고른다.

  v1: 검색 → 입력 가드(호출 한 번) → 답변 생성 또는 출력 가드
  v2: 검색 → 입력 가드(셋으로 나눠 동시에) → 답변 생성 또는 출력 가드. 실험용 스위치가 있다
  v3: v2에 검증한 설정을 고정한 것 (다수결, 새 답변 프롬프트 등)
  v4: v3에서 다수결을 빼고(확인을 한 번씩만) 요청 전체의 마감 시간을 둔 것

검색·모델 호출·구절 찾기 같은 부품은 core/에 있고 모든 흐름이 같이 쓴다.
"""
import importlib
import os

DEFAULT = "v1"


def load(name: str = ""):
    """흐름 이름으로 Pipeline 클래스를 돌려준다."""
    name = name or os.environ.get("PIPELINE", DEFAULT)
    if name.startswith("jh_"):
        from pipelines.kimjaehak.config import VARIANTS, configure
        if name not in VARIANTS:
            raise ValueError(f"Unknown experiment pipeline: {name}")
        configure()
        if name == "jh_baseline":
            return importlib.import_module("pipelines.v2.pipeline").Pipeline
        if name.startswith("jh_guided_"):
            module = importlib.import_module("pipelines.kimjaehak.guided")
            return {"jh_guided_input": module.GuidedInput,
                    "jh_guided_output": module.GuidedOutput,
                    "jh_guided_both": module.GuidedBoth,
                    "jh_guided_focus": module.FocusInput}[name]
        module = importlib.import_module("pipelines.kimjaehak.pipeline")
        return {"jh_peace_input": module.PeaceInput,
                "jh_peace_output": module.PeaceOutput,
                "jh_peace_both": module.PeaceBoth}[name]
    return importlib.import_module(f"pipelines.{name}.pipeline").Pipeline
