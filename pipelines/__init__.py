"""요청 처리 흐름. 구조가 다른 흐름을 폴더로 나눠 두고 환경변수 PIPELINE으로 하나를 고른다.

  v1: 검색 → 입력 가드(호출 한 번) → 답변 생성 또는 출력 가드
  v2: 리라이터 → 검색 → 입력 가드(셋으로 나눠 동시에) → 답변 생성 또는 출력 가드

검색·모델 호출·구절 찾기 같은 부품은 core/에 있고 모든 흐름이 같이 쓴다.
"""
import importlib
import os

DEFAULT = "v1"


def load(name: str = ""):
    """흐름 이름으로 Pipeline 클래스를 돌려준다."""
    name = name or os.environ.get("PIPELINE", DEFAULT)
    return importlib.import_module(f"pipelines.{name}.pipeline").Pipeline
