"""Distinguish similarly named exercises without changing retrieval or guard."""
from .focused_excerpt import Selected as SelectedFocused


class Pipeline(SelectedFocused):
    source_disambiguation = True
    source_extract_system = SelectedFocused.source_extract_system + """
같은 이름이나 소재가 여러 실습에서 반복될 수 있습니다. 이름이 일치한다는 이유만으로 선택하지 마세요.
질문이 지칭하는 자료형, 연산, 변화 규칙, 비교 대상이 그 교재의 실습과 일치하는지 대조하세요.
예를 들어 리스트·집합·튜플은 같은 소재를 써도 서로 다른 실습이며, 한 값의 반복 증감과 두 값의 비교도 다릅니다.
질문에서 주어진 조건과 교재가 명시한 조건이 충돌하면 그 섹션을 근거로 고르지 마세요.
질문이 묻는 미지의 값 자체가 질문에 없다는 이유로 근거를 제외하지는 마세요.
"""
