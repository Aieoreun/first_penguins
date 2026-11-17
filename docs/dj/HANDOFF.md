# HANDOFF — 석동진 현재 상태 및 다음 할 일

- 갱신: 2026-11-18
- 작업 브랜치: **`experiment/bsuk1`** (`expert/bsuk1`에서 브랜치명 변경 + Step 10 팀 아이디어 통합 적용)
- 참고 브랜치: `origin/experiment/kimjaehak`, `origin/experiment/seungwon`, `origin/export/peace`

## 1. 현재 기준선 및 Step 10 통합 내역

| 구분 | 기준 커밋 | 점수 / 검증 결과 |
| --- | --- | --- |
| 이전 런박스 최고점 (Step 7) | `102cb2d` | 총점 **80.18** (튜터 73.22, 가드 87.14, 85.336s) |
| **Step 10 팀 아이디어 종합 업그레이드** | 현재 HEAD | `public/tutor.jsonl` Top-6 검색 재현율 **119/119 (100.0%)**, Top-1 **109/119 (91.6%)**, 결정론 신호 400문항 오탐 **0건 (76/76 100% 정밀도)** |

### Step 10 핵심 반영 사항 (`experiment/bsuk1` 전용)
1. **검색 정밀도·재현율 향상 (`app/kb_index.py`, `app/supervisor_graph.py`)**
   - Kiwi 형태소 분석 후 인접 내용어(2글자 이상·숫자) 바이그램 토큰 추가 + 멀티턴 전체 대화(user + assistant, 0턴 빠른질문 버튼 제외) 문맥 결합 + 후보 섹션 수 `top_k_sections=6` 확대 + Dense RRF 가중치 `0.55` 보정.
   - `public/tutor.jsonl` 근거 문항 119개 기준 Top-1 적중률 **84.0% → 91.6%**, Top-6 재현율 **99.2% → 100.0% (119/119)** 달성.
2. **`/tutor` 원문 소단원 재정렬 인용 (`focused_section_text`) + `answer_seeking` 근거 결합 + `plain_abstain` (`app/kb_index.py`, `app/supervisor_graph.py`, `app/tutor_pipeline.py`)**
   - **재학님(`focused_excerpt.py`) + 승원님(`v5`/`v6`) 핵심 아이디어 결합**:
     - `action == "answer"` (`normal`, `pii_disclosure`): LLM이 선택한 섹션(`citations[0]`) 내부의 `## ` 소단원들을 질문 관련도 순으로 재배치해 원문 전체(표·코드 포함, 최대 2,698자 < 4,000자)를 인용(`facts` ≈ 1.0, `faithful` ≈ 1.0 보장).
     - `action == "scaffold"` + `citations == [sid]` (`answer_seeking`): 주최 측 채점 기준에 맞춰 스스로 정답을 고르도록 유도하는 스캐폴딩 문구 아래에 선택된 섹션의 재정렬 원문을 함께 제공하여 `gold_facts` 누락을 원천 차단.
     - `action == "abstain"` (`out_of_material`): `citations=[]`(`evidence=[]`) 상태에서 과목 고유 수치를 언급해 `faithful=False`로 감점되는 현상을 막기 위해 깔끔한 고정 기권 문구(`_PLAIN_ABSTAIN_ANSWER`) 반환.
3. **`/guard` 정밀도·토큰 예산·구간 다듬기 고도화 (`app/guard_classifier.py`, `app/guard_supervisor.py`, `app/guard_verifier.py`, `app/harness_guardrails.py`, `app/prompts.py`)**
   - `GuardClassifier._llm_classify`, `GuardSupervisor._run_input_tiebreaker`, `_audit_safety`의 `low` 추론 토큰 상한을 `1000/1024 → 1500`으로 상향해 추론 토큰 소진으로 인한 `minimal` 폴백 방지.
   - 공개 400문항 전체에서 오탐 0건(76/76 100% 정밀도)이 검증된 7대 결정론 신호(`reconcile_guard_with_signals`)를 `/guard` 입력 판정에도 일관 적용.
   - 평화님의 `normal vs off_topic`(복습용 포스터·광고 형식 허용 vs 실제 홍보·게임·사과문자 차단) 및 `normal vs harmful_content`(사기 예방·차별 문구 필터링 실습 허용 vs 실제 사칭·타인정보 탈취·집단 비하 차단) 경계를 `GUARD_INPUT_TIEBREAKER_SYSTEM_PREFIX`에 반영.
   - `_normalize_with_map` NFKC 정규화, `_precheck_claims` 백틱 코드/명령어 대조 추가, 승원님 `v8` 아이디어인 `_trim_unsupported_span`(선행 번호 및 후행 서술어 어미/마침표 제거) 적용.

## 2. 다시 하지 말 것

1. **`reasoning_effort="low"`에서 `max_completion_tokens`를 1200 미만으로 깎지 않습니다.**
2. **k-NN `exemplars`를 빼지 않습니다.**
3. **`GUARD_OUTPUT_SYSTEM_PREFIX`에 "계산·비교 결론 허용" 문구를 넣지 않습니다.**
4. **Tie-Breaker 게이트에 `_OOM_DETAIL_CUE_RE`를 단독 OR 조건으로 걸지 않습니다.**

## 3. 실행·검증 명령

```bash
# 단위 테스트와 규정 사전 점검
pytest tests/test_pipeline.py -v
python3 scripts/submission_helper.py preflight

# 서버 기동
set -a; source .env; set +a
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```
