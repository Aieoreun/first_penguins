# HANDOFF — 석동진 현재 상태 및 다음 할 일

- 갱신: 2026-11-18
- 작업 브랜치: **`experiment/bsuk1`** (Step 11: 3분할 직교 입력 + 적응형 2-of-3 다수결 + 전용 Source Selection + 5-vote 합의 출력 가드)
- 참고 브랜치: `origin/experiment/kimjaehak`, `origin/experiment/seungwon`, `origin/export/peace`

## 1. 제출 이력 및 Step 10 → Step 11 전환 배경

| 구분 | 기준 커밋 | 점수 | 핵심 진단 |
| --- | --- | --- | --- |
| Step 7 | `102cb2d` | 80.18 (튜터 73.22, 가드 87.14, 85.3s) | 단일패스 + GuardSupervisor |
| Step 10 | `e2485b3` | 80.09 (튜터 74.33, 가드 85.84, 87.8s) | `focused_section_text`로 튜터 +1.11p 상승했으나, 단일 10/11분류기(`write_node`) 한계 및 `scaffold`에 2,200자 전체 원문이 들어가 `out_of_material`/`scaffold`에서 손실 발생 |
| **Step 11** | 현재 HEAD | (런박스 검증 예정) | **단일 10/11분류기 폐기 → 3분할 직교 입력(`block` / `other` / `material`) + `confidence` 하한선 + 전용 Source Select + `SEEK_SYSTEM` + 5-vote 합의 출력 가드로 전면 전환** |

### Step 11 아키텍처 (`PIPELINE_MODE=orthogonal`, 기본값)
1. **이중 세분화 + 바이그램 하이브리드 검색 (`app/kb_index.py`)**:
   - 기존 `##` 청크(389) + `### Q.`·문단 단위 짧은 청크(718) + 인접 내용어 바이그램 + Dense RRF 결합.
   - `public/tutor.jsonl` 119문항 기준 Top-1 **91.6%**, Top-6 **100.0% (119/119)** 달성.
2. **3분할 직교 입력 분류 + 적응형 2-of-3 다수결 + 확신도 하한 (`app/orthogonal_guard.py`)**:
   - `block` (`requested_operation` CoT + 5종, 2회 호출 불일치 시 3회차, `min_conf=85` 단 `harmful_request`/`harmful_speech` 보호)
   - `other` (`emotional_distress`/`answer_seeking`/`low_signal_input` + `self_info`, 2회 호출 불일치 시 3회차, `min_conf=80`)
   - `material` (이중 청크 기반 원문 `quote` 유무 확인, `/tutor` 13s·`/guard` 10s 상한)
   - `seek_first=True` 우선순위 + 오탐 0건 검증 완료된 결정론 안전 신호 보정(명시적 탈옥/과제대필이 `answer_seeking`에 의해 덮이지 않도록 보호).
3. **`/tutor` 전용 출처 선택 + 소단원 재정렬 원문 인용 (`app/orthogonal_tutor.py`)**:
   - `normal`/`pii_disclosure`: 전용 `SOURCE_SELECT_SYSTEM`으로 최대 6후보 중 최적 섹션 1개만 선별 → `focused_section_text` 원문 인용(`facts` ≈ 1.0, `faithful` ≈ 1.0).
   - `answer_seeking`: 전용 `SOURCE_SELECT_SYSTEM`으로 섹션 확정 → `SEEK_SYSTEM`으로 핵심 사실·수치만 포함한 2~3문장 발판 생성(코드 원문 전체 덤프 방지).
   - `harmful_content`: `harmful_request`→`refuse`, `harmful_speech`→`redirect` 분리 + 문맥형 안전 거절.
   - `out_of_material`/`jailbreak`/`off_topic`/`academic_dishonesty`/`emotional_distress`/`low_signal`: 고정 정책 응답(`citations=[]`).
4. **`/guard` 5-vote 합의 출력 검증 + 거절 의미 검증 + 스팬 트리밍 (`app/orthogonal_guard.py`)**:
   - `block` 입력: `safe_by_rule` + 거절 표현 감지 시 `RefusalReview` 의미 검증.
   - `allow` 입력: `CONTEXT_GROUNDING_SYSTEM` 5회(잔여 예산 부족 시 3회) 병렬 검증 → `_snap_span_to_verbatim` + `_trim_unsupported_span` → 문자 위치 합의(`consensus_span_index`) 채택.

## 2. 실행·검증 명령

```bash
# 단위 테스트와 규정 사전 점검
pytest tests/test_pipeline.py -v
python3 scripts/submission_helper.py preflight

# 런박스 서버 기동 (기본 PIPELINE_MODE=orthogonal)
set -a; source .env; set +a
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```
