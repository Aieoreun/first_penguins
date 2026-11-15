# HANDOFF — 석동진 현재 상태 및 다음 할 일

- 갱신: 2026-11-16
- 작업 브랜치: **`expert/bsuk1`** (Step 7 `102cb2d` 병합 완료)
- 참고 브랜치: `origin/experiment/seungwon` (`bef5614` / `6ec111d`, v4)

## 1. 현재 기준선

| 목적 | 기준 | 점수 |
| --- | --- | --- |
| **현재 기본 코드 (최고 총점)** | `102cb2d` | 총점 **80.18** (튜터 73.22, 가드 87.14, 85.336s) |
| `/tutor` 최고 구성 | `102cb2d` (단일패스 + Kiwi/Dense RRF + `exemplars` + 안전 후처리) | 72.75 ~ 73.67 (무수정 4회 평균 약 73.2) |
| `/guard` 최고 구성 (팀 내) | 승원 `bef5614` (v4) | 가드 88.14 |

## 2. 다시 하지 말 것

1. **`reasoning_effort="low"`에서 `max_completion_tokens`를 1200 미만으로 깎지 않습니다.** 850/950으로 줄이면 `minimal` 폴백으로 정확도가 급락했습니다 (Step 8).
2. **k-NN `exemplars`를 빼지 않습니다.** 뺐을 때 Step 6c(`/tutor` -2.25p)와 Step 8(`/guard` -3.14p)이 모두 하락했습니다.
3. **`GUARD_OUTPUT_SYSTEM_PREFIX`에 "계산·비교 결론 허용" 문구를 넣지 않습니다.** `34e1607`에서 가드가 87.14 → 81.67로 떨어졌고, 되돌리자 86.25로 회복됐습니다.
4. **Tie-Breaker 게이트에 `_OOM_DETAIL_CUE_RE`를 단독 OR 조건으로 걸지 않습니다.** 정상 금융 질문에서 2차 호출이 자주 발동해 약 +7.4s 지연과 오판 역전이 생겼습니다.

## 3. 다음 후보 실험

### 안 A — 챔피언 직결 (기대 총점 약 80.8 이상)
- `/tutor`: 현재 `102cb2d` 그대로 유지
- `/guard`: 승원 v4 코드·프롬프트를 원문 그대로 이식
  - `# 예시` 포함 4종 프롬프트, `max_tokens=1500`
  - `candidate_answer` 포함 검색, `locate_all`, `safe_by_rule=True`
  - 우리 `BadRequestError → minimal` 재시도만 결합

### 안 B — 경량 튜닝 (약 85s 유지)
`102cb2d` 위에, 로컬 감사에서 오탐 0건(FP=0)이 확인된 세 가지만 적용합니다.
1. `_normalize_with_map`에 `unicodedata.normalize("NFKC", ch)` 추가
2. `_precheck_claims`에 백틱 코드·명령어 대조 추가 (TP 7/35 → 10/35, FP 0/32)
3. 정밀도 100% 결정론 신호 7종을 사후 고정: `has_other_pii_dox`, `has_hate_insult`, `has_imperative_jailbreak`, `has_emotional_distress`, `has_academic_dishonesty`, `has_answer_seeking`, `has_self_pii`

## 4. 실행·검증 명령

```bash
# 단위 테스트와 규정 사전 점검
pytest tests/test_pipeline.py -v
python3 scripts/submission_helper.py preflight

# 서버 기동
set -a; source .env; set +a
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```
