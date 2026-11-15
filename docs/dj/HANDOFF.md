# HANDOFF — 석동진 현재 상태 및 다음 할 일

- 갱신: 2026-11-16
- 문서 브랜치: `expert/bsuk1` (`docs/dj/` 최신 정리 완료)
- 코드 실험 브랜치: `feature/step6-dual-supervisor` (최고 총점 커밋 `102cb2d`, 최신 실험 커밋 `cc1aa25`)
- 참고 팀원 브랜치: `origin/experiment/seungwon` (`bef5614` / `6ec111d`, `v4`)

---

## 1. 검증된 최고 성능 기준선 정리

| 목적 | 권장 기준 코드 | 런박스 공식 점수 |
| --- | --- | --- |
| **현재 보유 최고 총점** | `feature/step6-dual-supervisor` @ **`102cb2d`** | **총점 `80.18`** (튜터 `73.22`, 가드 `87.14`, `85.336s`) |
| **`/tutor` 최고 구성** | `102cb2d` (단일패스 + Kiwi/Dense RRF + `public/tutor.jsonl` `exemplars` + 안전 후처리) | `72.75 ~ 73.67` (4연속 무수정 기준 평균 ~`73.2`) |
| **`/guard` 최고 구성 (팀내)** | `origin/experiment/seungwon` @ **`bef5614` (`v4`)** | **가드 `88.14`** (`v4` 3분할+예시+1500tok+`safe_by_rule`+`locate_all`) |

---

## 2. 핵심 교훈 요약 (재시도 금지 사항)

1. **`gpt-5-nano`에서 `reasoning_effort="low"` 시 `max_completion_tokens`는 최소 `1200~1800` 확보** (`850/950`으로 깎으면 `minimal` 폴백으로 정확도 급락).
2. **`public/guard.jsonl` 및 `public/tutor.jsonl` 동적 k-NN `exemplars`를 제거하지 않는다.** 제거 시 Step 6c(`/tutor -2.25p`)와 Step 8(`/guard -3.14p`) 모두 하락.
3. **`GUARD_OUTPUT_SYSTEM_PREFIX`에 "계산·비교 결론 허용" 완화 문구/예시를 넣지 않는다.** `34e1607`에서 이 변경만으로 `Guard 87.14 → 81.67 (-5.47p)` 하락, 원복 직후 `86.25 (+4.58p)` 회복.
4. **Tie-Breaker 게이트에 `_OOM_DETAIL_CUE_RE` 단독 `OR` 조건을 걸지 않는다.** `금리/한도/얼마` 등 정상 금융 질문 대부분에서 2차 호출이 발동해 `+7.4s` 지연과 오역전을 유발.

---

## 3. 다음 유망 실행 안 (우선순위)

- **안 A (가장 높은 기대 점수 ~`80.8+`): 검증된 챔피언 1:1 직결**
  - `/tutor`: **`102cb2d` 그대로 유지 (`73.2~73.7`)**
  - `/guard`: 승원 `origin/experiment/seungwon` **`v4` 원본 코드·프롬프트 그대로(`# 예시` 포함 원문 4종 프롬프트, `max_tokens=1500`, `candidate_answer` 포함 검색, `locate_all`, `safe_by_rule=True`)** 이식 + 우리 `BadRequestError -> minimal` 재시도만 결합.
- **안 B (`102cb2d` 경량 정밀 튜닝, `~85s` 유지)**
  - `cc1aa25`에서 과잉 발동한 `_OOM_DETAIL_CUE_RE` Tie-Breaker와 `2,200자 × 3섹션` 장문 주입(`top_k=3`)만 `102cb2d`(`top_k=2, 700자`) 수준으로 원복.
  - 로컬 감사에서 무오탐(`FP=0`)이 입증된 **`NFKC` 정규화 + `_precheck_claims` 백틱 코드 대조(`TP 7/35 → 10/35, FP 0/32`) + `enforce_high_precision_signals` 7종**만 남김.

---

## 4. 브랜치 전환 및 실행 명령어

```bash
# 최고 총점(80.18) 코드로 복귀해 바로 배포/실험할 때
git checkout feature/step6-dual-supervisor
git reset --hard 102cb2d   # 또는 cc1aa25에서 안 B 적용

# 런박스 배포 및 서버 기동
set -a; source .env; set +a
python3 scripts/submission_helper.py preflight
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```
