# R6 — 원격 팀원 작업 재검토와 점수 개선

2026-11-18 KST. 사용자 재개 지시: 사용량 **잔여15%**까지 실험 → 분석 → 다음 실험을 반복한다.
시작 사용률63%, 잔여37%. 잔여15% 이하를 관측하면 새 수집을 시작하지 않고 진행 중인 채점과
코드·결과·인수인계를 정리해 커밋한다. 이전 R4/R5 종료 기록은 역사로 보존한다.
사용량은 계정의 다른 작업과 공유하며 각 실험 묶음 시작 전/후 확인한다.

## 브랜치와 출처

`git fetch origin --prune` 후 `git pull --ff-only origin experiment/kimjaehak`: 이미 최신 `b199844`.
각 브랜치는 개인 실험 공간이다. 다른 팀원 브랜치를 병합·수정·푸시하지 않고 이번 작업은
`experiment/kimjaehak`에만 기록한다. 팀원 점수는 각자의 평가 조건이며 우리 점수와 단순 합산하지 않는다.

| 원격 브랜치 | 확인 HEAD | 확인한 변화 / 판단 |
|---|---|---|
| experiment/seungwon | f3e320b | vx 확신도 하한·적응형 재확인·material section 출력 제거·v8 span 다듬기. 낮은 확신 제거는 오차단을 줄이지만 튜터 손실과 반복 편차가 있어 기본 적용하지 않는다. material 불필요 필드 제거를 독립 검증한다. medium·재시도는 팀원 실험의 시간 실패를 고려해 후순위. |
| experiment/bsuk1 | de664f2 | 기존 expert/bsuk1 대신 새 이름. Step11의 직교 입력·적응 투표·focused 출력 상당 부분은 우리 코드와 공통이다. 짧은 Q/A 근거를 material 판정에 공급하는 방식과 마지막 질의 우대 검색이 새로운 후보. Kiwi/내용어 bigram/RRF의 공개119 top6=100%는 독립 검증 전 수치다. Step11 실제 제출은 아직 기록 없음. |
| export/peace | f7e3940 | 요청 본문 없는 오류/시간 로그 보존. 내용 기반 채점 자료를 수집하는 기능으로 옮기지 않는다. |
| export/peace-jh-full-no-trained | 2f51c1f | 우리 b199844 기반에서 학습형 모델을 끈 별도 실험. 공개 tutor 평균86.50·guard90.52는 원본과 재대조하지 않은 세 번 평가다. 새 parallel guard는 아직 결과 미확인. 학습 경로 제외 전체를 복제하지 않고 tutor의 고신뢰 출처 재확인으로 좁힌다. |
| dev / main | 429330c / 2043463 | 새 변경 없음. |

팀원 근거: 승원 `docs/sw/05_로컬_점수표.md`, `pipelines/vx/{pipeline,prompts}.py`;
동진 `docs/dj/HANDOFF.md`, `app/kb_index.py`, `app/orthogonal_guard.py`;
Peace `docs/PEACE_FULL_PORT{,_RESULTS}.md`, `pipelines/kimjaehak/parallel_guard.py`.

## 유지 조건

- 기존 실제 제출 추천 `jh_r4_material_budget`, 백업 focused, R5 dual6/all을 보존한다.
- 주최측 `grader`, `harness/score_local.py`, `core`, `server.py`, `SPEC`, `public`, `kb` 원본 유지.
- 우선 가드를 고정하고 튜터만 변경한다. 비스트리밍, 전체 deadline26초, slots64, embedding env.
- 제공 API만 사용. 비공개 입력/답변을 저장하거나 정답을 런타임에 전달하지 않는다.
- 개발 자료는 공개 파생 primary192 / alternative181, 교재 검토 질문의 topic114 / followup114.
  모두 개발에 사용한 자료이며 독립 비공개 점수가 아니다. 우수 후보는 새 기준선·역순 반복·다른 자료로 확인한다.
- 코드 커밋 후 깨끗한 트리에서 실행. 수집과 원본 평가기 채점을 겹치지 않는다. 단회 최고점으로 채택하지 않는다.

## 1차 후보 — 결과 확인 전 계획

| 후보 | 비교 기준 | 바뀌는 것 |
|---|---|---|
| jh_r6_short_material | jh_r5_dual_all | 짧은 dense 검색 근거 중 기존 material에 포함되지 않은 최대4청크를 material 판정 입력에 추가. 기존 근거는 모두 유지. state.chunks와 출처 후보 구성·가드·LLM 횟수는 유지. |
| jh_r6_material_nosection | jh_r4_material_budget | tutor material 프롬프트에서 section 설명·출력 필드만 제거. asked/quote/confidence, 예산, 검색, 답변은 유지. |
| jh_r6_context_verify | jh_r4_material_budget | 다중 발화 normal/PII tutor에서 학습형 출처를 후보로 보존하면서 기존 source LLM으로 재확인. 단일 발화·다른 정책·가드는 유지. |

단위 테스트는 추가근거 중복/한도·원본 보존, tutor/guard 분리, 프롬프트 필드 제거,
문맥 재확인 조건·비스트리밍·기존 시간 제한을 검사한다.
첫 묶음은 topic114에서 기준선2개와 신규3개를 비교하고, primary192 및 followup114에서 유망 후보를 검증한다.
결과에 따라 마지막 질문 우대 검색·범주 판정·근거 선택을 후속 실험한다.

## 실행 결과와 다음 판단

수집 전. 실행별 commit·데이터/평가기 해시·설정·오류·시간·점수는 로컬 캐시에 보존하고
여기에 집계와 해석을 이어 적는다.
