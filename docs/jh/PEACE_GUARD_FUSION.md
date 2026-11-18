# Peace 가드 융합 실험 — 2026-11-18

## 범위와 기준

- 사용자 지정 브랜치: `experiment/kimjaehak-peace-guard-fusion`, 시작 `ff0aa5e`.
- 기준선: `jh_r4_material_budget`, 기존 최종 후보와 기본 서버 설정 보존.
- Peace 대조: `3c64079eb1af14170b3bee4578ac6f916bec433e` 원본 스냅샷의 `jh_parallel_guard`.
- 융합: `jh_pg_fusion`. 우리 학습형 입력 판정·span 보정과 Peace 병렬 검사·독립 안전성 판정을 결합한다.
- 튜터는 MaterialBudget 상속 그대로. 평가기·core·server.py·SPEC·public·KB 수정 없음. 비스트리밍 호출 유지.
- 사용량 요청의 “80퍼센트까지”는 우선 **사용 80%, 잔여 20%에서 새 실험 중단**으로 해석했다. 비동기 확인 질문은 대기 중이다. 시작 사용 0%, 구현 확인 시 2%. 계정 공유 사용량이며 이 작업만의 소비량은 아니다.

## 구조와 변경 의미

검색 뒤 입력 판정, 후보 답변의 근거 검사 5회, 독립 안전성 검사를 동시에 시작한다. 입력 판정은 기존 로지스틱 회귀 fast path를 유지하고, LLM fallback에서는 Peace처럼 block/other 투표 뒤 material 판정을 한다. 전체 가드 시간은 슬롯 대기 포함 28초다. ground/safety는 각 13초, 입력 준비는 26초 범위 안에서 실행한다.

근거 검사 일부가 제한 시간에 걸려도 완료된 투표만 합의에 사용하고, 우리 span 모델로 선택된 오류 구절의 경계를 보정한다. 입력이 BLOCK이면 기존 규격대로 grounded=true, unsupported=[]로 병합한다. 안전성은 별도 상태에서 구하며 grounding의 safe 값으로 덮어쓰지 않는다. 기존 refusal review를 독립 safety 검사로 대체하는 것이므로 단순한 병렬화 실험은 아니다. 외부 취소와 시간 초과 때 모든 하위 작업을 취소·회수한다.

## 검증 순서

1. 실제 실행 클래스와 슬롯 수를 startup에서 확인하는 전용 비교 하네스 준비. Peace launcher의 표시값과 실제 server route 불일치를 피하고 원본 server를 실행한다.
2. 공개 guard200을 우리/Peace/융합 순서로 수집 후 원본 평가기로 채점. 모든 후보 slots64, 같은 .env·임베딩 색인. 공개자료는 우리 모델 학습에 사용했으므로 이 점수는 학습 내 성능이다.
3. 기존 입력 변형, novel answer60, refusal30으로 약점 분해. 이 자료도 개발에 사용한 진단자료이며 독립 holdout이 아니다. refusal30은 safe accuracy만 해석한다.
4. 실패 유형에 따라 입력 프롬프트·안전성 rubric·근거 투표/문맥을 하나씩 변경하고 대조 반복. 튜터 혼합 부하도 최종 점검한다.
5. 수집·채점·오류·기각 사유를 함께 기록하고, 중단선에 도달하면 새 수집 없이 남은 채점과 인수인계를 완료한다.

## 초기 검증

- 전체 테스트 636개 통과. 새 fusion 테스트 12개, 비교 하네스 테스트 10개 포함.
- `--prepare-only --name harness_verify_noapi`: 공개200 준비 및 Peace archive/protected file 해시 일치 확인, API 호출 0.
- 아직 이 라운드의 API 점수 결과는 없다. Peace 검토 기록은 [PEACE_REVIEW_20261118.md](PEACE_REVIEW_20261118.md).

## 실행 결과

후속 수집과 분석을 이 절에 누적한다. 실제 제출 점수와 로컬 진단 점수는 분리한다.
