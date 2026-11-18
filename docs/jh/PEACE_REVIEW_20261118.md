# Peace 원격 작업 검토 — 2026-11-18

## 확인 범위

- `git fetch origin`, `git pull --ff-only origin experiment/kimjaehak` 완료. 현재 브랜치는 이미 최신 `ff0aa5e`였다.
- 변경된 Peace 원격: `export/peace-jh-full-no-trained`, `881635f` → `3c64079eb1af14170b3bee4578ac6f916bec433e`.
- 별도 `export/peace`는 `f7e3940` 그대로다. 팀원 브랜치를 현재 브랜치에 병합하거나 수정하지 않았다.
- 로컬 임시 디렉터리에 해당 커밋의 Git archive를 풀어 검토했다. 대회 API 호출·새 점수 실험은 없다.

## 우리 작업을 적용한 범위

Peace의 `docs/PEACE_FULL_PORT.md`는 우리 `b19984475c2dd0c8fa54004bedf95f206ef55ed8` 전체 트리를
참고했다고 명시한다. material_budget/focused의 검색·입력 투표·섹션 선택·교재 인용·튜터 시간 예산을 가져왔다.
R5 dual 코드도 트리에 존재하지만 현재 실행 경로는 dual이 아니다. R6 파일들은 포함돼 있지 않다.

중요한 차이는 `material_budget.Pipeline.learned_models_enabled=False`다.
입력 category, guard 전용 category, section, harmful subtype, output span의 학습과 추론을 건너뛰고
기존 LLM 경로로 진행한다. 로지스틱 회귀 함수가 파일에 남아 있어도 선택된 운영 경로에서 활성화한 것이 아니다.

## Peace의 추가 변경

1. `jh_parallel_guard`가 비학습 material_budget을 상속한다. 검색 후 입력 판정·근거 검사·독립 안전 검사를 병렬로 진행한다.
2. 입력 판정 안에서는 block/other 투표를 병렬 실행한 다음 material 판정을 시작한다. 근거 판정은5회 합의다.
3. guard prepare26초/전체28초/material13초, request_slots64. tutor는 부모의 전체26초/material13초/prepare16초다.
4. 최신 BLOCK 프롬프트는 `requested_action`을 추가하고, 공격 문구의 분석·방어 학습과 현재 튜터에게 공격 실행을 요구하는 것을 구분한다.
5. 기본 `server.py`는 `jh_parallel_guard`를 고정 선택한다. public_guard_trace는 고정 공개 guard 데이터만 추적하는 개발 도구다.

## 재현한 검토 사항

### P1 — 실험 variant/메타데이터와 실제 서버가 달라질 수 있음

`server.py:28`이 `PIPELINE="jh_parallel_guard"`로 고정되어 있다.
반면 `experiments/jh_server.py:30–44`는 CLI variant를 로드하고 그 클래스의 슬롯을 수정한 뒤 해당 메타데이터를 출력한다.
이후 Uvicorn이 import하는 server는 별도의 고정 variant를 사용한다.

네트워크 없는 mock startup 재현:

| 항목 | 값 |
|---|---|
| 요청 | `--variant jh_r4_material_budget --request-slots 7` |
| EXPERIMENT_CONFIG | `material_budget.Pipeline`, slots7 |
| server 실제 선택 | `jh_parallel_guard`, `parallel_guard.Pipeline`, slots64 |

따라서 기존 비교 실행기와 `peace_full_http.py --variant ...`의 라벨을 그대로 믿으면 다른 알고리즘의 결과로 기록할 수 있다.
의도한 기본값은 유지하면서 `os.environ.get("PIPELINE", "jh_parallel_guard")`로 명시 선택을 존중하거나,
고정 서버를 유지한다면 실험 실행기가 실제 선택 클래스를 설정하고 검증하도록 분리해야 한다.
이 변경 이전 `8193e2f`에서 기록한 기존 점수가 틀렸다고 소급 주장하지 않는다.

### P2 — 근거 판정 일부가 끝나도 한 호출의 지연으로 결과가 사라짐

`parallel_guard.py:22–31`은5개 작업의 gather가 모두 끝난 뒤에만 성공한 근거 판정을 state에 반영한다.
`guard()`는 이 전체 작업을 SECOND_TIMEOUT으로 제한하므로, 일부 호출이 완료돼도 전체가 취소되면 합의 적용까지 도달하지 못한다.

실제 Pipeline의 guard 경로에 모의 응답을 넣고 시간만 축소한 재현:

- 4개 근거 호출은 정상 완료해 모두 `unsupported=["claim"]`을 반환.
- 마지막1개는 대기, 입력 NORMAL·독립 안전 판정은 완료.
- 전체 근거 제한에 걸린 뒤 최종 응답은 `grounded=true`, `unsupported=[]`, 오류는 `ground:Deadline`.

완료한 작업을 별도로 모으고 남은 작업을 취소·회수한 뒤, 유효한 완료 판정으로 합의를 적용하는 방식을 검토할 만하다.
원본 합의 코드에도 유사한 취소 취약성이 있으므로 Peace만 새로 만든 오류라고 단정하지 않는다.
실제 대회에서의 발생 빈도나 점수 영향은 이 모의 재현으로 알 수 없다.

## 점수 기록의 적용 범위

원격에 추가된 평가 문서는 `8193e2f`의 초기 비학습 이식 결과다.

| 공개200문항씩3회 | 평균 점수 |
|---|---:|
| tutor |86.50|
| guard |90.52|
| 두 endpoint 평균의50:50 합산 |88.51|

HTTP·형식·채점 오류0이지만 내부 오류 요청은 tutor30/600, guard54/600이다.
최신 병렬화·BLOCK 변경·64슬롯을 포함한 `3c64079`의 점수표는 확인한 원격 추적 파일에 없다.
최신 커밋 메시지에는 public 평가 수행 언급이 있으나 수치·응답·동일 조건 대조 자료가 없어 개선폭을 확인할 수 없다.
기존 설명서의 material_budget 실행 명령과 v1 기본값 설명도 최신 고정 서버와 맞지 않는다.

## 검증과 이식 판단

- no-trained/parallel-guard/vote-first/HTTP 전송의 관련21개 테스트 통과.
- 첫 실행의 localhost 바인딩4건은 샌드박스 권한 오류였고, 로컬 네트워크 권한으로 재실행해21개 모두 통과했다.
- mock으로 학습 함수가 호출되지 않음, variant 불일치, 완료 근거 판정 유실을 확인했다. 실제 LLM 의미 판정 개선 검증은 아니다.
- 우리 참고 커밋b199844 대비 grader·harness/score_local.py·core·public·KB·SPEC diff0.

참고할 우선순위는 BLOCK의 실제 요청 행동 구분과 독립 안전 검사 병렬화다.
BLOCK은 우리 R4 structured 실험에서 정상 회복과 실제 탈옥 오허용이 함께 있었으므로 독립 후보로 비교해야 한다.
가드를 고정해 온 우리 실험에는 병렬 가드를 그대로 합치지 않고 별도 후보로 검증하는 편이 맞다.
로지스틱 회귀 비활성화는 그 자체로 개선 근거가 아니며 같은 조건에서 학습형 ON/OFF 대조가 필요하다.
우선 실행기/메타데이터 일치를 고친 다음 최신 점수를 측정해야 비교 결과를 해석할 수 있다.

원격 근거:
- [이식 설명](https://github.com/Aieoreun/first_penguins/blob/3c64079eb1af14170b3bee4578ac6f916bec433e/docs/PEACE_FULL_PORT.md)
- [초기 이식 점수](https://github.com/Aieoreun/first_penguins/blob/3c64079eb1af14170b3bee4578ac6f916bec433e/docs/PEACE_FULL_PORT_RESULTS.md)
- [고정 서버](https://github.com/Aieoreun/first_penguins/blob/3c64079eb1af14170b3bee4578ac6f916bec433e/server.py#L28)
- [병렬 가드](https://github.com/Aieoreun/first_penguins/blob/3c64079eb1af14170b3bee4578ac6f916bec433e/pipelines/kimjaehak/parallel_guard.py#L22)
