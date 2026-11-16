# 김재학 실험 — 승원 v2 + Peace 가드

진행 상황·최고 관측 점수·재개 방법은 [HANDOFF.md](HANDOFF.md), 모든 후속 가설과 채택 여부는 [01_experiments.md](01_experiments.md), 집계 데이터는 [results.json](results.json)에 기록한다. 아래 네 비교군은 후속 실험과 별도로 그대로 유지한다.

2026-11-16 후속 라운드는 [ROUND2.md](ROUND2.md), 최신 튜터·가드 분리와 후속 비교는
[TUTOR_SPLIT.md](TUTOR_SPLIT.md), 해당 집계는 [tutor_split_results.json](tutor_split_results.json)을 본다.
현재 사용자 지시는 **잔여 사용량 60%에서 새 실험 중단**이다. 이전 50%를 사용자가 정정했으며,
아래 15%는 최초 라운드의 기준이다. 진행 중인 실험은 [TUTOR_CONTINUOUS.md](TUTOR_CONTINUOUS.md)를 본다.

2026-11-17 02:10 KST에 잔여60%를 확인해 새 실험을 중단했다. 현재 우선 실험 후보는
`jh_r3_selected_focused`(guard ContextFast 유지, embedding env, slots64)다.
공개 원본400개 대조 두 쌍의 평균 전체97.805→98.540, tutor96.310→98.125였다.
실제 비공개 제출 점수와 구분하며 실행 명령·후속 판단은 [HANDOFF.md](HANDOFF.md)를 우선한다.

후속 실제 제출은 안내한1→2→3 순서 기준 focused83.99, ContextFast82.65,
selected_extract83.53이었다. 최종 추천은 focused이며 서로 다른 seed의 한 번 관측이다.
시간 비교·해석의 한계·다음 개선 계획은 [실제 제출 비교](PRIVATE_SUBMISSIONS_20261117.md)에 기록했다.

이후 사용자가1번 focused를 재제출해 **84.68 / tutor81.58 / guard87.78 / 67.089초**를 보고했다.
focused 두 번 전체 평균84.335, 공식 오류는 모두0이다. 현재1번과 최신 제출 유지 권고이며
표본이 다른 소수 관측이므로 안정적인 후보 우위나 속도 차이는 확정하지 않는다.

## 출처와 범위

- 기준 코드: `experiment/seungwon`의 `6ec111d` (v2 사용).
- Peace 원본: `export/peace`의 `095cd4a`에 있는 `guard_classifier.py`, `guard_output.py`, `guard_schemas.py`.
- `pipelines/kimjaehak/peace/`는 원본을 복사하고 classifier의 import만 상대 경로로 바꿨다.
- 검색·청킹·답변 생성·기존 파이프라인과 주최 측 `grader/`, `harness/score_local.py`는 변경하지 않는다.

## 네 비교군

| 이름 | 입력 | 출력 |
|---|---|---|
| `jh_baseline` | 원래 v2 | 원래 v2 |
| `jh_peace_input` | Peace 단일 분류 | 원래 v2 |
| `jh_peace_output` | 원래 v2 | Peace 근거·안전 병렬 검사 |
| `jh_peace_both` | Peace 단일 분류 | Peace 근거·안전 병렬 검사 |

모든 비교군은 `config.py`의 동일한 v2 설정을 사용한다. 환경변수 잔존 영향을 없애고 프로세스를 새로 띄운다. 기준선은 원래 v2 클래스다.

Peace의 decision은 category에서 재계산하지 않고 그대로 보존한다. 튜터 action은 승원의 기존 범주 표로 정한다. Peace가 `harmful_content`만 반환하므로 승원의 기존 별칭(`harmful_request`/refuse)을 사용한다. 유해 발언의 redirect를 구분하지 못하는 것은 초기 비교군의 알려진 한계다. 교재를 받지 않는 Peace 입력 분류는 v2의 교재 확인을 대체하며 out_of_material 성능을 따로 측정한다.

출력 검사에는 v2와 같은 근거 범위(선택 섹션 전체 또는 상위 5개 조각)를 전달한다. 근거·안전 단계는 독립적으로 실행한다. 한 단계가 실패하면 성공한 결과는 보존한다. 실패한 근거 검사는 grounded=true/빈 구간, 실패한 안전 검사는 입력 allow 여부를 기본값으로 사용하며 단계 실패를 메타데이터로 기록한다. HTTP 503은 가져오지 않는다.

## 실행

Python 3.12 가상환경과 `requirements-jh.txt`를 사용한다. `.env`의 값은 출력하거나 커밋하지 않는다. launcher가 `.env`를 직접 읽는다.

```bash
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements-jh.txt
.venv/bin/python experiments/jh_server.py --variant jh_peace_both --port 8000
.venv/bin/python experiments/jh_compare.py --name four_way --workers 16
.venv/bin/python experiments/jh_compare.py --name load64 --workers 64 --no-judge
.venv/bin/python -m unittest discover -s tests -q
```

런박스에서 외부에 열 때는 서버 명령에 `--host 0.0.0.0`을 추가한다. 파이프라인은 실행 인자로 정하며 공통 `.env`를 실험마다 바꾸지 않는다.

## 평가와 기록

공개 400문항을 동일 순서로 HTTP 호출하고 모든 응답 수집을 마친 뒤 **원본 `harness.score_local.score_local`과 `grader.judge.make_judge`**로 채점한다. 채점 로직·프롬프트·가중치는 그대로다. 평가기 파일 해시를 실행마다 기록하고 종료 시 일치 여부를 검사한다. 로컬 판정 모델은 실제 비공개 판정 모델과 다르며 공식 제출 점수를 대체하지 않는다.

`--limit N`은 endpoint별 N개 smoke test다. `--no-judge`는 부하/형식 점검이며 완전한 튜터 점수가 아니다. `experiments/.cache/jh_<이름>/`에 실행별 manifest, 공개 응답 JSONL, 서버 메타데이터 로그, 원본 점수, 진단, 요약을 남긴다. 부분 수집은 줄마다 저장하고 누락된 응답으로 전체 점수를 계산하지 않는다. **이 개발 도구는 저장소의 공개 문항만 읽는다. 비공개 채점 요청·응답을 저장하는 서버 코드가 아니다.**

코드를 먼저 커밋하고 해당 커밋으로 실험한 다음 집계 결과와 분석을 별도 문서 커밋으로 남긴다. 현재 서버는 임베딩 주소·모델이 없으면 BM25만 사용한다. 초기 환경에서 채팅 연결은 성공했고 동일 주소의 두 허용 모델 식별자로 임베딩 호출은 404였다. 따라서 초기 비교는 모두 BM25-only이며 임베딩을 사용한 과거 점수와 직접 비교하지 않는다.

Codex 사용량은 실험 전후 확인한다. 잔여 15% 이하이면 새 실험을 시작하지 않고 결과·코드·HANDOFF를 정리해 커밋한다. 계정 사용량은 다른 작업과 공유된다.
