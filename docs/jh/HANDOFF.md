# 김재학 실험 인수인계 — 진행 중

2026-11-16 KST. 브랜치 `experiment/kimjaehak`. 아직 후속 실험 중이며 최종 선택 문서는 갱신 예정이다.

## 현재 확인한 최고 전체 점수

현재 가장 유망한 후보는 **jh_voted**다. 실제 HTTP 동시성 64, 공개 400문항 전체 평가를 두 번 반복해 **89.30 / 88.78 (평균 89.04)**을 기록했다. 각각 tutor 86.50 / 87.00, guard 92.10 / 90.57이다. 같은 실제 동시성의 v4는 85.80이었다. 원본 평가기 및 모델 판정을 사용한 로컬 점수이며 비공개 공식 점수가 아니다. 최초 v2 기준선은 82.20이다. 후보별 결과는 `01_experiments.md`, 집계는 `results.json`, 초기 네 버전의 고정 기록은 `01_four_way.json`에 있다.

초기 네 버전은 모두 유지한다. Peace 원본 단순 교체는 77.25 / 81.03 / 77.64로 기준선을 넘지 못했다. `.env`는 출력·커밋하지 않았고, 주최측 평가 코드·SPEC·KB·public은 변경하지 않았다.

## 재현

프로젝트 루트에서 실행한다. `.venv`와 사용자 공통 `.env`를 그대로 사용한다.

```bash
.venv/bin/python experiments/jh_server.py --variant jh_voted --port 8000
.venv/bin/python experiments/jh_compare.py --name voted_repeat_NEW --variants jh_voted --workers 64 --judge-workers 8
.venv/bin/python experiments/jh_compare.py --name v4_repeat_NEW --variants v4 --workers 16 --judge-workers 8
.venv/bin/python experiments/jh_compare.py --name load64_NEW --variants v4 --workers 64 --no-judge
.venv/bin/python -m unittest discover -s tests -q
```

`--name`은 중복되면 중단한다. 외부 런박스에 열 때만 서버에 `--host 0.0.0.0`을 추가한다. 기본 서버 `server.py`를 직접 실행하면 `.env` 자동 로드나 후보 선택이 적용되지 않으니 실험 launcher를 사용한다.

`.env` 사용 변수: `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY`, `TEAM_TOKEN`. 현재 임베딩은 사용할 수 없어 모든 비교는 BM25-only다. 같은 API 주소의 허용 임베딩 식별자 두 개는 404였고 임의 외부 주소를 사용하지 않았다.

검증 환경: Python 3.12.13, openai 3.51.0, httpx2 2.16.0, fastapi 0.163.1, uvicorn 0.54.0, pydantic 2.14.3, python-dotenv 1.2.4. 설치 입력은 `requirements-jh.txt`다.

프록시의 비스트리밍 completion 한도는 2000이다. 원본 판정기는 streaming 4000을 사용하며 그대로 둔다.

## 기록과 복구

- `experiments/.cache/jh_<실험명>/<variant>_1/`에 공개 응답, 원본 점수, 메타데이터, 서버 단계 오류, 판정 체크포인트가 있다. gitignore된 로컬 자료다.
- 공개 응답 수집이 끝나고 채점만 중단됐으면 `experiments/jh_resume_score.py <실행 폴더>`로 재개한다. 평가기 해시·모델·ID·문항 수를 확인하며 완성된 결과를 덮어쓰지 않는다.
- `--endpoints guard`는 200문항 가드 진단이다. 그 실행의 total을 전체 점수로 인용하지 않는다.
- `experiments/jh_output_replay.py`는 기존 **자체 입력 예측**을 고정한 출력 구성요소 진단이다. 정답 라벨을 파이프라인 상태로 쓰지 않는다. HTTP·부하·전체 점수 검증을 대체하지 않는다.
- `5426530` 이후 manifest에 실제 상속 설정과 프롬프트 해시를 기록한다. 그 이전 v4 manifest의 `pinned_v2` 표기는 실제 클래스 override를 나타내지 않는다. 상세 정정은 실험 기록에 있다.

## 남은 검증

짧은 분류만 세 번 투표하는 `jh_voted`는 무거운 material을 한 번만 부르고 내부 동시 요청 32개, 대기 포함 26초 제한을 쓴다. 실제 동시성 64의 두 실행은 HTTP·형식·채점 오류 0, 최대 지연 24.897 / 25.309초였다. material timeout/parse는 각각 9 / 8요청에 발생했으나 정상 형식 fallback을 반환했다.

첫 voted 64 부하 실험의 87.19는 클라이언트 기본 풀이 16개라 실제 64개를 보내지 못했다. `305ea02`에서 수정했고 기록을 보존한다. 기존 workers 16 비교에는 영향이 없다.

`jh_reasoned`·`jh_explicit`·`jh_diverse`·`jh_multiground`·`jh_material_review`·`jh_block5`·`jh_unanimous`·`jh_token_output`는 현재까지 미채택이다. 세 섹션 중 근거를 고르게 한 multiground는 전체 87.61, 전원 일치 차단은 전체 86.51이었다. 세부 실패·점수는 실험 기록에 모두 남긴다.

`jh_adaptive`는 결과를 바꿀 수 없는 세 번째 투표를 생략한다. 전체 88.59 (87.38 / 89.79), 내부 실패 15요청이었다. 지연 변동을 줄여 비교하기 위해 같은 시점의 부하 비교를 진행한다. `jh_adaptive_thin`은 교재 판정에만 검색 조각 5개를 제공하는 후보다. `jh_reconsider`와 `jh_medium_output`도 아직 진단을 마치지 않았다.

현재 `jh_batch_adaptive_s32/jh_adaptive_1` 응답 수집 중이다. `--collect-only`로 수집한 폴더는 채점을 하지 않은 상태이므로, 수집 완료 후 아래 명령으로 원본 평가기를 실행해야 한다. 이후 내부 동시성 64와 짧은 교재 문맥 조건의 수집을 연속 비교할 계획이다.

```bash
.venv/bin/python experiments/jh_resume_score.py experiments/.cache/jh_batch_adaptive_s32/jh_adaptive_1
```

`--request-slots`는 서버 내부 동시성이다. `--workers`는 HTTP 클라이언트 동시성이며 서로 다르다. 기본 추천 명령은 검증된 voted / 내부 32 그대로다. 아직 점수가 없는 후보를 최종 선택으로 취급하지 않는다.

출력 구성요소 비교와 공개 예시 참고, 자체 분류기 교차검증은 유의미한 우위를 확인하지 못했다. `jh_compact`의 짧은 오류 구절 추출은 같은 입력의 v4 88.48 대비 88.87이었으나 반복 확인 전이다. 자체 분류기는 학습 문항 100%까지 적합되더라도 grouped OOF 약 74%였으며, 확신이 높은 경우에만 사용해도 기존 LLM과 결합한 입력 점수가 개선되지 않아 배포하지 않았다. 선택 의존성 `requirements-jh-research.txt`와 `jh_classical_cv.py`, `jh_gate_cv.py`로 재현한다.

새 후보도 전체 400문항, 반복 실행, 실제 HTTP 동시성 64에서 확인해야 한다. 1점 안팎의 차이는 모델 변동일 수 있다. 공식 비공개 평가는 제출하지 않았다. 현재 테스트 127개 통과.

Codex 잔여 사용량 15% 이하를 확인하면 새 실험을 시작하지 않고 진행 중 결과·코드·이 문서를 정리해 커밋한다. 마지막 조회는 잔여 28%였다. 원격 push는 하지 않았다.
