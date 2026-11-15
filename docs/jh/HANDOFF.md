# 김재학 실험 인수인계 — 진행 중

2026-11-16 KST. 브랜치 `experiment/kimjaehak`. 아직 후속 실험 중이며 최종 선택 문서는 갱신 예정이다.

## 현재 확인한 최고 전체 점수

원래 **v4 86.28** (tutor 83.38 / guard 89.17). 공개 400문항, 동시 HTTP 16, 원본 평가기 및 모델 판정 사용. 비공개 공식 점수가 아니다. 최초 v2 기준선은 82.20이다. 후보별 결과는 `01_experiments.md`, 초기 네 버전의 기계 판독 결과는 `01_four_way.json`에 있다.

초기 네 버전은 모두 유지한다. Peace 원본 단순 교체는 77.25 / 81.03 / 77.64로 기준선을 넘지 못했다. `.env`는 출력·커밋하지 않았고, 주최측 평가 코드·SPEC·KB·public은 변경하지 않았다.

## 재현

프로젝트 루트에서 실행한다. `.venv`와 사용자 공통 `.env`를 그대로 사용한다.

```bash
.venv/bin/python experiments/jh_server.py --variant v4 --port 8000
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

v4의 명확한 거절·학습 발판·입력 경계를 보완한 `jh_calibrated`를 평가하고 있다. 출력은 원래 v4 근거 검사 + 범주별 안전 검사인 `jh_v4_safety`를 기존 입력 예측을 고정해 비교할 예정이다. 어느 새 후보도 아직 최종 채택하지 않았다.

유망한 후보는 전체 400문항, 반복 실행, 실제 HTTP 동시성 64에서 확인해야 한다. 현재 최고 점수는 단일 실행이며, 1점 안팎의 차이는 모델 변동일 수 있다. 공식 비공개 평가는 제출하지 않았다.

Codex 잔여 사용량 15% 이하를 확인하면 새 실험을 시작하지 않고 진행 중 결과·코드·이 문서를 정리해 커밋한다. 마지막 조회는 잔여 32%였다. 원격 push는 하지 않았다.
