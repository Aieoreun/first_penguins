# 재학님 전체 하네스 이식 — 학습형 모델 제외

## 보존

- 이전 코드 백업: `backup/peace-before-jh-full-20261117`, `93814e5955119025f9b5f919b51725d50a611004`
- 참고 코드: `origin/experiment/kimjaehak`, `b19984475c2dd0c8fa54004bedf95f206ef55ed8`
- 새 브랜치: `experiment/peace-jh-full-no-trained`
- 실행 variant: `jh_r4_material_budget`

원본 전체 트리에서 시작했다. 입력/유해세부/섹션/출력구간 학습 모델의 학습·추론만 비활성화했다. 기존 프롬프트, BM25+임베딩 검색, 입력 투표, 출력 5회 합의, 실패 기본값, 튜터 섹션 선택·원문 인용, 시간 예산을 그대로 보존했다. 학습 기능의 비활성화로 학습 모델이 선택하던 경로는 기존 LLM 경로로 진행한다.

## 실행

프로젝트 `.env`를 설정하고 원본 실행기를 사용한다. 원본 실행기는 `requirements-jh.txt`의 python-dotenv가 필요하다. 이 실험에서는 설치 대신 이미 존재하는 로컬 캐시 모듈을 PYTHONPATH로 연결했다. API 키·토큰·색인은 Git에 포함하지 않는다.

```sh
python experiments/jh_server.py --variant jh_r4_material_budget --embedding-mode env --request-slots 64 --host 0.0.0.0 --port 8000
```

`PIPELINE` 미지정 상태의 원본 서버 기본값은 v1이다. 위 variant를 명시해야 이식한 하네스로 실행된다.

## 평가

```sh
python experiments/peace_full_http.py --variant jh_r4_material_budget --endpoints tutor guard --repeat 3 --workers 64 --request-slots 64 --judge-workers 8 --output-dir experiments/.cache/새결과폴더
```

튜터 200개씩 3회, 가드 200개씩 3회 순차 평가. 각 실행마다 새 HTTP 서버. 클라이언트 전송~본문 수신 완료 30초 제한. 튜터는 원본 공식 로컬 채점기의 LLM judge를 실제 호출한다. 로그의 내부 단계 실패는 HTTP 성공과 별개로 집계한다. 원본의 정상 형식 fallback 때문에 HTTP 오류 0이 모든 내부 호출 성공을 의미하지 않는다. 공개 평가 결과는 비공개 공식 점수가 아니다.

## 현재 환경에서 추가 패키지 설치 없이 실행

```sh
set -a
source .env
set +a
python run_jh.py
```

`run_jh.py`는 원본 실험 실행기와 동일하게 variant와 64슬롯을 선택하고 동일한 server:app을 실행한다. dotenv 자동 로딩 대신 셸에서 환경변수를 읽는다. 판정 로직은 추가하지 않는다.
