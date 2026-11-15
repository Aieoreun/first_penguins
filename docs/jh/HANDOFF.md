# 김재학 실험 인수인계 — 진행 중

2026-11-16 KST. 브랜치 `experiment/kimjaehak`, 기준 커밋 `6ec111d`(승원). 후속 실험을 진행 중이며 아래 수치는 완료된 실행만 포함한다. 원격 push나 공식 비공개 평가 제출은 하지 않았다.

## 현재 선택

반복 확인된 공개 최고 후보는 **`jh_candidate` / 내부 slots 64**다. 공개 400문항 전체 점수 **96.90 / 97.11, 평균 97.005**다. 두 번째 실행은 tutor 97.00 / guard 97.22, 수집 17.36초, 요청 p95 5.983초 / 최대 9.418초였다. 두 실행 모두 HTTP·형식·채점 오류 0, 내부 material parse 실패는 각각 1 / 2요청이다.

공개 category와 tutor evidence·harmful action 라벨로 처음부터 학습한 TF-IDF + LogisticRegression 분류기를 포함한다. **위 점수는 학습 데이터 내 점수이며 비공개 성능이 아니다.** 질문→답 표, reference_answer 조회, 요청 ID 비교, 비공개 요청 저장·학습은 없다. 입력 C=1000 / 확률 기준 0.95, 튜터 섹션 기준 0.98, harmful 세부 action 기준 0.95다. 불확실하면 기존 LLM 경로로 간다.

여기에 간결한 일반 답변, 교재 밖 질문의 짧은 안내, 대화 맥락을 포함한 출력 가드를 결합했다. 출력에는 원래 BM25 top1 섹션을 쓰며 학습 섹션 선택은 튜터에만 적용한다. 통합 전 `jh_concise_fitted`는 95.72 / 96.00(평균 95.86), `jh_learned_sections_tutor`는 95.26 / 95.78(평균 95.52)이었다.

공개 라벨로 학습하지 않는 대안은 `jh_adaptive` / slots 64이다. 전체 **89.72 / 88.19, 평균 88.96**, 수집 54~60초다. `jh_voted` / slots 32는 **89.30 / 88.78, 평균 89.04**지만 수집 89~95초였다. 양쪽 모두 공개 프롬프트 튜닝을 거쳤다.

## 최초 네 가지 비교

| 버전 | 전체 | tutor | guard |
|---|---:|---:|---:|
| 승원 v2 기준선 | 82.20 | 80.12 | 84.28 |
| Peace 입력만 | 77.25 | 73.38 | 81.12 |
| Peace 출력만 | 81.03 | 80.62 | 81.45 |
| Peace 둘 다 | 77.64 | 74.44 | 80.84 |

네 비교군은 동일 HTTP 동시성 16으로 측정했고 그대로 유지한다. Peace 단순 교체는 개선되지 않았다. 후속 후보는 Peace의 실제 요청 의도 판정 기준을 반영하고, 짧은 분류의 선택적 다수결·요청 마감·고신뢰 학습 분류를 조합했다. 전체 기록은 `01_experiments.md`, 완료된 공개 실행 집계는 `results.json`, 최초 네 결과는 `01_four_way.json`에 있다.

## 일반화에 관한 증거와 한계

- 원본 유사 문항을 묶은 category 분류기 OOF 정확도는 74.25%다. 기준 0.95의 OOF 예측을 기존 adaptive 결과에 결합하면 guard F1 0.9000→0.8908, tutor action 0.875→0.890으로 상충한다. `learned_cv.json` 참조.
- 섹션 분류기 OOF 단독 적중은 47/119로 BM25 108/119보다 낮다. 기준 0.98을 통과한 OOF 질문은 없어 결합 결과는 BM25와 같았다. `section_cv.json` 참조.
- 제공 모델로 만든 공개 표현 변형 388건의 실제 HTTP 입력 진단에서는 adaptive F1 0.9038→learned_gate95 0.9694, 과잉 차단 15→2로 개선됐다. 라벨은 학습한 원본에서 상속했으므로 독립 holdout은 아니다. 변형의 의미가 달라졌을 가능성도 있다. `transfer_results.json` 참조.
- 변형으로 추가 학습하는 증강은 grouped OOF를 악화시켜 미채택이다. 생성 데이터는 배포 분류기 학습에 쓰지 않는다. `augmented_cv.json` 참조.
- KB만으로 생성한 검색 질문 261개는 보조 색인 후보다. 전체 실행은 비학습 89.37, 학습 결합 95.13으로 기존 반복 최고를 넘었다고 판단할 수 없다. `kb_expansion_grid.json` 참조.

## 실행

프로젝트 루트의 기존 `.venv`와 공통 `.env`를 사용한다. 학습 후보에는 `requirements-jh-research.txt`가 필요하다.

```bash
uv pip install --python .venv/bin/python -r requirements-jh-research.txt
.venv/bin/python experiments/jh_server.py --variant jh_candidate --request-slots 64 --port 8000
.venv/bin/python experiments/jh_compare.py --name candidate_NEW --variants jh_candidate --workers 64 --request-slots 64 --judge-workers 8
.venv/bin/python -m unittest discover -s tests -q
```

실험 이름은 새 이름을 쓴다. `--workers`는 HTTP 클라이언트 동시성, `--request-slots`는 서버 내부 동시성이다. 내부 기본값 32와 구별해야 한다. 런박스 외부 연결이 필요할 때만 launcher에 `--host 0.0.0.0`을 추가한다. 기본 `server.py`는 변경하지 않았으므로 후보 선택과 `.env` 로딩에는 실험 launcher를 쓴다.

`.env` 변수는 `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY`, `TEAM_TOKEN`이다. 값은 출력·커밋하지 않았다. 같은 API 주소의 허용 임베딩 식별자 두 개가 404여서 모든 비교는 BM25-only다. 임의 외부 API는 사용하지 않았다. 비스트리밍 completion 한도는 2000이며 원본 판정기는 streaming 4000 그대로다.

## 기록·복구

응답·manifest·runtime·판정 체크포인트는 gitignore된 `experiments/.cache/jh_<이름>/<variant>_1/`에 남아 있다. 공개 자료와 제공 모델의 공개·KB 파생 자료만 저장한다. 서버는 들어오는 요청 본문을 저장하지 않는다.

```bash
.venv/bin/python experiments/jh_resume_score.py experiments/.cache/jh_RUN/VARIANT_1
.venv/bin/python experiments/jh_resume_transfer.py experiments/.cache/jh_TRANSFER/VARIANT_1
.venv/bin/python experiments/jh_summarize.py
```

첫 명령은 원래 공개 400문항, 두 번째는 공개 파생 전체 진단용이다. 평가기·모델·데이터 해시와 수집 완료를 검증하며 완료된 결과를 덮어쓰지 않는다. 채점 중단 재개는 같은 호출의 체크포인트만 사용한다. 좋은 점수가 나올 때까지 같은 응답을 재채점하지 않는다.

초기 `full_voted_64`는 executor 문제로 실제 HTTP 16이었다. `305ea02`에서 수정했다. manifest의 `http_transport=explicit_worker_thread_pool` 이후 64 실행은 실제 HTTP 64다. 기존 HTTP 16 비교에는 영향이 없다. `--no-judge`, guard-only, 출력 구성요소 진단, 공개 파생 점수를 전체 원본 400점수와 섞지 않는다.

## 현재 진행·다음 판단

학습 강도 C=3000 / 입력 기준 0.95의 `jh_stronger95`는 공개 전체 96.89였다. 같은 C / 기준 0.98은 94.70이었다. 전자는 공개 파생 입력에서 고신뢰 적용 범위를 늘렸지만 원본 grouped OOF guard F1은 기존보다 낮아졌다. `gate_transfer_grid.json`, `stronger_cv.json`의 상충 결과를 확인해야 한다. 기본 통합 후보는 C=1000을 유지한다.

섹션 기준만 0.95로 낮춘 `jh_learned_sections_tutor95`는 원본 공개 전체 95.20이었다. 공개 파생 388건 전체에서는 adaptive 87.86 대비 93.01로 개선됐다. 모든 HTTP·형식·채점 오류는 0이다. 통합 후보는 같은 파생 388건에서 92.42(tutor 93.10 / guard 91.74)로 앞선 93.01보다 낮았다. HTTP·내부 단계·형식·채점 오류는 모두 0이었다. 공개 원본 최고와 파생 최고가 다르다.

교재만으로 생성·검토하고 응답·점수를 보기 전에 품질 제외한 55문항에서는 adaptive 90.00 / candidate 90.00 / candidate_stronger 87.73 tutor였다. 모든 오류 0이다. 학습에 넣지 않으며 약한 사실·단일 섹션 인용 라벨이라는 한계가 있다. `kb_probe_results.json`에 검토·제외·데이터 해시와 결과를 보존했다. 공개 원본 400 점수와 섞지 않는다.

통합 C=3000의 공개 반복은 97.31 / 96.19, 평균 96.75로 현재 후보보다 낮았다. OOF·교재 진단도 불리해 C=1000을 유지한다. 짧은 정책 문구의 36건 구성요소 비교는 1회 개선, 1회 동률이었다. 문구·학습 안내·일반 답변 추론량 변경을 각각 전체 평가 중이다. medium 답변 첫 실행의 JSON 모드 설정 오류는 채점에서 제외했고 `1173ff2`로 수정 후 새 이름으로 재수집한다.

최근 전체 테스트 153개 통과. 이후 조건부 교재 선택·맥락 거절의 관련 검증도 통과했다. 주최측 평가기, SPEC, public, KB, core와 기본 server는 그대로 유지한다. 사용량은 마지막 조회 기준 **19% 남음**이다. 15% 이하를 확인하면 새 실험을 중단하고 진행 중 채점·결과·인수인계를 마무리해 커밋한다. 계정 사용량은 다른 작업과 공유된다. 현재 문서는 종료 보고가 아니다.

출력 세 번 합의는 고정 입력 두 번 모두 개선됐으며 첫 전체 97.16, 두 번째 채점 중이다. 불확실한 normal·pii 질문만 세 교재로 답하는 후보는 별도 55문항에서 tutor 93.18(기존 90.00)이었다. 원본·공개 파생 전체 비교 중이다. 모든 결과는 `01_experiments.md` 15차와 각 JSON에 기록한다.
