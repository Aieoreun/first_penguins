# 김재학 실험 인수인계 — 진행 중

2026-11-16 KST. 브랜치 `experiment/kimjaehak`, 기준 `6ec111d`(승원 v2). 원격 push·공식 비공개 제출은 하지 않았다. 새 실험을 진행 중이며 아래 점수는 완료된 실행만 포함한다.

## 현재 선택

**공개 반복 평균 최고: `jh_learned_output5` / 내부 slots 64.** 전체 98.39 / 97.44, 평균 **97.915**. tutor 97.88 / 96.38, guard 98.90 / 98.51이다. 두 실행 모두 HTTP·형식·채점 오류 0. 수집 19.37 / 17.74초, p95 6.908 / 6.467초, 최대 11.493 / 14.511초. 내부 단계 실패는 4 / 1요청이다.

공개 category·tutor evidence·harmful action·guard unsupported 구절 라벨로 처음부터 학습한 분류기와 추출기를 포함한다. **이 점수는 학습 데이터 내 점수이며 비공개 성능이 아니다.** 질문→답 표, reference_answer 조회, 요청 ID 비교, 비공개 요청 저장·학습은 없다.

구성은 다음과 같다.

- category TF-IDF + LogisticRegression C=1000 / 적용 확률 0.95, tutor 섹션 C=1000 / 0.98, harmful action C=1000 / 0.95. 불확실하면 LLM 분류로 복귀한다.
- 일반 질문은 간결한 교재 답변, 교재 밖 질문은 짧은 확인 불가 안내, 유해 요청·발언은 대상에 맞춘 거절·예방 학습 안내. 거절 생성 실패는 고정 문구로 복귀한다.
- 출력 근거는 원래 BM25 top1 + 대화 맥락. 5회 병렬 판정 중 서로 가장 일치하는 실제 판정 하나를 선택한다.
- 문자 구절 추출기 C=100 / 모든 문자 최소 신뢰도 0.95. LLM이 찾은 오류와 겹칠 때만 경계를 보정하며 grounded·safe는 바꾸지 않는다.
- safe는 승원 v4의 입력 판정 기반 규칙을 유지한다(allow이면 true, block이면 false). grounded는 별도 판정이다.
- 요청 전체 26초, 준비 14초, 내부 동시성은 launcher에서 64로 지정한다.

출력 구절을 학습하지 않는 비교 후보는 `jh_context_policy`로 전체 97.56 / 97.20, 평균 **97.38**이다. 이전 `jh_candidate`는 네 번 평균 **96.78**이다(초기 두 번만의 97.005와 구분). 공개 라벨을 직접 학습하지 않는 `jh_adaptive`는 89.72 / 88.19, 평균 88.96이었으며 공개 프롬프트 튜닝은 거쳤다.

## 최초 요청한 네 버전

| 버전 | 전체 | tutor | guard |
|---|---:|---:|---:|
| 승원 v2 기준선 | 82.20 | 80.12 | 84.28 |
| Peace 입력만 | 77.25 | 73.38 | 81.12 |
| Peace 출력만 | 81.03 | 80.62 | 81.45 |
| Peace 둘 다 | 77.64 | 74.44 | 80.84 |

네 버전은 같은 HTTP 동시성 16으로 비교했다. Peace 단순 교체는 개선되지 않아 실제 요청 의도 경계, 선택적 다수결, 요청 마감, 고신뢰 학습 분류와 생성·출력 방식을 차례로 비교했다. 원래 네 버전은 그대로 선택할 수 있다. `01_four_way.json` 참조.

## 별도 진단에서 확인된 한계

- category 분류기 원본 그룹 OOF 정확도 74.25%. 기준 0.95를 LLM 입력 결과와 결합하면 guard F1 0.9000→0.8908, tutor action 0.875→0.890으로 상충했다. C=3000도 OOF는 더 좋지 않았다.
- 섹션 분류기 단독 OOF는 47/119로 BM25 108/119보다 낮다. 기준 0.98에서는 OOF 적용이 없어 원래 검색을 유지한다.
- 새 표현의 공개 사용자 발화 388건 전체: adaptive 87.86, learned_sections_tutor95 93.01, candidate 92.42, uncertain_sources 93.58. 학습한 부모에서 상속한 약한 라벨이며 독립 holdout이 아니다. 마지막 후보는 전체가 높아도 인용·facts는 candidate보다 낮아 기본에 넣지 않았다.
- KB만으로 생성·검토하고 점수 확인 전에 품질 제외한 55문항 tutor: adaptive 90.00, candidate 90.00, candidate_stronger 87.73, uncertain_sources 93.18, resilient_sources 92.27, six_sources 90.45. 학습에는 넣지 않는다. 단일 섹션 인용·사실 라벨이 불완전할 수 있고 동일 교재·제공 모델에 의존한다.
- 구절 모델 단독 OOF span F1은 0.4122로 낮다. 신뢰도만으로 판정을 교체하지 않는다. LLM과 겹치는 오류만 고신뢰 보정하면 OOF 적용 0건으로 원래 0.8644 그대로, 학습 내 0.9700이다. 유사도 0.75로 합쳐진 후보 답변 쌍은 없어 실제 fold 그룹은 86개였다.
- 후보 답변 자체를 바꾼 새 표현 60건은 학습에 넣지 않았다. 같은 LLM 출력에 구절 보정만 적용한 통제 비교에서는 실제 변경 0건으로 개선이 없었다. 따라서 공개 출력 보정의 이득을 새 답변 성능으로 해석하지 않는다.

## 실행

프로젝트 루트의 기존 `.venv`와 공통 `.env`를 사용한다. 학습 후보에는 `requirements-jh-research.txt`가 필요하다.

```bash
uv pip install --python .venv/bin/python -r requirements-jh-research.txt
.venv/bin/python experiments/jh_server.py --variant jh_learned_output5 --request-slots 64 --port 8000
.venv/bin/python experiments/jh_compare.py --name NEW_COMPARISON --variants jh_learned_output5 --repeat 2 --workers 64 --request-slots 64 --judge-workers 8
.venv/bin/python -m unittest discover -s tests -q
```

실험 이름은 새 이름을 쓴다. `--workers`는 HTTP 클라이언트 동시성, `--request-slots`는 서버 내부 동시성이다. 기본 server는 변경하지 않았고 기본 파이프라인도 v1 그대로라, 후보와 `.env` 선택에는 실험 launcher를 사용한다. 런박스 외부 연결에는 `--host 0.0.0.0`을 추가한다.

`.env`는 `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY`, `TEAM_TOKEN`을 사용한다. 값은 출력·커밋하지 않았다. 같은 API 주소의 허용 임베딩 식별자 두 개가 404여서 모든 비교는 BM25-only다. 임의 외부 API·모델 파일은 사용하지 않았다.

## 기록과 복구

`01_experiments.md`에 모든 차수의 가설·채택/제외 이유를 남겼다. `results.json`은 완료된 원본 공개 실행과 구성요소 진단을 scope별로 집계한다. 소규모 smoke, 판정 생략, guard-only, 구성요소, 공개 파생 점수를 원본 400 전체와 섞지 않는다. 별도 자료는 `transfer_results.json`, `kb_probe_results.json`, `novel_guard_results.json`, 각 `*_cv.json`을 본다.

응답·manifest·runtime·판정 체크포인트는 gitignore된 `experiments/.cache/jh_<실험>/`에 있다. 공개 자료와 제공 모델이 만든 파생 자료만 저장했다. 서버는 들어오는 요청·답변 본문을 저장하지 않는다.

```bash
.venv/bin/python experiments/jh_resume_score.py experiments/.cache/jh_RUN/VARIANT_1
.venv/bin/python experiments/jh_resume_transfer.py experiments/.cache/jh_TRANSFER/VARIANT_1
.venv/bin/python experiments/jh_summarize.py
```

첫 명령은 원본 공개 전체, 두 번째는 공개 파생 전체·교재 tutor 진단의 중단 채점 재개다. 평가기·모델·데이터 해시와 수집 완료를 확인하고 완료 결과는 덮어쓰지 않는다. 같은 실행·같은 호출의 체크포인트만 재사용하며 좋은 점수가 나올 때까지 같은 응답을 재채점하지 않는다. `jh_tutor_components.py`는 추가 LLM 호출 없이 저장된 판정과 원래 문항 점수의 일치를 검증한다.

초기 `full_voted_64`는 executor 문제로 실제 HTTP 16이었다. `305ea02` 이후 `http_transport=explicit_worker_thread_pool`의 64 실행은 실제 HTTP 64다. medium 답변 첫 JSON 모드 오류 실행은 채점하지 않고 제외했다. 잘못 생성된 첫 교재 초안과 70건 예비 수집도 평가에서 제외했다. 모두 이유를 캐시에 보존했다.

## 현재 진행

C=3000을 구절 보정 조합에 적용한 후보와 /guard에서 불필요한 harmful request/speech 구분을 생략한 후보를 각 두 번 수집해 채점 중이다. 비교 조건과 사용자 상황을 구분한 추가 출력 원칙은 첫 고정 입력 비교에서 동률이었다. 빠른 입력 처리로 확보한 시간 안에서 medium 출력 1회 / 4000토큰 / 20초를 별도 진단 중이다.

최근 전체 테스트 161개 통과 후 가드 전용 범주 경로와 공통 출력 호출의 관련 테스트도 통과했다. 주최측 평가기·SPEC·public·KB·core·기본 server는 그대로다. 사용량 마지막 조회 **17% 남음**. 15% 이하를 확인하면 새 실험을 멈추고 진행 중 채점, 코드·결과·인수인계·커밋을 마무리한다. 계정 사용량은 다른 작업과 공유된다. 이 문서는 아직 종료 보고가 아니다.
