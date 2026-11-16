# 튜터 개선 4차 기록 — 잔여40%에서 종료, 48회 완료

**현재 추천은 `jh_r3_selected_focused`를 유지한다.** 사용자가 보고한 실제 비공개 제출 점수는 전체 **83.99 / 84.68**이며, 최신 관측값은 **84.68**이다. `material_budget`의 **98.18**은 공개 파생 자료에서 얻은 개발 단회 점수다. 실제 비공개 점수와 비교하거나, 후보의 성능 우위가 확인된 것으로 해석하지 않는다.

이 문서는 **완료된 HTTP 실행 48개**까지 정리한다. R4-13까지 비교를 완료하고 잔여40% 기준으로 종료했다. 아직 완료되지 않은 점수는 기재하지 않는다.

- 실행별 원자료는 [round4_results.json](round4_results.json)의 `http` 전체 48개 레코드다. 아래 표는 그 요약이며 전체 레코드를 대체하지 않는다.
- 운영 상태와 중단 기준은 [ROUND4_STATE.json](ROUND4_STATE.json)에 기록한다. 완료된 실행과 다음 비교 계획을 구분한다.

## 기준선과 운영 제약

2026-11-17 사용자 지시로 실험을 재개했다. 재개 당시 계정 사용량은 used 43% / remaining 57%였으며, **remaining 40%에서 새 실험을 중단**한다. 이전의 remaining 60% 중단 기록은 역사로 보존하고 로컬 중단 표식만 해제했다. 목표는 정직한 실제 제출 점수 향상이다. 100점을 지향하지만 달성이나 재현을 보장하지 않는다.

| 항목 | 고정 기준 |
|---|---|
| 추천 기준선 | `jh_r3_selected_focused` |
| 실행 설정 | embedding `env`, request slots 64 |
| 실제 비공개 관측 1 | 전체 83.99 / tutor 79.17 / guard 88.81 |
| 실제 비공개 관측 2 | 전체 84.68 / tutor 81.58 / guard 87.78 |
| 보호 대상 | guard ContextFast, 주최측 평가기, public, KB, 공유 core |

비공개 질문과 응답은 수집하거나 학습·규칙·캐시에 사용하지 않는다. 운영진 제공 모델만 호출하며, 공개·교재 진단의 정답 라벨은 runtime 선택에 전달하지 않는다. 실제 제출 결과는 사용자 보고로만 기록하고 공개 점수와 섞어 평균 내지 않는다.

성공·실패·부정 결과를 모두 보존한다. 수집과 원본 채점을 분리하고, 완료된 판정은 재채점하지 않는다. SPEC 설명상 로컬 판정 모델과 실제 채점 판정 모델은 다르므로, 작은 로컬 판정 변동을 실제 오답 개선으로 과장하지 않는다. 의미적 정확성도 함께 검토한다.

## 완료 실행과 코드 이력

별도로 전체/guard를 표시한 공개400 비교 외의 수치는 **개발용 tutor-only 점수**다. guard를 수집하지 않은 실행의 aggregate total은 tutor 점수의 절반이므로 비교에 사용하지 않는다. 수집 시간은 로컬 부하 측정이며 실제 제출 약 600개 요청의 duration과 직접 비교할 수 없다.

| 단계 | 새 HTTP 실행 | 누적 완료 | 수집 commit | 주요 내용 |
|---|---:|---:|---|---|
| R4-01/02 | 8 | 8 | `ab39eaf` | category 정규화, 마지막 질문 dense 후보 |
| R4-03/04 | 7 | 15 | `d3c35a9` | 조건부 dense, structured BLOCK |
| R4-05 | 4 | 19 | `9501351` | source payload 키 순서 |
| R4-06/07 | 5 | 24 | `aac7100` | 새 관계 진단 16개, material 시간 배분 |
| R4-08 | 4 | 28 | `ad209ae` | material 반복 |
| R4-09 | 4 | 32 | `1f1afd0` | source stream |
| R4-10 | 6 | 38 | `940d02e` | material + stream 조합 |
| R4-11 | 4 | 42 | `5645691` | source ID-only |
| R4-12 | 4 | 46 | `6448ed0` | 실습 구분 지침 |
| R4-13 | 2 | 48 | `a721b72` | source minimal 추론 |

각 수집 시점은 모두 clean 상태였다. 완료된 48개 실행의 **HTTP·형식·원본 judge 오류는 모두 0**이다. 내부 단계 오류와 fallback은 별도로 보존하며 점수에서 제외하지 않는다. 기본 수집 설정은 workers 64 / slots 64 / embedding env, judge workers 32이고, 수집과 판정은 직렬로 진행했다.

문서의 각 “사전 등록”은 **해당 실험을 시작하기 전 당시의 계획과 상태**다. 당시의 “API 시작 전” 표현을 현재 상태로 읽지 않도록, 완료 결과와 함께 정리했다. 후보는 등록·검증·커밋 후 수집하며 기존 추천이나 기본값을 자동 교체하지 않는다.

## 시작 전 실패 감사와 가설 선택

저장 점수 **1,251개**를 복원해 원본과 일치함을 확인했다. 자체 응답이 answer이고 인용이 1개인 **855회** 중 facts/faithful 실패 **33회**를 다음과 같이 분류했다.

| 실패 유형 | 횟수 |
|---|---:|
| 잘못된 출처 | 21 |
| canonical 출처와 다르지만 동일한 사실 | 3 |
| 올바른 교재 사실을 보존한 응답 | 9 |

올바른 교재 사실을 보존한 9회는 모두 **동일 judge payload가 다른 완료 실행에서 통과**했다. 따라서 계산·비교 설명 생성 단계를 먼저 추가하지 않고 출처 후보를 우선 실험했다. 전체 대상에는 이와 별도로 action 오류 **58회**가 있었다. 상세 원본은 [실패 감사](../../experiments/.cache/jh_r4_failure_audit/AUDIT.md)에 보존했다.

초기 가설은 세 가지였다.

1. standalone `ㅋㅋ`·`ㅠㅠ`가 붙은 정상 질문에서 category 고신뢰 gate 이탈을 줄일 수 있는지 확인한다.
2. 올바른 출처의 facts 실패, 잘못된 출처, action 오류, 판정 변동을 구분한다. full section에 답이 명시된 사례에 생성 단계를 무조건 추가하지 않는다.
3. 이전 최대 8개 후보 확대가 실패했으므로 후보 수는 6개로 유지하고, 전체 대화 검색에 가려진 마지막 질문의 의미 검색이 도움이 되는지 확인한다.

## R4-01/02 — category 보호 정규화와 마지막 질문 dense

### 사전 등록한 변경 범위

`jh_r4_normalized_tutor`는 **tutor category 모델 입력만** 보호 정규화한다. 원 대화, LLM payload, 검색, section/harmful 모델, guard는 유지한다. 코드·따옴표·URL 보호와 정서 표현만 있는 입력 보존을 검증했다. 원본 tutor 200개와 normal/PII에 고정 prefix/suffix를 붙인 잡음 222개를 각각 1회 비교했다. 실행 순서는 원본 baseline→candidate, 잡음 candidate→baseline이었다.

잡음은 합성 스트레스 진단이며 실제 분포나 정서 recall 개선을 뜻하지 않는다.

`jh_r4_last_dense`는 기존 출처 후보 첫 2개 다음에 마지막 질문 dense 상위 2개를 넣고 중복을 제거해 최대 6개를 유지한다. 불확실한 normal/PII 출처 선택에서만 호출한다. 제한은 3초이며 남은 시간이 4초 미만이면 생략하고, 오류가 나면 기존 후보 목록으로 돌아간다. topic/followup 각 114개에서 이득이 있으면 primary/alternative 회귀를 확인하는 계획이었다. 정답을 보고 후보를 넣지 않는다.

### 결과

| 자료 | 기준선 tutor / 수집 초 | 후보 tutor / 수집 초 | 점수 차이 |
|---|---:|---:|---:|
| 원본 tutor 200 | 98.12 / 12.75 | 정규화 97.00 / 12.04 | -1.12 |
| normal/PII 잡음 222 | 99.66 / 11.90 | 정규화 98.76 / 12.78 | -0.90 |
| 주제 전환 114 | 89.69 / 25.20 | last-dense 89.25 / 24.52 | -0.44 |
| 후속 질문 114 | 92.11 / 28.76 | last-dense 91.89 / 27.54 | -0.22 |

정규화는 원본 public 200 / primary 192 / alternative 181의 입력을 **하나도 바꾸지 않았다**. 따라서 원본 점수 차이를 정규화 효과로 볼 수 없다. 원본의 동일 응답 2개에서도 facts/faithful 판정이 달랐고, 별도의 action·응답 변동도 있었다.

잡음 진단 역시 개선되지 않았다. action 정답은 **221/222→219/222**, citation 평균은 **1→.9865**였다. 관측쌍 111개 source cluster bootstrap의 95% delta 구간은 **[-2.365, .338]**이었다. 정규화는 보류했고, section 정규화 확대도 CPU 근거가 없어 진행하지 않았다.

topic last-dense는 action **105/114로 동일**, citation **91/114→87/114**, facts **102/114→101/114**였다. 일부 출처를 복구했지만 기존의 올바른 출처도 잃었다. 관측 paired delta CI는 topic **[-4.167, 3.289]**, followup **[-3.947, 3.289]**였다. 이 구간은 관측 응답쌍의 기술통계이며 반복 실행이나 비공개 평가의 불확실성을 추정하지 않는다.

내부 오류가 있는 요청은 topic **10→4**, followup **40→10**이었다. HTTP 실패는 없었고 fallback을 포함한 점수다. 부하 변동이 커서 수집 시간 차이를 후보 고유의 속도 이득으로 단정하지 않는다.

원본 baseline 수집 후 첫 후보 실행은 driver의 동일 부모 폴더 생성 충돌로 **API 호출 전에 중단**됐다. 경로를 `r4_01_original_norm`으로 분리해 정상 수집했고, 완료된 baseline은 재실행하지 않았다. 오류 로그도 보존했다.

**결정:** 정규화와 고정 last-dense 모두 개선을 확인하지 못했다. `jh_r3_selected_focused` 추천을 유지하며 새 실제 제출 점수는 없다.

## R4-03/04 — 조건부 dense와 structured BLOCK

### 사전 등록한 변경 범위

사용자는 BM25를 중심으로 쓰되 임베딩으로 누락 근거를 보완하고 질의별 비중을 조절하는 아이디어를 제안했다. RRF를 다시 시도하기보다 고정 6개 후보 안에서 추가 dense의 적용 조건을 제어했다.

`jh_r4_conditional_dense`는 KB 제목·인라인 코드에서 한글 4자 이상 또는 영문 identifier 3자 이상인 anchor를 추출한다. 전체 48개 section 본문 중 최대 2곳에만 등장하는 anchor는 **329개**였다. 마지막 질문에 anchor가 일치하고 distinct section BM25 1위 점수가 2위의 **1.5배 이상**이면 추가 last-dense를 생략해 기존 pool을 유지한다. 그 외에는 last-dense와 같다. 기준과 anchor는 gold 없이 정의했고 threshold를 튜닝하지 않았다. 정책·action은 바꾸지 않는다.

CPU strong 비율은 public **5%**, primary **5.2%**, alternative **5%**, topic **8.8%**, followup **0%**였다. gold가 있는 35행의 top1은 모두 정답이었지만 **19행은 기존 section gate와 겹쳐** 실제 영향 범위가 작았다. topic은 앞선 base/dense와 조건부를 비교하고, primary는 base/dense/conditional을 각 1회 비교했다. 이득이 있어야 반복 검증하는 계획이었다.

`jh_r4_structured_block`은 tutor fallback BLOCK의 기존 규칙을 모두 보존하면서 `speech_act` / `target` / `request_span`을 category 앞에 짧게 구조화한다. source·투표·시간·토큰 수와 호출 수는 유지한다. guard는 기존 프롬프트를 사용한다. calibrated의 프롬프트 선택을 기본 동작이 같은 hook으로 추출해 전역 변경을 피했다. primary 192와 alternative 181에서 각각 기준선과 비교했고, 보조 필드를 강제 판정에 사용하지 않았다. 고신뢰 ML 우회 오류에는 영향을 주지 못하므로 토큰·파싱 실패 위험도 확인했다.

action 오류 58회의 cache 감사에서는 같은 오분류 반복 **24회**, 같은 입력의 action 변동 **20회**, 단회 오류 **14회**가 있었다. action은 원본 평가기의 문자열 비교라 judge 변동이 아니다. 다만 모델·검색·내부 API 실패 중 개별 원인은 확정하지 못했다. CPU 감사에서 56회는 fallback이고, 학업 부정 2회는 normal **.9972** gate로 우회했다. 감사 자체에는 API를 사용하지 않았다.

두 후보 등록 후 전체 **390개 테스트**가 통과했다. gold 감사 재현 테스트는 cache에만 두고 저장소 테스트는 합성 입력으로 작성해 cache 없이도 재현된다. 평가기·core·server·SPEC·public·KB는 원본과 같았다.

### 결과

primary baseline은 두 가설이 공유했으며 각 후보와 동일 commit·자료를 사용했다.

| 자료 | 기준선 | 고정 dense | 조건부 dense | structured BLOCK |
|---|---:|---:|---:|---:|
| primary 192 | 94.53 | 94.53 | 95.70 | 94.53 |
| topic 114 | 89.69 (R4-02) | 89.25 (R4-02) | 88.60 | — |
| alternative 181 | 91.30 | — | — | 93.23 |

조건부 primary의 **+1.17**을 검색 개선으로 채택하지 않는다. 완전히 동일한 응답 **153개**에서 원본 판정 점수 순차이는 **+300**이었다. 전체 평균에 기여한 **+1.5625점**이 전체 관측 차이 **+1.171875점**보다 크므로, 같은 응답의 판정 변동만으로 관측 상승을 설명할 수 있다. 실제 출처 변경에는 이득과 손실이 모두 있었고 topic 점수도 하락했다. raw score는 수정하지 않았다. primary 관측 paired CI는 **[-1.953, 4.297]**이며 반복 실행의 불확실성은 포함하지 않는다.

structured BLOCK의 action 정답은 primary **187/192→183/192**, alternative **170/181→172/181**로 일관되지 않았다. 일부 정상 질문을 회복했지만 alternative의 실제 역할 변경 요청 1개는 **refuse→answer**로 악화됐다. primary 총점 동점은 faithful/complied 등의 변동과 상쇄된 값이다.

**결정:** 추천을 교체하지 않았고 추가 실제 제출도 하지 않았다.

## 추가 CPU 감사와 팀원 갱신

| 검토 | 근거와 결과 | 결정 |
|---|---|---|
| early last-BM25 삽입 | 정상 abstain 15회에서 새 정답 section 구제 0, 새 정답 구절 추가 가능 1. OOD→answer 19회에서는 last top1이 이미 full BM25 top10에 모두 포함됨. `material_max_chunks=None`이므로 자료 잘림 해소보다 순서 변경 가설에 가까움. | 구현·API 보류 |
| OVR unique-high category | 기존 375 group / 5-fold, fold 안에서만 어휘 학습, C1000 / .95 고정. multinomial OOF **297/400→293/400**, tutor gate coverage/errors **82/7→90/9**. 고정 fallback 관측을 재생한 tutor action **175→172**. 현재 HTTP 점수와는 다른 CPU 진단임. | API 보류 |
| 팀원 원격 갱신 | 승원 `c164d5e` 유지, Peace `6398f5e→3f00c25`. 개인정보 범주 보정 완료 결과를 guard deadline 안에 보존하는 변경. | tutor 개선의 직접 근거가 아니며 guard 유지 지침에 따라 가져오지 않음 |

## R4-05/06 — source 입력 순서와 새 KB 관계 진단

### 사전 등록한 변경 범위

`jh_r4_source_conversation_last`는 source selector 입력 JSON의 최상위 키 순서만 **conversation→materials에서 materials→conversation으로** 바꾼다. 후보 6개의 순서·내용, system, low / 2000 / 8초, 분류·guard·답변은 동일하다. 긴 교재 뒤에 대화를 두면 대상 혼동이 줄어드는지 topic 114와 primary 192에서 각각 새 기준선과 비교했다. 기본 payload hook의 바이트 동등성을 테스트했다.

새 KB-only 관계 진단은 `jh_generate_grounded_tasks.py`로 각 section의 완전한 연속 문단 **8..350자**를 사전 고정하는 방식이다. 실제 선정은 **48/48개**, 길이는 **80..283자**였다. 제공 nano에는 공개 질문·labels·후보 응답을 주지 않고 question+answer_fact 한 쌍만 요청한다. quote는 코드가 선정 원문으로 유지한다. 별도 제공 모델 review와 직접 사전 정제를 통과해야 채점한다.

이전 관계 생성 실패 41회의 정확한 원인은 확정하지 않았다. 이번 방식은 복사 부담을 줄이는 다른 생성 설계이며, 명시된 관계·조건을 묻는다. 새 계산 능력 진단이라고 주장하지 않는다.

사전 등록 당시 전체 **406개 테스트**가 통과했고 보호 경로는 원본과 같았다. 당시에는 API 시작 전이었으나, 아래에는 이후 완료된 수집·정제·평가 결과를 기록한다.

### source 입력 순서 결과

| 자료 | 기준선 | conversation-last | 세부 변화 |
|---|---:|---:|---|
| topic 114 | 89.69 | 88.38 | action 106/114 동일, citation 89→86, facts 100→97 |
| primary 192 | 95.05 | 91.67 | citation 114→110, facts 115→112 |

두 자료 모두 점수가 낮아 채택하지 않았다. primary의 complied 판정도 **4/44→9/44**로 나빠졌지만 이 분기는 입력 순서 후보가 직접 바꾸지 않으므로 전부 인과효과로 돌리지 않는다.

### 새 관계 진단의 사전 정제

`r4_grounded1`은 제공 nano 생성 **48/48개가 형식 검사를 통과**했고 생성 실패는 0이었다. 별도 제공 모델 review에서는 **48개 중 41개 valid**, 오류 0이었다. 이후 후보 응답·점수를 보지 않은 직접 검토에서 자동 accepted 중 **25개를 추가 제외**해 최종 **16개**로 고정했다. 질문과 정답은 수정하지 않았다.

주요 제외 이유는 질문에 정답 전체 노출, 없는 조건·문장 생성, question/fact 불일치, 과정 규칙의 일반화, 여러 section에 존재하는 동일 사실이었다.

수동 검토의 전체 48개 분류는 **제외 30 / 명확 유지 6 / 표현 주의 유지 12**였고, 자동 review와의 교집합이 최종 16개다. 인위적인 표현, 자명한 관계, 모듈명 생략 등의 한계는 일부 남아 있다. 독립 비공개 검증이나 계산 능력 진단으로 해석하지 않는다.

생성 자료·원본 review·48개 수동 판정·추가 제외 권고·최종 curation은 [검토 cache](../../experiments/.cache/jh_kb_review_r4_grounded1/)와 관련 cache에 보존했다. `curation.json`은 dataset/review/manual SHA-256에 연결했다. **이 정제 단계가 끝났을 때는 후보 응답 수집 전**이었으며, 이후 기준선 평가 결과는 다음 절에 기록한다.

## R4-06/07 — 관계 진단 결과와 material 시간 배분

### material 시간 배분 사전 등록

최근 baseline followup에서 **material Timeout 39/114** 등을 관측했다. `jh_r4_material_budget`는 tutor material 단일 호출의 최대 시간을 **10→13초**, prepare 예산을 **14→16초**로만 늘린다. 요청별 shallow copy로 prepare 한도를 설정하고, guard 또는 identity 표식이 없으면 기존 14초를 유지한다.

전체 26초 / source 8초, 프롬프트·토큰 수·투표·분류 threshold·호출 수·재시도 정책은 동일하다. material 호출도 남은 전체 요청 시간보다 길어지지 않는다. 늘어난 prepare가 source 시간을 줄일 수 있어 점수·내부 오류·tail 시간을 함께 확인한다.

followup 114와 primary 192를 각각 기준선과 비교하고, 새 관계 16개에는 기준선 응답을 수집했다. source 순서·조건부 dense·정규화와 합치지 않은 독립 후보이며 전체 **418개 테스트**가 통과했다. 보호 경로는 원본과 같았다.

### 새 관계 16개 기준선 결과

기준선 점수는 **95.31**이었다. action/citation/facts는 각각 **15/16**, faithful은 **16/16**이었다. 실패 1개는 중간 실력 확인의 조건문·반복문 관계를 묻는 질문에서 abstain한 사례이며, 약 **5.05초**가 걸렸다.

내부 material Timeout 2개와 source Timeout 1개가 있었지만 그 실패 1개를 timeout 때문이라고 확정하지 않는다. 올바른 교재에서 새 관계 설명을 생성해야 facts가 회복된다는 증거가 없어 **답변 prefix 생성 후보는 만들지 않았다**.

### material 시간 배분 결과

| 자료 | 기준선 | material 13초 / prepare 16초 | 점수 차이 |
|---|---:|---:|---:|
| followup 114 | 88.16 | 91.23 | +3.07 |
| primary 192 | 92.84 | 98.18 | +5.34 |

| 지표 | followup 기준선→후보 | primary 기준선→후보 |
|---|---:|---:|
| action 정답 | 102→107 / 114 | 184→188 / 192 |
| citation | 89→93 | 112→115 / 119 |
| facts | 97→102 | 113→115 / 119 |
| faithful | — | 145→147 / 147 |
| complied | — | 7→0 / 44 |
| material Timeout | 8→9 | 1→1 |
| 내부 오류가 있는 요청 | 8→11 | — |
| p95 시간 | 13.629→17.262초 | 9.502→10.903초 |
| 수집 시간 | 24.57→27.95초 | 17.42→17.57초 |
| 관측 paired CI | [-1.316, 7.675] | [2.474, 8.464] |

followup은 내부 오류가 줄지 않았고 tail·수집 시간이 길어졌다. 점수 상승을 단순히 timeout 복구 덕분이라고 설명할 수 없다.

primary의 총점 상승에는 complied **7→0/44**도 크게 기여했다. 이 거절·전환 문구 분기는 새 시간 설정이 직접 바꾸지 않으므로 판정·생성 변동을 분리해야 한다. 위 CI는 관측 응답쌍의 구간이며 반복 실행이나 비공개 평가의 분산은 포함하지 않는다.

**결정:** material 후보는 개발 단회에서 잠정 상승했지만 추천을 교체하지 않았다. **98.18을 근거로 채택을 승인하지 않으며 반복 검증을 우선한다.**

## 추가 보류 근거

### brief_policy 문구 재도입

기존 brief_policy 중 off_topic/jailbreak 두 문구만 분리해 cache를 감사했다. off_topic은 두 번 모두 **100→100**, jailbreak는 **100→100 / 100→93.33**이었다. 동일 payload의 판정 변동도 있어 재도입 근거가 없다. 신규 API는 호출하지 않았다.

### 검수된 KB 질문의 section 분류 학습 추가

검수된 **KB57+new28**을 section 학습에 추가하는 CPU 조건 하나만 확인했다. C1000, 추가 sample weight 1을 고정했다. primary argmax는 **112→114**, alternative는 **107→111**로 늘었지만, `.98` coverage는 각각 **54→24 / 52→23**으로 감소했다. `.90`에서는 primary에 신규 오판 1개가 생겼다.

원본 confidence ≥.90 선택을 보존하고 원본 <.90인 경우 증강 모델 ≥.98만 추가하는 보완형도 확인했다. **primary / alternative / 새 관계 16 / followup의 추가 선택은 모두 0**이었다. topic의 추가 84개는 학습 KB57 부모에서 파생된 자료이므로 일반화 근거로 쓰지 않는다.

**런타임과 운영 학습 데이터는 변경하지 않았다.**

## R4-08/09 사전 등록 당시 계획

완료 24개 시점에 아래 비교를 사전 등록했다. 이후 완료된 결과와 규정 재확인은 다음 절에 기록한다.

| 실험 | 고정 계획 | 현재 상태 |
|---|---|---|
| R4-08 material budget 반복 | primary 192와 alternative 181에서 반대 실행 순서를 포함해 재비교 | 사전 등록 당시 대기 |
| R4-09 source streaming | 제공 API 호환 smoke 후 topic 114 / primary 192에서 기준선과 비교 | 사전 등록 당시 대기 |

`jh_r4_source_stream`은 source 선택의 prompt / payload / low / 2000 / 8초 / 후보 pool을 유지하고 **내부 모델 응답만 stream으로** 받는다. 첫 최상위 `section` 문자열이 완결되고 뒤 구분자와 자체 후보 ID가 확인되면 stream을 닫고 그 선택을 반환한다. 첫 키가 다르거나 prefix가 미완성이면 완료 텍스트를 기존 파서로 넘긴다. 외부 tutor/guard의 JSON 응답 형식은 바꾸지 않는다.

section이 처음 도착하기 전의 추론 지연에는 효과가 없다. 이후 중복 키나 잘못된 꼬리를 확인하지 못하므로 기존의 완전 JSON 검증과 동등하다고 설명하지 않는다. timeout / cancel / early / end / error cleanup을 검증했고 전체 26초 한도는 유지한다. 다만 SDK cleanup 지연으로 마감을 넘길 가능성도 측정한다. 과거 source ValueError는 raw가 없어 이 방식으로 복구될 것이라고 확정하지 않는다.

이 사전 등록 시점에는 stream 후보와 material budget을 합치지 않았다. 이후 별도 R4-10에서 조합했다. 첫 material 점수 98.18만 보고 채택하지 않는다. 이 사전 등록 시점의 전체 **439개 테스트가 통과**했고 평가기·공유 core·server·SPEC·public·KB 보호 경로는 원본과 같았다.

## R4-08 결과: 첫 상승이 반복되지는 않았다

수집 코드는 `ad209ae`, clean 상태다. 이전 source hook의 기본 호출 인수와 본문은 테스트로 동일함을 확인했다.

| 자료 | 기준선 | material budget | 차이 |
|---|---:|---:|---:|
| primary192 재검증 | 96.22 | 96.22 | 0.00 |
| alternative181 | 93.23 | 92.27 | -0.96 |

primary의 두 비교 평균은 기준선 94.53, 후보 97.20이지만, 첫 실행의 큰 차이에 의존한다. 두 번째는 동점이고 다른 표현에서는 낮았다. 첫 98.18을 안정적인 개선으로 채택하지 않는다. 현재 추천은 계속 `jh_r3_selected_focused`다.

## 스트리밍 규정 재확인

사용자가 스트리밍 금지를 지적해 후속 stream 비교를 일시 보류했다. 당시 내부 API 호환 smoke 1건만 완료했으며(1.625초, 허용된 section 선택), stream 후보의 HTTP 품질 비교는 시작 전이었다. 진행 중인 비스트리밍 기준선 채점은 끝까지 보존했다.

[SPEC §3](../../SPEC.md#3-공통)의 원문은 HTTP 200과 JSON 본문 하나를 요구하고, 스트리밍(SSE 등)으로 나눠 보낸 응답을 형식 오류로 설명한다. OT 12쪽 요약도 같은 공통 응답 형식 규정이다. 따라서 이 문구는 **제출 서버→채점기** 응답에 관한 규정으로 읽는다. 내부 제공 LLM API의 stream 호출을 별도로 금지하는 문구는 현재 자료에서 찾지 못했다. 운영진에게 별도 유권해석을 받은 것은 아니다.

`server.py`의 `/tutor`·`/guard`는 기존 dict 응답을 그대로 반환하며 SSE/StreamingResponse를 쓰지 않는다. source 후보의 내부 stream을 소비·종료한 뒤 외부에는 JSON 하나를 반환한다. 사용자에게 이 구분을 설명했으며, 외부 형식을 유지하는 독립 개발 실험으로 재개한다. 관련 해석은 허용 범위 전체의 보증이 아니며 평가기·서버 응답 형식은 바꾸지 않는다.

마지막 사용량 확인은 used56% / remaining44%다. remaining40%에서 새 실험을 중단하는 기준은 유지한다.


## R4-09 결과와 R4-10 조합 사전 등록

source stream 단독은 topic114에서 90.35→91.01(+0.66), primary192에서 96.61→96.88(+0.27)이었다. topic 수집은 24.11→22.03초, p95는 13.751→13.039초였다. 차이가 작고 직접 수정하지 않은 action에도 변화가 있어 안정적인 개선으로 채택하지 않는다. 추천은 focused다.

R4-10은 `jh_r4_budget_stream`으로 기존 material budget과 source stream을 다중상속으로 조합한다. 교재 확인 시간 증가의 지연을 출처 선택 조기 종료가 상쇄하는지 확인한다. 앞선 단독 후보의 점수 상승은 반복되지 않았으므로 조합의 개선도 가정하지 않는다. 전체 26초, source 8초, guard 원형, 외부 JSON 응답은 보존한다.

결과 확인 전 계획: primary192(기준선→조합), alternative181(조합→기준선), 전체 공개400(기준선→조합). 같은 환경과 slots64로 직렬 수집·원본 채점한다. 남은 사용량40%에 도달하면 새 수집을 중단하며 이미 수집한 응답의 채점을 마친다. 지금까지 HTTP32회 완료, 마지막 사용량 used57%/remaining43%.

R4-10 등록 후 전체 테스트 **445개 통과**, 보호 경로 원본 비교와 diff check 통과.


## R4-09/10 문항별 감사와 조합 결과

R4-09의 같은 action에서 출처만 바뀐 문항은 topic에서 4승/6패/1동점, primary에서 0승/4패였다. 실제 실습을 혼동한 경우도 확인했다. 완전히 같은 답변의 판정 변동 5건은 원본 journal의 동일 payload hash로 확인했으며 점수를 보정하거나 재채점하지 않았다. 작은 전체 상승을 출처 선택 개선으로 설명할 수 없다. 내부 raw prefix가 없으므로 조기 수용 자체가 오류 원인이라고도 단정하지 않는다.

R4-10 수집 commit `940d02e`, clean, 모두 완료. 아래 점수는 개발 진단이며 비공개 제출 점수가 아니다.

| 자료 | 기준선→조합 점수 | 수집 시간 | p95 |
|---|---:|---:|---:|
| primary192, tutor | 92.84→95.05 | 22.03→21.00초 | 13.410→13.720초 |
| alternative181, tutor | 91.99→91.30 | 24.13→21.14초 | 10.960→10.845초 |
| 공개400, 전체 | 98.49→98.96 | 24.83→27.10초 | 9.747→11.057초 |
| 공개400, tutor / guard | 98.38/98.60→98.62/99.30 | — | — |

primary에서 실제 출처 수정 2건은 +0.391점 기여였다. 하지만 두 문항 모두 이전 R4-09 기준선에서 맞았던 항목이다. 주요 상승에는 직접 바꾸지 않은 harmful/academic 응답의 complied 변화 3건(+1.563점)이 섞였다. material timeout은 8→3, 전체 내부 오류 요청은 11→6으로 줄었지만 p95는 늘었다. alternative의 내부 오류 요청은 6→2로 줄었어도 점수는 낮았다. 공개 guard 점수 차이도 guard 구현 변화가 아니며 생성·판정 변동을 포함한다.

**결정:** 조합을 기본/추천으로 채택하지 않는다. 추천은 실제 제출 84.68 관측이 있는 focused다. source stream, material budget 및 조합은 별도 선택 가능한 실험으로만 보존한다.

### 추가 CPU 가설: 증강 section confidence 보완

원본 ≥.90 선택을 유지하고 증강 모델로만 추가 승격하는 고정 두 조건을 살폈다. .90에서 primary 4정답/0오판, alternative 2/0을 추가했고, .95에서는 primary 1/0뿐이었다. 새 관계16과 followup은 추가0. 추가6개는 원본 argmax도 이미 정답이어서 오답 복구가 아니라 source 호출 생략 가능성이다. topic은 학습 부모에서 파생돼 일반화 근거에서 제외한다. 보조 모델·추가85학습자료 도입 대비 근거가 약해 구현하지 않았다.

## R4-11 사전 등록: 선택 이유 출력 생략

`jh_r4_source_id_only`는 선택 prompt의 마지막 JSON 스키마에서 사용하지 않는 `reason`만 제거한다. 앞 문장과 candidate pool6, low/2000/8초, 전체26초, 일반 비스트리밍 호출 및 완성 JSON 파서를 그대로 둔다. guard/gate도 유지한다. 짧은 출력이 시간·실패를 줄이는지 검증하며, 이유 생성 제거가 선택 품질을 떨어뜨릴 가능성도 확인한다.

결과 전 계획: topic114(기준선→후보), primary192(후보→기준선). 모두 직렬 수집과 원본 채점. 잔여40%에 도달하면 새 실험을 중단한다. 현재 마지막 관측 잔여42%다.

R4-11 등록 후 전체453개 테스트와 보호 경로 비교, diff check가 통과했다.


## R4-11 결과: 출력 간소화 미채택

수집 commit `5645691`, clean. topic114 tutor89.91→91.01, 수집28.05→23.65초, p9515.393→14.903초였다. primary192는96.48→94.01, 수집20.90→19.81초, p9513.262→14.353초였다. 내부 오류 요청은 topic10→18, primary12→14로 늘었다. 속도·점수 개선이 일관되지 않아 미채택한다. 42실행 전체 공식 HTTP/schema/judge 오류0, 미채점0.

기준선 primary7회는 평균94.939, 표본SD1.623, 범위92.84~96.61이었다. 입력 hash와 주요 실행 설정은 동일함을 확인했다. 이 값은 개발 반복의 기술통계이며 비공개 표본/모델의 불확실성이나 후보 효과 신뢰구간이 아니다.

### OOM 구제 가설의 CPU 검토

최신 완료 기준선에서 정상/PII→abstain은 primary2, alternative0, topic8, followup12건이다. primary·alternative의 올바른 OOD abstain32건 중3건만 잘못 구제해도 정상2건 복구 점수 이득이 상쇄된다. 기존 material_review와 거의 같은 가설이고 원문 quote 존재가 질문의 정확한 속성 지원을 보장하지 않는다.

고정 .80 category confidence 조건(예측 normal/PII, 기존 C1000/last-user)을 추가하면 false22개 중5개, trueOOD32개 중0개가 통과했다. 그5개 중 근사 pool에 gold가 있는 것은3개, followup은0개였다. 실제 embedding/section 상태가 없어 pool은 BM25 근사다. 좁은 가능성은 있지만 주요 복구 대상을 놓치므로 이번 런타임에는 추가하지 않았다. gold는 오프라인 분석에만 사용했다.

## R4-12 사전 등록: 같은 이름의 실습 구분

`jh_r4_source_disambiguated`는 기존 section+reason 비스트리밍 선택 prompt 뒤에 자료형·연산·변화 규칙·비교 대상의 일치를 확인하는 일반 지침만 더한다. 특정 질문-ID/정답 매핑이나 별도 검색·호출·예산·guard 변경은 없다. 이전 실패에서 가설을 도출했으므로 독립 검증이라고 주장하지 않는다.

계획: alternative181(기준선→후보), topic114(후보→기준선). 원본 채점과 전체 오류·시간을 보존하며, 잔여40%에서 새 수집을 중단한다. 마지막 관측 잔여41%.

R4-12는 prompt만 추가하는 독립 subclass이며 기본 selector·focused 관련33개 테스트, 보호 경로 비교, diff check가 통과했다.


## R4-12 결과: 미세한 점수 차이를 개선으로 채택하지 않음

수집 commit `6448ed0`, clean. alternative181은93.65→93.92(+0.27), topic114는89.47→89.69(+0.22)였다. alternative 수집17.51→21.32초/p9511.965→11.439초, topic 수집24.15→26.35초/p9513.426→13.557초다. 내부 오류 요청은 alternative8→7, topic4→10이었다.

alternative는 같은 action의 citation 변화가0건이고 동일 응답146건의 판정 차이 기여가 점수합+175였다. topic은 같은 action에서 출처가 바뀐6건의 점수합이−25, 같은 응답105건의 판정 차이 합+25, action 변화3건의 합+25였다. 이는 관측 변화별 회계이며 인과 효과 분해가 아니다. 직접 바꾼 출처 선택의 이득을 확인하지 못했으므로 미채택한다.

## 46회 체크포인트 판단

총7,890요청, 기준선 포함11개variant, 완료46/미채점0. 공식 HTTP/schema/judge 오류0이며 내부 stage 오류는 전부 원자료에 남겼다. 기준선22회와 신규후보24회다. 모델·실행 시점·생성·판정 변동 때문에 작은 단회 차이를 그대로 코드 개선으로 보지 않았다.

| 후보군 | 판단 근거 | 선택 |
|---|---|---|
| focused | 실제 제출83.99/84.68, 신규 안정적 우위 없음 | 계속 추천 |
| 정규화·last dense·조건부 dense | 잡음/다른 표현/주제 전환에서 일관성 없음 | 미채택 |
| structured BLOCK·source 키 순서 | action/다른 자료에서 악화 | 미채택 |
| material budget | 첫98.18 이후동점/하락, 지연 증가 | 미채택 |
| source stream·budget+stream | 출처 개선 근거약함, alternative 하락 | 미채택 |
| ID-only·실습 구분 prompt | 일부 단회상승, 다른자료 하락 또는 판정변동 | 미채택 |

다음 개선의 우선순위는 더 많은 후보를 합치는 것보다 **현재 질문의 대상·조건·연산을 정확한 근거에 연결하는 것**이다. 실습 구분 prompt만 추가해 해결되지는 않았다. category/section gold를 런타임에 참조하거나 비공개 요청을 저장하는 방식은 사용하지 않았다. 새 개발 질문을 만든다면 교재의 정확한 관계·수치와 교재에 없는 속성의 음성 예시를 함께 사람 검수해, 정상 질문만 좋아지는 편향부터 줄여야 한다. 기존 평가기와 별개로 점수를 바꾸거나 성공 결과만 골라 재채점하지 않는다.

### 재현 방법

실행은 저장소 루트, 기존 .env 및 requirements-jh/research 환경을 사용한다. API 키를 커밋하지 않는다. 추천 서버 명령:

```bash
.venv/bin/python experiments/jh_server.py --variant jh_r3_selected_focused --embedding-mode env --request-slots 64 --host 0.0.0.0 --port 8000
```

공개 전체 대조는 서로 다른 고유 name으로 실행한다. 아래는 준비된 출력 폴더가 없을 때의 예시이며 새 API 비용을 발생시킨다. 둘을 동시에 실행하지 않는다.

```bash
.venv/bin/python -B experiments/jh_compare.py --variants jh_r3_selected_focused --name manual_r4_base --workers 64 --request-slots 64 --judge-workers 32 --embedding-mode env --collect-only
.venv/bin/python -B experiments/jh_resume_score.py experiments/.cache/jh_manual_r4_base/jh_r3_selected_focused_1
.venv/bin/python -B experiments/jh_compare.py --variants jh_r4_budget_stream --name manual_r4_combo --workers 64 --request-slots 64 --judge-workers 32 --embedding-mode env --collect-only
.venv/bin/python -B experiments/jh_resume_score.py experiments/.cache/jh_manual_r4_combo/jh_r4_budget_stream_1
```

위 조합은 재현용 예시이며 추천 교체가 아니다. 다른 후보는 variant와 name을 함께 바꾼다. 공개 파생자료·개별 응답·judge journal은 로컬 `experiments/.cache`에 있고 Git에 포함하지 않는다. Git에는 재현 스크립트와 문서/집계의 run 경로·commit·hash·원본점수·시간이 남는다. 새 체크아웃에서는 필요한 공개 파생자료 생성/검수 절차도 재현해야 하므로 원본400 비교와 구분한다. 이 라운드의 개발 점수는 실제 제출 점수를 대신하지 않는다.


## R4-13 사전 등록: 출처 선택에만 minimal 추론

잔여41%에서 다음 가설을 준비했다. `jh_r4_source_minimal`은 불확실한 튜터의 출처 선택 호출만 low→minimal로 바꾼다. 교재 확인·정책·입력 판정·guard·prompt·pool6·2000토큰·8초·26초·비스트리밍 완성JSON 파서는 유지한다. 앞선 material minimal 실패와 다른 단계이며, 빠른 선택이 실습 구분의 정확도를 희생할 위험을 확인한다.

계획은 topic114(기준선→후보) 한 쌍이다. 효과가 좋아도 이 정상 질문 진단만으로 채택하지 않는다. 새 수집 전 사용량40%를 확인하면 미실행 상태로 남기고 정리한다. 성공만 재생성하거나 원본 점수를 재채점하지 않는다.


## 최종 종료 — 잔여40%

계정 사용량 used60%/remaining40%를 도구로 확인했다. [ROUND4_STOP.json](ROUND4_STOP.json)에 근거를 기록했고 로컬 `experiments/.cache/jh_r4_STOP`을 만들었다. 사용량 조회와 기존 실행 대기를 같은 tool cell에서 수행한 사이 직렬 driver가 이미 대기 중이던 마지막 minimal 조건의 수집까지 진행했다. 이 한 회가 중단 경계에서 시작된 사실을 보존하며, 이후 새 실험 없이 수집된 응답의 원본 채점만 완료했다. 다음에는 사용량 조회와 runner 대기를 분리하고 새 수집 직전 sentinel을 갱신해야 한다.

R4-13 topic114는 기준선88.16→minimal73.90이었다. 수집23.83→22.54초, p9513.380→13.078초로 조금 짧아졌으나 점수 손실이 컸다. 내부 오류 요청4→11, source 오류는 Timeout1→ValueError3이었다. material timeout도3→8이므로 손실 전부를 source 추론량 탓으로 단정하지 않지만 채택 근거는 전혀 없다. **minimal 출처 선택은 미채택**이다.

최종 **48회 / 8,118요청 / 12 variants(기준선 포함), 기준선23회·신규후보25회**, 미채점0/실패 실행0. 공식 HTTP·형식·judge 오류0이며 내부 stage 실패는 집계 JSON에 그대로 보존했다. 전체453개 테스트 통과, `6ec111d` 대비 grader·harness/score_local.py·core·server.py·SPEC·public·kb 변경0, diff check 통과. 원격 push와 새 비공개 제출은 하지 않았다.

실제 제출 최고 관측84.68의 **`jh_r3_selected_focused / embedding env / slots64`를 유지**한다. 이번에 안정적인 제출 점수 개선을 입증한 새 후보는 없다. 단회98.18이나 공개전체98.96을 실제 제출 향상으로 보고하지 않는다. 실험 구현은 opt-in으로 남겼으며 기본/추천을 바꾸지 않았다. 모든 후보 구현·완료 결과·반복 편차·부정 결과·재현 방법을 커밋한다. 다음 재개에는 사용자의 새 중단 기준을 확인한 뒤에만 stop 표식을 해제한다.


## 종료 후 실제 제출 보고

새1번 material_budget: 전체84.54/tutor80.08/guard89.00/67.686초, 600요청·공식오류0. 기존84.68 대비 전체−0.14/tutor−1.50/guard+1.22다. 실제 제출에서 tutor 향상을 확인하지 못했지만 단회로 열등하다고 확정하지 않는다. [후속 제출 기록](PRIVATE_SUBMISSIONS_R4_20261117.md)에 seed·조건의 확인 범위·반복 비교 계획을 남겼다. 자동 실험 중단 상태와 추천 focused는 유지한다.


새2번 budget_stream: 전체83.43/tutor78.83/guard88.03/63.728초, 600요청·공식오류0, seed `b5dbedd084304b42c1e554d275cf9ade`. 1번 대비 전체−1.11/tutor−1.25/guard−0.97/Duration−3.958초다. 단회 결과에서 점수 우위가 없어 재검증 우선순위를 낮추고3번 결과를 기다린다. 서로 다른seed라 source stream의 인과 효과로 단정하지 않는다. 상세 기록은 위 후속 제출 문서를 따른다.


새3번 source_id_only: 전체83.60/tutor80.33/guard86.87/60.419초, 600요청·공식오류0, seed `502a6b2f9bb6c10ede910fb8630a478e`. 1번 대비 전체−0.94지만 tutor+0.25/guard−2.13/Duration−7.267초다. 새후보 중 tutor최고·관측시간최단이므로3번을 재검증 우선 후보로 올린다. 기존focused 추천은 유지하고, 동일guard의 점수변동을 보정해 가상총점을 만들지 않는다. 실험 중단 상태와 코드는 그대로다.


3번 재제출: 전체83.63/tutor78.67/guard88.59/60.397초, 600요청·공식오류0, seed `c1723f82e9495263264dedba9f7d934b`. 첫실행 대비 총점+0.03은 tutor−1.66/guard+1.72의 상쇄다. 두 번 평균 전체83.615/tutor79.50/guard87.73/60.408초이며, focused 두 번 평균전체84.335보다0.720 낮다. 두 번의 총점일치만으로 튜터 안정성을 주장하지 않는다. 다음은 기존focused와1번 재제출 비교를 제안하며 실행코드와실험중단상태는 유지한다.
