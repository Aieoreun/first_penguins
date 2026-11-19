# 추론 수준 분리 실험

사용자 요청: 기존 A/C를 보존하고 별도 후보에서 추론 수준을 높여 점수·시간을 비교한다. 실제 A/C5회씩 제출은 아직 시작 전이며, 사용자가 로컬 API 실험부터 진행하도록 확인했다. 로컬 수집·judge를 동시에 돌리거나 실제 제출과 겹치지 않는다. 주최측 evaluator/core/KB/public/SPEC는 수정하지 않는다.

## 결론 및 현재 상태

**기존 A `jh_cm_char_mlp50`와 C `jh_np_timeouts`를 유지한다.** medium 후보로 교체할 반복 개선 근거가 없다. 공개 파생 완결 비교15회·2,924응답을 채점했고, 별도 중단 배치616개 저장 응답은 기술적 실패·보조 진단으로 구분해 남겼다. 완결15회 HTTP 오류·공식 형식 오류·judge 오류는 모두0이다. 내부 timeout/parse 실패는 아래 표와 JSON에 포함한다.

로컬 API 수집·채점 프로세스와 임시 서버는 종료했다. A/C 실제 번갈아 제출을 시작해도 된다. 기존 A/C 기본값·학습 모델·검색·평가기·`.env`를 바꾸지 않았다. 새 후보는 별도 variant로만 선택한다. 신규 실제 제출 점수는 없다. 이 결과만으로 nano 자체의 성능 한계나 비공개 점수 하락 원인을 확정하지 않는다.

[완결 비교15회 전체 집계](reasoning_ablation_results.json) / [중단 실행·보조 결과](reasoning_ablation_aborted.json).

## 실제 설정 감사 및 정정

이전 설명의 “답변 생성 minimal”은 config 환경변수만 보고 한 불완전한 설명이었다. 실제 A/C의 `answer_effort`는 상속한 v4 클래스가 **low**로 고정한다. GPT 입력 판정·근거·안전성·교재 선택도 실제 low 경로이다. ReasonedTie 파일은 있지만 A/C는 그 클래스를 상속하지 않고 Adaptive의 low tie를 사용한다.

튜터의 NORMAL/PII는 선택한 교재를 그대로 발췌하는 경로가 많다. 모든 답변을 LLM으로 생성하는 구조가 아니므로 답변 effort만 높이면 선택/입력 오류는 그대로다. 교재 선택, 입력 판정, GPT 텍스트 생성을 각각 나누어 비교한다.

## 사전 고정 후보

|라벨|variant|기준선|변경|
|---|---|---|---|
|A|jh_cm_char_mlp50|기존 A|변경 없음|
|C|jh_np_timeouts|기존 C|변경 없음|
|GI|jh_re_c_input|C|guard GPT 입력 block/other/material 및 tie: low→medium|
|GG|jh_re_c_ground|C|guard 근거 검사5표: low→medium|
|GS|jh_re_c_safety|C|guard 독립 안전성 검사: low→medium|
|TI|jh_re_a_input|A|tutor GPT 입력 block/other/material 및 tie: low→medium|
|TS|jh_re_a_source|A|tutor 섹션 선택 호출: low→medium|
|TA|jh_re_a_answer|A|상속된 GPT 텍스트 생성 answer_effort: low→medium|

첫 비교에서는 timeout·토큰 한도·프롬프트·검색·MLP 임계값·투표 수를 고정한다. TA의 교재 선택/직접 발췌/별도 유해 요청 응대 생성은 기존 그대로다. 실제 호출 수로 변경 적용 범위를 확인한다. 전역 모듈 설정이나 기존 A/C 클래스는 바꾸지 않는다. 필요한 단계에서만 얕은 복사 객체와 LLM 어댑터를 사용하여 effort만 바꾸며 원래 client·slots를 공유한다. C의 요청별28초 예산과 취소 정책을 유지한다.

## 검증

- guard: 기존 공개 파생196, C/GI/GG/GS 동일 조건·64 HTTP workers/서버slots.
- tutor: 기존 공개 파생192, A/TI/TS/TA 동일 조건·64 workers/slots.
- 유망 후보는 반복하고, 반복에서도 개선되면 혼합388로 검증한다. 이번에는 반복 개선 후보가 없어 혼합 부하 단계로 올리지 않았다. medium에서 timeout/출력 토큰 소진이 생기면 별도 후보로만 예산 보완을 분리한다.
- 기존 scorer 그대로 사용. 점수·p50/p95/max·전체 수집 시간·endpoint별 내부 timeout/parse 오류·실제 effort 호출 수 기록.
- 개발 runner에서 완료 응답의 finish_reason과 토큰 사용량을 숫자로만 집계한다. 추론/답변 본문은 이 계측에 저장하지 않는다. 모델·프롬프트·timeout·토큰 한도 자체는 계측에서 수정하지 않는다.
- 공개 파생 개발 자료의 응답과 grader journal은 기존 캐시에 보존한다. 비공개 제출 본문 수집 기능을 추가하지 않는다. 공개 부모·학습 자료를 공유하므로 독립 holdout이나 실제 점수 추정으로 해석하지 않는다.

## 결과

1차 가드·튜터 비교8회, 총1,552응답 수집과 채점을 완료했다. 기존 A/C 실제 제출 후보는 변경하지 않았다.

|가드 후보|가드 점수|p95 초|medium 완료/토큰 소진|주요 관측|
|---|---:|---:|---:|---|
|C|94.64|9.887|0/0|low 완료694 중 length1|
|GI|93.08|15.713|237/43|입력 parse43, medium 호출250|
|GG|92.47|17.172|336/118|근거 ValueError118, ground Deadline5|
|GS|93.76|11.911|83/0|안전성 Deadline2|

고정 토큰 한도1500 안에서 추론 토큰과 JSON 출력이 경쟁한다. GG의 length118과 근거 파싱 오류118이 함께 관측돼, 추론 수준 자체의 효과와 출력 실패를 분리할 필요가 있다. 후속 비교에서 GI/GG의 토큰 한도만4000으로 늘려 보았으나, 아래의 제공 API 제약 때문에 중단했다. 기존1500 후보는 그대로 보존한다. 이 표는 공개 파생 개발 점수이며 실제 제출 점수와 직접 비교하지 않는다.


### 튜터 1차 결과

|후보|튜터 점수|수집 시간(초)|요청 p95(초)|medium 호출|내부 주요 실패|
|---|---:|---:|---:|---:|---|
|A|96.09|18.220|11.264|0|material parse1, context_policy timeout1|
|TI|94.27|23.818|16.164|262|length46, other timeout20, block timeout5, material timeout5|
|TS|95.31|19.868|12.063|27|source timeout7|
|TA|94.27|20.056|13.428|16|generate timeout10, 완료6 중 length1|

각192문항, 공식 형식 오류·judge 오류0. 내부 실패는 fallback 덕분에 공식 오류0이어도 점수에 영향을 줄 수 있다. TA는 답변 생성 경로16회만 변경하며 전체192개를 medium으로 바꾸는 것이 아니다. TS와 TA는 이미 호출 시간 제한이 문제여서 이번 토큰 예산 보완 대상으로 포함하지 않았다. 튜터 비교는 단회이며 확정 우열이 아니라 현재 예산에서의 채택 여부를 판단하는 개발 실험이다.

### 제공 API 제약 및 중단 실행

`re_guard2`는4000토큰 보완을 시도했으나 제공 프록시가 비스트리밍 `max_completion_tokens`를 **2000 이하**로 제한해 거절했다. 최소 진단 호출에서4000/3000/2048 모두 같은 명시적 제한 오류를 확인했다. 오류 메시지가 stream=true를 제안했지만 적용하지 않았으며, 비스트리밍을 유지했다. 부적합4000 설정의 등록 이름은 제거했고 실험 이력은 커밋 `f9e7de5`와 캐시에 남겼다. 상한 준수 수정은 `3a8d371`이다.

중단 신호가 전파되는 동안 GG4k/GS/GI4k196건씩이 수집됐고 C는28건만 저장됐다. 서버 종료를 확인한 후에만 다음 API 실험을 시작했다. 완전 수집3개의 가드 지표는 원래 score_guard로 오프라인 계산해 보존했다. 가드는 LLM judge를 사용하지 않으므로 이 진단 계산은 추가 API를 호출하지 않는다. C 부분 수집에는 점수를 만들지 않았다.

|후보|진단 가드 점수|상태|
|---|---:|---|
|GG4k|82.46|근거 API380회 BadRequestError, 설정 미지원|
|GS|93.99|지원되는 설정의 보조 반복, 같은 배치 C 미완료|
|GI4k|75.78|입력 API343회 BadRequestError, 설정 미지원|
|C|—|28/196 저장, 중단·미채점|

4000 후보는 낮은 점수 때문에 제외한 것이 아니라 API가 설정을 거절해 의도한 medium 처리가 실행되지 않았기 때문에 성능 순위에서 제외한다. GS93.99는 유효한 보조 결과로 따로 보존하되 완결된 짝비교와 구분한다. [중단 실행 전체 집계](reasoning_ablation_aborted.json).

### 허용 범위 내 후속 대조

`jh_re_c_input2k`, `jh_re_c_ground2k`: 각각 GI/GG에만 토큰 한도2000을 적용한다. C의 입력13초/교재16초/근거18초/안전성13초와 요청 전체28초는 그대로다. 입력용 어댑터가 tutor·안전성으로 새지 않고 남은 요청 시간으로 제한되는지 테스트했다. 기존 A/C·첫6후보는 유지했다. `re_guard3`의 순서는 GG2k → GS → GI2k → C이며 같은 공개 파생196문항·64동시 조건이다. GS와 C를 반복해 단회 변동도 관찰한다. 결과 수집·채점 완료.

|후보|가드 점수|수집 시간(초)|p95(초)|medium 완료/length|
|---|---:|---:|---:|---:|
|C|93.79|23.363|12.170|0/0|
|GI2k|93.24|29.367|16.716|211/13|
|GG2k|94.46|33.056|19.165|327/52|
|GS 반복|93.40|28.620|12.649|83/1|

GG2k는 GG1500의 length118→52로 줄였지만 ground Deadline은5→12로 증가했다. C와 GG2k의 grounded_accuracy/span_f1은 **동일한0.9765**다. 전체 가드+0.67은 목표로 바꾼 근거 단계의 집계 성능 개선에서 나온 것이 아니며 입력·안전성 판정의 변동을 포함한다. 첫 C94.64보다도 낮고 p95가 길어 단회 최고로 채택하지 않는다.

`re_guard4`는 C → `jh_re_c_groundlow2k`(low 유지·토큰만2000) → GG2k 순서로 반복했다. 프롬프트·timeout·입력·안전성 변경 없이 토큰과 effort 효과를 분리한다. 새 low 대조는 `4e91369`에 기록했다.

|마지막 대조|가드 점수|근거 정확도 / 구절 F1|수집 시간(초)|p95(초)|중요 내부 실패|
|---|---:|---:|---:|---:|---|
|C low1500|95.00|0.9765 / 0.9765|24.384|12.520|근거 ValueError1|
|근거 low2000|94.49|0.9529 / 0.9529|25.512|11.995|ground Deadline1|
|근거 medium2000|93.03|0.9529 / 0.9529|33.888|18.731|근거 length/ValueError45, ground Deadline18|

C와 GG2k를 모두 측정한 두 배치에서 평균 가드는 **C94.395 / GG2k93.745**, 배치별 p95 평균은 **12.345초 / 18.948초**다. 평균 차이는−0.65점/+6.603초로 medium을 채택하지 않는다. 반복 수2라 통계적 확정은 아니지만 운영상 교체할 증거가 없고 지연 부담이 명확하다. GG2k는 두 번 모두 목표 근거 지표가 C보다 높지 않았다.

GS는 완결 짝비교93.76/93.40, 중단 배치의 유효 보조 반복93.99였다. 세 번 모두 비교 대상 기존 C 관측 범위와 겹치거나 낮고, medium 안정적 개선은 없다. 입력 medium은1500/2000 모두 해당 배치 C보다 낮았다. 튜터 세 후보도 첫 비교에서 기준선보다 낮고 내부 실패가 증가해 추가 제출 우선 후보로 올리지 않는다.

### 해석과 다음 개선 방향

- reasoning effort는 모델 크기 변경이 아니다. 같은 제공 nano의 추론 예산만 바꿨다. 본 실험은 low→medium이며 high는 테스트하지 않았다.
- max_completion_tokens에는 추론 토큰도 포함된다. 현재 비스트리밍 상한2000 안에서 긴 추론이 JSON/답변 출력 공간을 소모했고, 완료 지연도 늘었다. 토큰을 늘리자 일부 length는 줄었지만 timeout이 증가했다.
- 공식 오류0은 내부 검사 성공100%를 뜻하지 않는다. fallback/투표 결과가 유효한 형식으로 반환돼도 출력 품질은 떨어질 수 있다.
- 다음 추론 실험을 한다면 모든 입력·근거 표를 medium으로 올리기보다 **low 표가 불일치하고 요청 예산이 충분할 때 한 번만 medium으로 재검사**하는 별도 후보가 더 합리적이다. 아직 구현·검증하지 않았으며 이득을 보장하지 않는다.
- 현재 최우선은 변경하지 않은 A/C를 실제 제출에서 번갈아 반복하는 것이다. 공개 파생·학습 부모 공유 자료의 고득점은 비공개 성능을 보장하지 않고, 로컬 judge도 실제 judge와 다를 수 있다.

표의 수집 시간은 해당192/196요청을64동시로 수집한 벽시계 시간이고, p95는 개별 HTTP 요청의95백분위다. 실제 제출600요청의 Duration과 직접 대응시키지 않는다. guard-only 실행의 total(튜터0에 따른 절반 총점)은 비교 지표로 사용하지 않았다.

최종 검증: 전체861테스트 통과(6.015초), `git diff 67bd16d -- core grader harness/score_local.py server.py SPEC.md public kb` 변경 없음. 구현 커밋 `8ac7ffa`. 실험 중 보호 파일과 파이프라인 해시를 고정하며 수집·judge·후보 간 API 부하를 겹치지 않는다.


## 재현 명령

프로젝트 루트에서 기존 `.env`와 `.venv`를 사용한다. 아래 명령을 **순서대로** 실행한다. 같은 키로 실제 제출 또는 다른 API 부하 실험을 동시에 진행하지 않는다. 이름이 이미 있으면 새 이름으로 바꾼다. 기본 모델과 임베딩 설정은 바꾸지 않는다.

```bash
.venv/bin/python experiments/jh_guard_fusion_compare.py --name re_guard1_repro --dataset paraphrase --order np_timeouts re_c_input re_c_ground re_c_safety
.venv/bin/python experiments/jh_tutor_mlp_compare.py --name re_tutor1_repro --order cm_char_mlp50 re_a_input re_a_source re_a_answer
.venv/bin/python experiments/jh_guard_fusion_compare.py --name re_guard3_repro --dataset paraphrase --order re_c_ground2k re_c_safety re_c_input2k np_timeouts
.venv/bin/python experiments/jh_guard_fusion_compare.py --name re_guard4_repro --dataset paraphrase --order np_timeouts re_c_groundlow2k re_c_ground2k
.venv/bin/python experiments/jh_reasoning_report.py --guard-runs re_guard1_repro re_guard3_repro re_guard4_repro --tutor-runs re_tutor1_repro --output /tmp/jh_reasoning_repro.json
```

비교 자료는 기존 gitignore 개발 캐시의 `jh_augment_public2_1/paraphrases.jsonl`을 사용한다. 새 체크아웃에는 캐시가 포함되지 않으므로 이 명령은 해당 공개 파생 캐시가 있는 개발 환경용이다. 런박스의 실제 제출은 `experiments/jh_server.py --variant <후보>`로 실행하며 파생 자료·정답은 필요 없다. 기존 A/C 실행 방법은 [반복 제출 안내](AC_REPEAT_SUBMISSION.md)를 따른다.
