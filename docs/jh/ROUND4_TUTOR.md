# 튜터 개선 재개 — 잔여40%까지

2026-11-17 사용자 재개 지시에 따라 시작했다. 시작 시 계정 used43%/remaining57%,
새 실험 중단은 remaining40%다. 이전60% 중단 기록은 역사로 보존하고 로컬 중단 표식만 해제했다.
목표는 정직한 실제 제출 점수 향상이며100점을 지향하되 달성이나 재현을 보장하지 않는다.

## 기준선과 제약

- 기준선: `jh_r3_selected_focused`, embedding env, request slots64.
- 사용자 실제 제출: 전체83.99/84.68, tutor79.17/81.58, guard88.81/87.78.
- guard ContextFast와 주최측 평가기·public·kb·core는 유지한다.
- 비공개 질문·응답을 수집하거나 학습/규칙/캐시에 사용하지 않는다.
- 제공 모델만 호출하며, 공개/교재 진단의 금값은 runtime 선택에 전달하지 않는다.
- 모든 성공·실패·부정 결과를 보존한다. 수집과 원본 채점을 분리하고 완료 판정은 재채점하지 않는다.
- 실제 제출은 사용자 보고로만 기록한다. 공개 점수와 비공개 점수를 섞어 평균 내지 않는다.
- SPEC의 설명상 로컬 판정 모델은 실제 채점 판정 모델과 다르다. 로컬의 작은 판정 변동을
  실제 오답 개선으로 과장하지 않고 의미적 정확성도 함께 검토한다.

## 시작 가설

1. **학습 분류용 질의의 보호 정규화.** 정상 질문에 붙은 standaloneㅋㅋ/ㅠㅠ 때문에
   고신뢰 gate가 빠지는 현상을 줄일 수 있는지 본다. tutor category 모델의 입력만 바꾸며
   원 대화·LLM payload·검색·section/harmful 모델·guard는 유지한다. 코드/따옴표/URL 보호와
   정서표현만 있는 입력 보존을 검증한 뒤 원본과 잡음 진단을 비교한다.
2. **오답 원인 감사.** selected_focused의 공개 파생/교재 진단에서 올바른 출처의 facts 실패와
   잘못된 출처·action 오류·판정 변동을 구분한다. 계산/비교 설명 부족이라는 가설은 감사 결과를
   보고 채택한다. 기존 fullsection에 답이 명시된 사례에 무조건 생성 단계를 더하지 않는다.
3. **불확실한 출처 후보의 구성.** 이전 최대8개 확대는 성능이 나빠졌다. 후보 개수6은 유지하며,
   전체 대화 검색에 가려진 마지막 질문의 의미 검색이 도움이 되는지 검토한다. 정답을 아는 상태로
   출처를 넣지 않고 제공 임베딩의 자체 검색 결과만 사용한다. 시간 예산과 fallback을 먼저 설계한다.

## 운영 기록

초기 코드·별도 모의 테스트와 CPU 감사를 준비 중이다. 아직 R4 API 수집은 시작하지 않았다.
새 후보는 등록·검증·커밋 뒤에 수집한다. 기존 제출 후보와 기본값을 자동 교체하지 않는다.
상태는 [ROUND4_STATE.json](ROUND4_STATE.json), 결과 집계는 `round4_results.json`에 기록한다.

## R4-01 사전 등록

- 기존 실패 감사: 저장 점수1,251개 복원 일치. 자체 answer+인용1개855회 중 facts/faithful 실패33회는 잘못된 출처21, 비canonical 동일 사실3, 올바른 교재 사실 보존9였다. 올바른 교재9회 모두 같은 judge payload가 다른 완료 실행에서 통과했다. 따라서 계산 설명 생성보다 출처 후보를 먼저 실험한다. 전체 대상에는 별도 action 오류58회가 있다. 캐시 감사 원본 `experiments/.cache/jh_r4_failure_audit/AUDIT.md`.
- 정규화: `jh_r4_normalized_tutor`. 원본 tutor200개와 normal/PII에 고정 prefix/suffix를 붙인 잡음222개를 기준선과 각1회 비교한다. 잡음은 합성 스트레스 진단이며 실제 분포나 정서 recall 개선으로 해석하지 않는다. 원본은 baseline→candidate, 잡음은 candidate→baseline 순서.
- 마지막 질문 의미검색: `jh_r4_last_dense`. 기존출처후보 첫2개 다음에 마지막질문 dense 상위2개를 넣고 중복제거, 최대6개. 불확실한 정상/PII만 호출하며 3초 제한, 잔여4초미만 생략, 실패시 기존풀. topic/followup 각각114개를 기준선과 비교하고 이득이 있으면 primary/alternative로 회귀를 확인한다.
- 두 후보 모두 독립 variant이며 추천/기본값을 바꾸지 않는다. collection workers64/slots64/env, judge32, 수집과 판정 직렬.

## R4-01/02 결과 — 8개 실행 완료

모든 수집 코드 `ab39eaf`, clean, env/64/32. tutor-only 점수이며 total은 guard 미수집으로 절반이므로 비교에 쓰지 않는다. HTTP/형식/원본judge 오류 모두0. 수집 초는 로컬 부하 시간이며 실제 제출600개 duration과 비교할 수 없다.

| 자료 | 기준선 tutor / 수집초 | 후보 tutor / 수집초 | 차이 |
|---|---:|---:|---:|
| 원본 tutor200 |98.12 /12.75|정규화97.00 /12.04|-1.12|
| normal/PII 잡음222 |99.66 /11.90|정규화98.76 /12.78|-0.90|
| 주제전환114 |89.69 /25.20|last-dense89.25 /24.52|-0.44|
| 후속질문114 |92.11 /28.76|last-dense91.89 /27.54|-0.22|

- 정규화는 원본 public200/primary192/alternative181 입력을 하나도 바꾸지 않는다. 원본 점수 차이를 정규화 효과라고 해석할 수 없다. 원본 같은 응답2개에서도 facts/faithful 판정이 달랐고, 다른 action/응답 변동도 있었다.
- 잡음 진단 개선 없음. 기준선 action221/222→후보219/222, citation 평균1→.9865. 관측쌍111 source cluster bootstrap95 delta[-2.365,.338]. 정규화는 보류, section 정규화 확장도 CPU 근거 없어 하지 않는다.
- topic last-dense는 action105/114동일, citation91/114→87/114, facts102/114→101/114. 일부출처를복구했지만다른정답출처를잃었다. paired delta CI[-4.167,3.289]. 후속은CI[-3.947,3.289]. 이는관측된응답쌍만의기술통계이며반복/비공개불확실성을추정하지않는다.
- 내부오류요청 topic기준선10/후보4, followup40/10. 실제HTTP실패는없고fallback을포함한점수다. 부하변동이커서수집시간차이를후보의고유속도이득으로단정하지않는다.
- baseline원본후 첫후보는 driver의 동일부모폴더 생성충돌로 API전중단했다. `r4_01_original_norm`으로분리해정상수집했고 완료된기준선을재실행하지않았다. 해당오류로그보존.
- 결정: 기존 `jh_r3_selected_focused` 추천유지. 고정 last-dense와정규화는개선확인못함. 새로운실제제출점수는없다.

## R4-03/04 사전 등록 — 사용자 조건부 검색 아이디어 반영

사용자는 BM25중심, 임베딩으로누락근거보완,질의별비중조절을제안했다. RRF재시도보다fixed6 후보에서추가dense의적용조건을제어한다.

1. `jh_r4_conditional_dense`: KB 제목·인라인코드의 한글4자이상/영문identifier3자이상 anchor중 전체48section본문의최대2곳에만있는329개를사용. 마지막질문에서해당anchor가일치하고 distinctsection BM25 1위가2위의1.5배이상이면추가last-dense를생략해기존pool유지. 그렇지않으면last-dense와동일. 기준과anchor는gold없이정의했고 threshold튜닝안함. 정책/action을바꾸지않음. CPU strong비율 public5%,primary5.2%,alternative5%,topic8.8%,followup0%. gold있는35행top1모두정답이지만19행은기존sectiongate와중복, 실제효과범위는작다. topic은앞선base/dense와조건부를비교하고 primary는base/dense/conditional각1회비교. 효과가있어야반복검증한다.
2. `jh_r4_structured_block`: source/투표/시간/토큰수는그대로. tutor fallback BLOCK의기존규칙을모두보존하고 speech_act/target/request_span을category전에짧게구조화한다. 별도호출없음. guard는기존프롬프트. calibrated의기존프롬프트선택을기본동작동일hook으로바꿔전역변경을피한다. primary192,alternative181에서각기준선과비교. 보조필드는강제판정에쓰지않는다. 고신뢰ML우회오류에는효과없으며토큰/파싱실패위험을측정한다.

캐시action감사58회: 같은오분류반복24,같은입력action변동20,단회14. action은원본평가기의문자열비교라judge변동이아니다. 모델/검색/내부API실패원인은개별확정불가. CPU56회는fallback,학업부정2회는normal .9972 gate우회. 감사자체는API없음.

두후보등록후전체390개테스트통과. conditional의gold감사재현테스트는캐시에만두고, 저장소테스트는합성입력으로검증해캐시없이재현된다. 평가기·core·server·SPEC·public·KB는원본동일.

## R4-03/04 결과 — 누적15개 HTTP 실행 완료

새7개수집 `d3c35a9` clean, env/64/32, HTTP/형식/judge오류0. primary 기준선은두가설공유이며각후보와동일commit/자료다.

| 자료 | 기준선 | 고정dense | 조건부dense | structured BLOCK |
|---|---:|---:|---:|---:|
| primary192 |94.53|94.53|95.70|94.53|
| topic114 |89.69(R4-02)|89.25(R4-02)|88.60|—|
| alternative181 |91.30|—|—|93.23|

조건부primary +1.17을검색개선으로채택하지않는다. 완전히동일응답153개에서원본판정점수순차이+300, 전체평균기여+1.5625점으로전체차이+1.171875보다크다. 같은응답의판정변동만으로관측상승을설명할수있다. 실제로출처가바뀐경우에이득과손실이있으며topic도하락. raw score는변경하지않는다. primary관측paired CI[-1.953,4.297]이며반복실행불확실성을포함하지않는다.

structured BLOCK도원본튜터action primary187/192→183/192, alternative170/181→172/181로일관되지않다. 직접목표였던일부정상질문은회복했지만alternative 실제역할변경요청1개는refuse→answer로악화. primary 총점동점은faithful/complied 등의변동과상쇄된값이다. 따라서현재추천을교체하지않는다. 추가실제제출은하지않았다.

## 추가 CPU 감사와 팀원 갱신

- early last-BM25 삽입: 정상abstain15회에서새정답section구제0,새정답구절추가가능1. OOD→answer19회는lasttop1이이미fullBM25 top10에모두있음. `material_max_chunks=None`이므로자료잘림해소가아닌순서가설에가깝다. 구현/API보류.
- OVR unique-high category: 기존375group5fold, fold내어휘학습, C1000/.95고정. 기존multinomial OOF297/400→OVR293/400, tutor gate82/7오류→90/9오류. 고정fallback관측재생tutoraction175→172. 현재HTTP점수와구분하며API보류.
- 팀원원격갱신: 승원c164d5e그대로, Peace6398f5e→3f00c25. 개인정보범주보정완료결과를guard deadline내보존하는변경이며현재tutor개선직접근거가아니다. guard유지지침에따라가져오지않았다.

## R4-05/06 사전 등록

1. `jh_r4_source_conversation_last`: source selector의입력JSON 최상위키순서만 conversation→materials에서materials→conversation으로변경. 후보6개순서/내용,system,low/2000/8초,분류/guard/답변동일. 긴교재뒤에대화를놓는것이대상혼동을줄이는지topic114와primary192에서각새기준선과비교한다. 기존payload기본hook은바이트동일성을테스트했다.
2. 새로운KB-only 관계진단: `jh_generate_grounded_tasks.py`가각section의완전한연속문단8..350자를사전고정한다. 제공nano에공개질문/labels/후보응답을주지않고question+answer_fact한쌍만요청한다. quote는코드가선정원문을유지한다. 문단선정48/48,80..283자. 기존별도제공모델review와직접사전정제를통과해야채점한다. 이전관계생성실패41회의정확한원인은확정하지않았으며, 이번방식은복사부담을줄이는다른생성설계다. 관계·조건에명시된사실만묻고새계산능력진단이라고주장하지않는다.

전체406개테스트통과,보호경로원본동일. 아직새진단/출처순서후보API는시작전이다.

## R4-05 결과와 새 관계 진단 정제

수집코드 `9501351` clean. 출처payload순서후보는topic89.69→88.38,primary95.05→91.67로두자료모두낮았다. topic action106/114동일인상태에서citation89→86, facts100→97. primary citation114→110, facts115→112. primary의complied판정도4/44→9/44로나빠졌으나이분기는순서후보가바꾸지않으므로전부인과효과로돌리지않는다. 채택안함. 누적HTTP19개모두공식형식/judge/HTTP오류0이며내부오류는별도기록한다.

`r4_grounded1`은제공nano로48/48문항형식통과,생성실패0. 별도제공모델review48개중41개valid,오류0. 이후후보응답/점수를보지않은직접검토에서자동accepted중25개를추가제외해16개로고정했다. 질문에정답전체노출,없는조건/문장생성,question/fact불일치,과정규칙일반화,복수section의동일사실등이주된이유다. 정답/질문은고치지않았다.

수동검토상전체48개는제외30/명확유지6/표현주의유지12였고,자동검토와교집합은16개다. 일부문항의인위적표현·자명한관계·모듈명생략한계는남는다. 독립비공개검증또는계산능력진단으로해석하지않는다. 생성자료·원본리뷰·48개수동판정·추가제외권고·최종curation은 `experiments/.cache/jh_kb_review_r4_grounded1/`과관련cache에보존했다. `curation.json`은dataset/review/manual SHA256에묶었다. 아직이16개후보응답은수집하지않았다.

## R4-07 사전 등록 — 튜터 material 시간 배분

최근baselinefollowup materialTimeout39/114등을관측했다. `jh_r4_material_budget`는튜터material단일호출의최대한도10→13초,prepare14→16초만바꾼다. request별shallowcopy로prepare한도를설정하고guard/누락identity는14초그대로다. 전체26초/source8초,프롬프트/토큰수/투표/분류threshold/호출수/재시도정책은동일하다. material도남은전체시간보다길게호출하지않는다. 늘어난prepare시간이source시간을줄일수있으므로점수·내부오류·tail시간을함께본다.

followup114와primary192를각기준선과비교하고새관계16개에기준선을수집한다. source순서/조건부dense/정규화와합치지않는독립후보다. 전체418개테스트통과,보호경로원본동일.
