# R6 — 원격 팀원 작업 재검토와 점수 개선

2026-11-18 KST. 사용자 재개 지시: 사용량 **잔여15%**까지 실험 → 분석 → 다음 실험을 반복한다.
시작 사용률63%, 잔여37%. 잔여15% 이하를 관측하면 새 수집을 시작하지 않고 진행 중인 채점과
코드·결과·인수인계를 정리해 커밋한다. 이전 R4/R5 종료 기록은 역사로 보존한다.
사용량은 계정의 다른 작업과 공유하며 각 실험 묶음 시작 전/후 확인한다.

## 브랜치와 출처

`git fetch origin --prune` 후 `git pull --ff-only origin experiment/kimjaehak`: 이미 최신 `b199844`.
각 브랜치는 개인 실험 공간이다. 다른 팀원 브랜치를 병합·수정·푸시하지 않고 이번 작업은
`experiment/kimjaehak`에만 기록한다. 팀원 점수는 각자의 평가 조건이며 우리 점수와 단순 합산하지 않는다.

| 원격 브랜치 | 확인 HEAD | 확인한 변화 / 판단 |
|---|---|---|
| experiment/seungwon | f3e320b | vx 확신도 하한·적응형 재확인·material section 출력 제거·v8 span 다듬기. 낮은 확신 제거는 오차단을 줄이지만 튜터 손실과 반복 편차가 있어 기본 적용하지 않는다. material 불필요 필드 제거를 독립 검증한다. medium·재시도는 팀원 실험의 시간 실패를 고려해 후순위. |
| experiment/bsuk1 | de664f2 | 기존 expert/bsuk1 대신 새 이름. Step11의 직교 입력·적응 투표·focused 출력 상당 부분은 우리 코드와 공통이다. 짧은 Q/A 근거를 material 판정에 공급하는 방식과 마지막 질의 우대 검색이 새로운 후보. Kiwi/내용어 bigram/RRF의 공개119 top6=100%는 독립 검증 전 수치다. Step11 실제 제출은 아직 기록 없음. |
| export/peace | f7e3940 | 요청 본문 없는 오류/시간 로그 보존. 내용 기반 채점 자료를 수집하는 기능으로 옮기지 않는다. |
| export/peace-jh-full-no-trained | 2f51c1f | 우리 b199844 기반에서 학습형 모델을 끈 별도 실험. 공개 tutor 평균86.50·guard90.52는 원본과 재대조하지 않은 세 번 평가다. 새 parallel guard는 아직 결과 미확인. 학습 경로 제외 전체를 복제하지 않고 tutor의 고신뢰 출처 재확인으로 좁힌다. |
| dev / main | 429330c / 2043463 | 새 변경 없음. |

팀원 근거: 승원 `docs/sw/05_로컬_점수표.md`, `pipelines/vx/{pipeline,prompts}.py`;
동진 `docs/dj/HANDOFF.md`, `app/kb_index.py`, `app/orthogonal_guard.py`;
Peace `docs/PEACE_FULL_PORT{,_RESULTS}.md`, `pipelines/kimjaehak/parallel_guard.py`.

## 유지 조건

- 기존 실제 제출 추천 `jh_r4_material_budget`, 백업 focused, R5 dual6/all을 보존한다.
- 주최측 `grader`, `harness/score_local.py`, `core`, `server.py`, `SPEC`, `public`, `kb` 원본 유지.
- 우선 가드를 고정하고 튜터만 변경한다. 비스트리밍, 전체 deadline26초, slots64, embedding env.
- 제공 API만 사용. 비공개 입력/답변을 저장하거나 정답을 런타임에 전달하지 않는다.
- 개발 자료는 공개 파생 primary192 / alternative181, 교재 검토 질문의 topic114 / followup114.
  모두 개발에 사용한 자료이며 독립 비공개 점수가 아니다. 우수 후보는 새 기준선·역순 반복·다른 자료로 확인한다.
- 코드 커밋 후 깨끗한 트리에서 실행. 수집과 원본 평가기 채점을 겹치지 않는다. 단회 최고점으로 채택하지 않는다.

## 1차 후보 — 결과 확인 전 계획

| 후보 | 비교 기준 | 바뀌는 것 |
|---|---|---|
| jh_r6_short_material | jh_r5_dual_all | 짧은 dense 검색 근거 중 기존 material에 포함되지 않은 최대4청크를 material 판정 입력에 추가. 기존 근거는 모두 유지. state.chunks와 출처 후보 구성·가드·LLM 횟수는 유지. |
| jh_r6_material_nosection | jh_r4_material_budget | tutor material 프롬프트에서 section 설명·출력 필드만 제거. asked/quote/confidence, 예산, 검색, 답변은 유지. |
| jh_r6_context_verify | jh_r4_material_budget | 다중 발화 normal/PII tutor에서 학습형 출처를 후보로 보존하면서 기존 source LLM으로 재확인. 단일 발화·다른 정책·가드는 유지. |

단위 테스트는 추가근거 중복/한도·원본 보존, tutor/guard 분리, 프롬프트 필드 제거,
문맥 재확인 조건·비스트리밍·기존 시간 제한을 검사한다.
첫 묶음은 topic114에서 기준선2개와 신규3개를 비교하고, primary192 및 followup114에서 유망 후보를 검증한다.
결과에 따라 마지막 질문 우대 검색·범주 판정·근거 선택을 후속 실험한다.

## 실행 결과와 다음 판단

실행별 commit·데이터/평가기 해시·설정·오류·시간·점수는 로컬 캐시와 `round6_results.json`에 보존한다. 아래는 완료된 비교이며 후속 실험을 계속한다.

## 사용자 추가 조건과 CPU 진단

- 추가 지시: `jh_r3_selected_focused`를 기준선에 포함한다. 사용자 확인상 R5 dual_all/dual6의 새 실제 제출 결과는 아직 없다.
- 실제 제출 추천은 기존 기록을 유지한다. 개인 브랜치 간 통째 병합 없이 필요한 아이디어만 옮긴다.
- 첫 API 묶음은 topic114에서 material_budget → dual_all → focused → short_material → nosection → context_verify 순서다.
- 새 짧은 근거 추가는 같은 공개 질의 벡터를 재생한 CPU 진단에서 정답 support_quote 포함을 **102/114→104/114**로 늘렸다.
  평균 추가량은0.56청크·123자다. 이 수치는 검색 근거 포함이고 튜터 점수가 아니다.

동진의 내용어 바이그램 및 마지막 질문 우대를 동일219개 청크·같은 BM25 공식으로 분리했다.
8개 개발 자료718질의에 대해 `jh_round6_lexical_screen.py`로 CPU 진단했고 전 결과가 재현됐다.
질문 정규화는 이718개 자료에서 문자열 변화가 없었고, KB에서 원문/정규화 형태소 토큰 차이도0이었다.
Kiwi 내장 의미 벡터는 사용하지 않는다.

| 자료 | 기존 top1 / top3 / top6 | Kiwi+내용어 bigram (전체 대화) |
|---|---|---|
| public119 | 108 / 114 / 116 | 108 / 117 / 117 |
| primary119 | 109 / 113 / 117 | 110 / 118 / 119 |
| topic114 | 4 / 86 / 100 | 2 / 89 / 103 |
| followup114 | 85 / 106 / 111 | 85 / 103 / 111 |

마지막 질문 가중0.75는 topic top1을 크게 높였지만 followup top1을85→20(raw 혼합) 또는5(점수 정규화)로
낮췄다. 출처 모델 confidence≥.25/.50 조건이나 지시어 보존 규칙은 후속 질문 손실을 막았지만
public/primary에서도 top1이 하락했다. 이 가중 검색은 현재 HTTP 후보에 넣지 않는다.
조건부 규칙은 개발 템플릿에 맞을 가능성도 있어 CPU 숫자를 실제 제출 우위로 읽지 않는다.

## 2차 후보 사전 등록

- `jh_r6_kiwi_bigram`: material_budget의 **튜터 최초 BM25 검색만** Kiwi+내용어 바이그램으로 교체한다.
  긴 임베딩·원 대화·source selector·focused 원문 재정렬·가드는 유지한다. 팀원 고유명사 사전은
  `de664f2`의 교재 어휘23개만 사용하며 번역 alias·RRF·예시 검색은 가져오지 않는다.
  선택한 후보만 추가 의존성 `requirements-jh-kiwi.txt`의 kiwipiepy0.24.1이 필요하다.
- `jh_r6_resolved_focused`: 과거 검증한 minimal 질문 해소기를 material_budget의 focused 출처 선택에
  결합한다. 충돌하는 다중 발화에서만4초 이내로 검색 질문을 만들고, 실패하면 병렬 준비한 기존 검색으로 복귀한다.
  판정/답변에 전달하는 원 대화는 바꾸지 않는다. 기존 R3은 ContextFast 생성 경로였으므로 이번 조합은 새 비교다.
- 첫 비교는 topic114에서 시행하고 primary192·followup114에 확장한다. 필요한 기준선은 같은 시점에 다시 실행한다.
- 구현 회귀 테스트 추가5개 및 전체476개 통과. 기존 첫 묶음 후보 코드는 수집 후에도 변경하지 않았다.

## 1차 완료 — topic114 × 6후보

수집 commit `482fdd1`, 전체 clean. HTTP684요청, HTTP/형식/판정 오류0. 튜터 전용 개발 점수다.

| 후보 | tutor | 수집 초 | 내부 실패 요청 | normal에서 answer / 정답 인용 |
|---|---:|---:|---:|---|
| jh_r4_material_budget | 88.16 | 29.17 | 14 | 106 / 86 |
| jh_r5_dual_all | 90.13 | 30.25 | 3 | 105 / 90 |
| jh_r3_selected_focused | 90.79 | 25.33 | 8 | 107 / 90 |
| jh_r6_short_material | 89.04 | 29.58 | 10 | 109 / 87 |
| jh_r6_material_nosection | 86.62 | 27.14 | 4 | 102 / 84 |
| jh_r6_context_verify | 90.79 | 31.03 | 5 | 107 / 91 |

- ShortMaterial은 기준 dual_all보다1.09점 낮았다. 근거 포함은 늘었지만 전체 점수 개선은 확인되지 않았다.
- NoSection은 material_budget보다1.54점 낮았다. 응답/시간 실패 감소가 점수 개선으로 이어지지 않았다.
- ContextVerify는 material_budget보다2.63점 높았지만 **topic114·followup114에서 고신뢰 출처 게이트 대상이0개**였다. 이번 자료에서는 바뀐 경로가 작동하지 않았으므로 상승을 개선 효과로 해석하지 않는다. primary의 대상80개는 출처 예측이80개 모두 정답이어서 재확인 이득 근거도 없다. 후순위로 둔다.
- focused가90.79점으로 이번 대조에서 높았다. 단회 값이며 기존 제출 추천을 교체하지 않는다.
- 현재 사용률65%(잔여35%). 다음은 실제 검색이 바뀌는 KiwiBigram 및 ResolvedFocused를 topic/followup과primary로 비교한다.

## 출처 선택 진단과 3차 후보

공개 topic 질의의 캐시 벡터와 당시 검색 설정으로 후보 풀을 재구성했다. 실제 런타임 trace가 아니며,
임베딩 호출 실패·캐시 반올림 등에 따라 실제 후보와 차이가 있을 수 있다.

| 후보 | 정답 포함 / 114 | 오답 인용 | 정답 포함인데 오답 인용 |
|---|---:|---:|---:|
| material_budget | 104 | 20 | 13 |
| dual_all | 110 | 15 | 14 |
| focused | 104 | 17 | 9 |
| short_material | 110 | 22 | 20 |
| KiwiBigram | 110 | 15 | 12 |

따라서 후보 수를 늘리는 것만으로 해결되지 않는 출처 선택 문제가 남는다.
다음 두 후보는 material_budget 기준으로 검색·판정·최종 인용·가드를 유지하고 선택 단계만 변경한다.

- `jh_r6_evidence_first`: source LLM에 보내는 각 섹션의 문단을 기존 focused 방식으로 재정렬한다.
  섹션 전체 내용·코드·표를 보존한다. 최종 인용 답변과 동일한 관련 문단 우선 배치를 선택 입력에도 적용한다.
- `jh_r6_source_vote3`: 동일 source 선택을 3회 병렬 호출하고 유효한 2표가 일치하면 나머지를 취소한다.
  전체 선택 단계는 기존8초 한도를 공유하며 요청26초 제한도 유지한다. 일부 실패 시 유효표를 사용하고,
  3표가 서로 다르면 호출 생성 순서로 우선한다. 유효표가 없으면 기존 출처로 fallback한다.
  비스트리밍이며 가드 경로는 변경하지 않는다. 호출량 증가에 따른 동시성64 환경의 지연·실패를 함께 측정한다.
- 회귀 테스트483개 통과: 조기 합의 취소, 외부 취소 후 자식 작업 회수, 부분 실패, 동률, 원문·상태 보존 포함.

측정 주의: 01·06·09 일부 수집 중 CPU 진단을 병행했고, 10 수집 끝부분과 07 채점 시작이 잠시 겹쳤다.
이 실행들의 시간 차이를 알고리즘 속도 개선으로 단정하지 않는다. 다음 수집은 CPU 진단과 채점 모두
종료된 뒤 순차 실행한다. 응답을 버리거나 유리한 결과만 선택하지 않고 모두 기록한다.

## 2차 완료 — topic 반복 및 검색 변경

| 후보 | tutor | 수집 초 | 내부 실패 요청 |
|---|---:|---:|---:|
| KiwiBigram | 88.60 | 25.58 | 11 |
| ResolvedFocused | 92.11 | 27.76 | 8 |
| focused 반복 | 87.94 | 31.25 | 20 |
| material_budget 반복 | 88.60 | 27.10 | 4 |

누적1140요청, HTTP/형식/판정 오류0. focused 2회 평균89.365, material_budget 2회 평균88.38.
ResolvedFocused는 이 자료에서 유망하지만 단회 결과다. 다른 자료에서도 확인한다.
Kiwi는 검색 후보의 정답 섹션 포함을 늘렸지만 점수에서 뚜렷한 이득이 없었다. primary에서 추가 검증한다.

출처 진단의 '오답 인용'은 **개발 라벨과 섹션 ID가 불일치**한다는 뜻이다. 실제 사례에서 체크카드 나이·한도,
통장 이체 한도·예금 가입금액 등 같은 사실이 여러 섹션에 중복되어 있었다. 일부는 사실 답변이 맞아도
단일 evidence 라벨과 달라 감점된다. 이를 전부 사실 오류로 해석하지 않는다. 기존 평가기·라벨은 수정하지 않는다.
공개 KB로 만든 합성 topic 진단에 특화된 순위를 실제 제출 우위로 단정하지 않는다.

다음 수집: source evidence/vote3의 topic 비교, ResolvedFocused·material_budget·focused의
followup114 및 primary192 비교, Kiwi primary192 비교. 순차 API 수집 후 원본 채점기로 채점한다.

## 공개 오답 수동 검토와 다음 후보

보조 에이전트가 API 없이 원문과 캐시를 대조했다. 08의 인용 ID 불일치13건 중10건은 다른 섹션에도
같은 금융 사실이 있고,3건은 선택한 섹션의 대상·세부 근거가 부족했다. 기준선01/03/10의 실제 근거
불충분 사례는11/6/11건이었다. '남은 돈' 질문은 대상 자체가 생략되어 있어 합성 질문의 모호성도 있다.
카드 연령 한 건에서는 완전히 동일한 facts 채점 payload가 실행별0→1→0→1로 흔들렸다.
캐시 재생으로12개 실행의 항목별 채점 구성요소를 확인했고 모호한 중복 key/재계산 불일치는 없었다.
실행 간 같은 payload의 판정 변화는 별개 문제이며 기존 점수를 수정하지 않는다.

- `jh_r6_resolved_source_hint`: 성공한 재작성 검색의 query를 request-local 속성에만 보존하고 source
  선택 JSON에 보조 필드로 전달한다. 원 대화가 우선이며 재작성은 근거/명령이 아니라는 지시를 붙인다.
  재작성 실패·기존 검색 복귀 시 필드를 넣지 않는다. 추가 API 호출 없이 기존4/8/26초 예산을 유지한다.
  material 판정 입력은 기존대로 둔다. 사용자 응답/서버 로그에는 query를 내보내지 않는다.
- `jh_r6_computable_material`: tutor material에서 교재 코드·표·수식의 명시된 조건으로 결정되는
  결과도 해당 원문을 quote로 사용할 수 있다고 설명한다. 계산값을 quote로 만들거나 누락된 조건을
  추측하는 것은 금지한다. 원래 값 계산 규칙/기존 출력형식 boundary와 구분되는 좁은 가설이다.
- `jh_r6_topic_focused`: 기존 R3 full/last 선택기를 MaterialBudget에 결합했다. R3에서 topic 성적이
  약했으므로 구현은 보존하되 위 두 후보보다 후순위다. 기존 가설을 새 구조에서 재검증하는 것이지
  처음 제안한 방법은 아니다.
- 전체490테스트 통과. 기존 평가기·KB·public·core·server diff 없음. 새 후보는 추천 교체 전이다.

11 evidence_first=89.69,12 source_vote3=88.82(topic114). source_vote3의 source 오류는0이었지만
선택 이전 action 차이가 있어 전체 점수만으로 출처 다수결의 인과 효과를 확정할 수 없다.
후속 질문13 resolved=86.18,14 material_budget=86.62,15 focused=89.69.
resolved의 수집34.56초/내부실패15는 기준23.90초/2, focused26.71초/4보다 높았다.
주제 전환의 이득이 후속 질문으로 그대로 이어지지 않았다. primary192 비교 채점을 계속한다.

## 19개 HTTP 실행 완료

누적2478요청(튜터 전용), HTTP/형식/판정 오류0. 실행별 전체 결과는 round6_results.json.

| primary192 후보 | tutor | 수집 초 | 내부 실패 요청 |
|---|---:|---:|---:|
| material_budget | 95.83 | 21.31 | 2 |
| resolved_focused | 94.92 | 24.41 | 4 |
| selected_focused | 93.75 | 16.82 | 2 |
| kiwi_bigram | 93.23 | 18.31 | 2 |

Kiwi의 형태소 검색 이득은 이번 전체 튜터 비교로 이어지지 않았다. 후순위로 둔다.
ResolvedFocused는 topic92.11이 높지만 followup86.18/primary94.92에서 일관된 우위가 없다.
공식 제출 추천 material_budget 및 비교·백업 focused를 유지한다. 새 후보의 공식 제출은 아직 없다.

### 남는 source 시간 활용 가설

- `jh_r6_source_budget12`: source의 최대8초를12초로 확대하되 low/2000토큰과 기존 요청26초 제한을 유지한다.
  실제 호출 timeout은 남은 요청시간에서0.5초를 뺀 값으로 계속 제한한다. material·guard는 그대로다.
- `jh_r6_source_reason12`: 위12초 후보에서 source만 medium/3000토큰으로 바꾼다.
  추론량과 토큰 상한을 함께 바꾸는 후보이므로 개선 시 어느 요인인지 후속 분리한다.
  source의 근거 판단을 더 기다리는 이득과 동시64 부하의 실패 증가를 함께 본다.
- 선택 시간이 요청 잔여 시간으로 제한되는 테스트 포함492개 통과.

모든 실제 수집은 다음 clean commit에서 시작하며 공개 추적 도구는 HTTP 성능 비교와 별도 실행한다.
현재 사용률68%(잔여32%). 중단 기준은 여전히 잔여15%이며 실험은 진행 중이다.

## 정상 학습 질문 gate의 좁은 완화

primary16–19에서 일상적 공부 걱정이4회 모두 escalate로 잘못 분류됐다. 공개 학습 분류기 자체는
normal=.944385였고 기존.95보다 낮아 LLM fallback으로 넘어간 경우였다. learned emotional 오판이
아니므로 emotional gate 재확인은 이 오류를 고치지 않는다. 기존 raw CPU 예측을 다시 학습하지 않고
재생했고 모든 원본 source 해시가 일치했다(`jh_normal_gate94_screen.py`, `normal_gate94_screen.json`).

| normal cutoff | primary 추가 우회 | alternative 추가 우회 | grouped OOF tutor 추가 우회 |
|---|---|---|---|
| .94 | normal4 | normal3 | PII1 |
| .92 | normal9, 학업부정1 | normal5 | normal4, OOM1, off-topic1, PII1 |
| .90 | normal9, 학업부정1 | normal6 | normal8, 유해2, OOM2, off-topic1, PII1 |

추가 우회만 센 수치이며 기존 gate 전체 오류가0이라는 뜻은 아니다. 개발 자료에서 고른 기준이고
PII의 tutor action은 answer라 같지만 범주가 normal과 일치하는 것은 아니다.
`jh_r6_normal_gate94`는 tutor의 argmax가 normal이고 확률이[.94,.95)일 때만 요청별 pipeline copy의
threshold를.94로 낮춘다. 기존 전체 classify 경로와 섹션 후처리를 통과하며 label을 직접 지정하지 않는다.
다른 범주·guard·섹션 .98/.90 임계값은 유지한다. .92/.90은 실제 차단 누락 반례가 있어 구현하지 않는다.
전체508테스트 통과. primary192와alternative181에서 양 기준선과 비교할 계획이다.

공개 추적 도구 `jh_public_trace.py`를 추가했다. 완료된 public/파생 실행의 해시·모델·채점 완료를 확인한
자료만 받아 직접 tutor를 호출하며, 원래 hook의 인자·결과·예외·취소를 관찰한다. 추가 검색·모델 호출을
만들지 않는다. labels/ID/gold는 pipeline body로 넘기지 않는다. HTTP 벤치마크/공식 점수로 읽지 않는다.
`_ask`의 entry_*는 부모 prompt 변환 전이고 source_response는 외부 결과라 VoteThree 내부3raw는 아니다.
모의14테스트에서 호출 수·원 응답·취소·병렬 격리·데이터 범위를 확인했다. 실제 제출 서버에서는 사용하지 않는다.

## 20–23 완료 — query 보조 전달과 계산 근거

| 후보 | 자료 | tutor | 수집 초 | 내부 실패 요청 |
|---|---|---:|---:|---:|
| resolved_source_hint | topic114 | 92.76 | 23.62 | 4 |
| resolved_source_hint | followup114 | 90.35 | 25.75 | 8 |
| computable_material | topic114 | 89.25 | 24.09 | 2 |
| computable_material | followup114 | 89.04 | 24.32 | 2 |

누적2934 HTTP요청, 오류(HTTP/형식/판정)0. source_hint가 유망하지만 source만 바꿨는데도
앞단 action 정답 수가 topic107→108, followup100→107로 변했다. 이를 source 변경의 직접 효과로
세지 않는다. 반복 및 primary/alternative 검증이 필요하다. 컴퓨팅 근거 설명도 아직 안정적인 우위는 아니다.

## 단일 근거 판정 후보

`jh_r6_joint_evidence`는 material 판정에 최종 selector와 같은 최대6개 full section을 주고
supported boolean·section·짧은 reason을 한 번에 받는다. block/other와 병렬이며 기존 요청 한도 내13초,
low/2000토큰·비스트리밍이다. 기존 chunk 기반 material_source/quote 검증과는 입력 정보와 출력 계약이
다르지만, v2의 full-section material/model-source 선택과 연결되는 재검증이다.

strict boolean 및 후보 내 ID를 확인한 뒤 기존 voted 우선순위에 맞게 supported=false는 빈 quote,
true는 실제 선택 섹션 원문으로 호환한다. 이 quote는 **모델이 발췌한 인용이 아닌 내부 schema adapter**다.
reason을 교재 사실이나 답변으로 사용하지 않는다. 전체 classify가 정상 완료되고 최종 normal/PII,
기존 learned_section/tutor_section_gate가 미적용인 경우에만 joint section을 복원하고 별도 selector를 생략한다.
최종 focused 원문과 길이 제한 fallback은 유지한다. prepare 취소, 다른 범주, 기존 section gate에서는 재사용하지 않는다.
invalid/timeout은 추가 material 재시도 없이 기존 실패 복구를 따른다. guard 코드는 그대로다.

보조 검토에서 후보 계산 예외도 fallback에 포함하도록 try 범위를 보완했다.
실제 classify MRO를 통과하는 모의 통합검사에서 BLOCK2+OTHER2+joint1 뒤 추가 source 호출0을 확인했다.
최종518테스트 통과. 이 호출 수는 해당 검사 조건의 수이며 모든 요청의 고정 호출 횟수는 아니다.

다음 우선순위는 normal .94의 primary/alternative 대조와 새 source 후보의 교차검증이다.
