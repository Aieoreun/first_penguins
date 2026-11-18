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

## 24–27 완료 — normal .94

| 자료 | material_budget | focused | normal .94 |
|---|---:|---:|---:|
| primary192 | 95.83 (17) | 93.75 (18) | 95.96 (24) |
| alternative181 | 90.47 (25) | 92.27 (26) | 92.54 (27) |

24 수집21.03초/내부실패13(embedding timeout9 포함),25는17.87초/3,26은19.35초/4,
27은16.82초/1. HTTP/형식/판정 오류0. 누적3669 HTTP요청.
primary의 새 gate 대상4개는25/100/100/50점에서 모두100점이 됐다. 이 부분은 총점+0.6510점,
나머지188문항 변동은−0.5208점이다. alternative의 새 대상3개는75/100/50에서 모두100,
이 부분+0.4144점, 나머지178문항+1.6575점이다. 전체 차이를 임계값의 인과 효과로 세지 않는다.
반복/혼합 부하 검증 전이지만 좁은 정상 질문 개선 후보로 유지한다.

### 공개 실패 trace 5건

baseline17의 action 불일치5건만 `jh_public_trace_r6_primary_failures_1`로 새로 실행했다.
새로운 API 응답을 관찰한 직접 호출 진단이며 원래 HTTP 실행의 내부 trace를 복원한 것은 아니다.
걱정 질문은 material이 정답 근거를 찾았지만 OTHER 두 표가 emotional_distress였다.
없는 점수 기준 요청에는 관련 분류 예문, 없는 기능의 확인창 요청에는 '그 기능은 만들지 않는다'는
문장을 quote로 반환해 normal이 됐다. 관련 근거와 질문한 속성의 실제 지원을 구분해야 한다는 가설을 뒷받침한다.
나머지2건은 이 진단에서 normal로 회복됐다. trace5건은 위 HTTP 점수/요청 수에 합산하지 않는다.

### 임베딩이 쓰이지 않는 경로의 호출 생략

`jh_r6_trusted_sparse`는 기존 tutor 입력 normal/PII confidence≥.95, known section confidence≥.90,
retrieved 제한 없음 조건에서 query embedding만 생략하고 원래 BM25를 실행한다. guard 및 나머지 경로는
그대로다. 분류·출처 후처리·최종 focused 인용을 미리 대신하지 않고 원래 메서드를 모두 실행한다.
공유 llm/searcher를 변경하지 않으며 _search 동안만 요청별 facade를 사용한다.

모의 테스트 포함522개 통과. 실제 공개 모델을1회 학습한 별도 CPU screen은 public104/200,
primary74/192,alternative67/181의245쌍에서 고의로 dense 검색 결과를 바꿔도 최종 JSON이 모두 같았다.
모의 embedding245→0, chat0, 불일치0. 이 수치는 실제 API 시간·점수 개선이 아니다.
실제 deadline 실패가 있는 경우까지 동등하다는 주장도 아니다. 결과는 trusted_sparse_equivalence.json.
다음에는 실제 혼합 tutor/guard 부하에서도 확인한다.

### 원격 재확인

실험 중 추가 fetch에서 Peace export/peace-jh-full-no-trained만2f51c1f→6f7281c로 갱신됐다.
새 변경은 guard의 block/other 뒤 material 직렬 실행, prepare22초이며 tutor는 기존 경로에 위임한다.
새 동일조건 전후 점수 파일은 없었다. 가드 고정 조건에서 이식하지 않는다. 다른 개인 브랜치 ref 변화 없음.

## 28–35 수집과 후속 후보 사전 등록

28 source_budget12 / 29 source_reason12 / 30 joint_evidence는 topic114,
31 source_hint / 32 computable_material / 33 joint_evidence는 primary192,
34 material_budget / 35 selected_focused는 topic114 재실행이다. 수집 commit 1e9a43e, 모두 clean.
8개를 순차 수집한 뒤 원본 평가기로 순차 채점한다. 누적 수집4815요청이며 API 수집과 채점을 겹치지 않는다.

29에서는 source_extract:BadRequestError가104건 발생했다. 당시 source medium/3000 설정이
기존 01_experiments.md에 기록된 프록시의 비스트리밍2000토큰 제한을 위반했다. 개발 시 제한을
재확인하지 않은 실수다. tutor52.19와 원본 결과를 보존하되 medium 추론의 품질 비교로 해석하지 않는다.
수정 commit c466086은 medium/2000으로 제한하고 모든 등록 R6 후보의 토큰 상한·deadline 회귀 검사를 추가했다.
수정본은 작은 실제 API smoke를 먼저 확인하며, 옛29 실행의 manifest/응답/점수를 덮어쓰지 않는다.

새 후보 두 개를 별도 등록했다. 기존 제출 후보·평가기·가드는 그대로이며 전체537개 테스트 통과.

- `jh_r6_normal94_sparse`: normal .94 분류와 trusted_sparse 검색만 조합한다. 검색 시 임베딩 생략은
  기존 .95 이상 입력/.90 이상 출처 조건 그대로이며 새 [.94,.95) 정상 질문에서는 임베딩을 유지한다.
  실제 분류·출처 후처리·생성까지의 모의 동등성과 동시 요청의 모델/임계값/클라이언트 불변성을 검사했다.
- `jh_r6_joint_quote`: joint 한 호출에서 질문 대상/요청 속성을 분리한 asked_fact와 연속 원문 quote를
  먼저 적고 supported를 결정하도록 한다. quote가 선택 섹션의 공백 정규화 부분문자열이 아니면 빈 quote와
  InvalidQuote 진단으로 기존 OOM 우선순위에 전달한다. 추가 호출·재시도는 없다. 최종 focused 전체 인용은 유지한다.
  **문자열 포함 검사는 의미적 답변 가능성의 증명이 아니다.** 무관하지만 실제 원문인 quote도 통과할 수 있다.
  원본 joint adapter를 동작 동일한 hook으로 분리했으며 기존10개+새9개 테스트에서 원래 결과·가드·취소·우선순위를 확인했다.

joint primary33에서 OOM20건 중 answer8건은 material 실패가 아니라 판정의 과잉지원으로 보인다.
8건 모두 입력 .95 gate 미달이며 material_joint/prepare 오류0이다. 기존17은 answer2건이었다.
원래 입력모델 argmax OOM, .5≤p<.95로 재확인 대상을 좁히면 이8건 중4건만 포착한다.
primary 전체 대상9건은 OOM8/normal1, alternative7건은 OOM5/normal1/PII1이다.
grouped OOF10건에는 normal4건도 포함되어 확률만으로 OOM을 강제하는 방식은 채택하지 않는다.
선택적 재확인은 후속 가설로 남기고 먼저 quote 후보의 이득/정상 질문 손실을 확인한다.

다음 비교는 trusted_sparse와 normal94_sparse의 primary/alternative, 수정 medium의 topic,
joint_quote의 topic/primary, 기존 두 기준선과 유망 조합의 public400 혼합 부하다.
public 원본은 학습 자료이므로 일반화 점수로 쓰지 않고 tutor/guard 동시 실행의 회귀·지연 확인에 쓴다.

## 28–35 채점 완료

| 번호 / 자료 | 후보 | tutor | 수집 초 | 내부 실패 요청 |
|---|---|---:|---:|---:|
| 28 topic114 | source_budget12 | 89.69 | 25.90 | 2 |
| 29 topic114 | source_reason12 **잘못된 3000토큰 설정** | 52.19 | 20.36 | 104 |
| 30 topic114 | joint_evidence | 92.32 | 13.65 | 3 |
| 31 primary192 | resolved_source_hint | 93.10 | 23.36 | 5 |
| 32 primary192 | computable_material | 93.62 | 22.18 | 9 |
| 33 primary192 | joint_evidence | 94.14 | 19.40 | 1 |
| 34 topic114 | material_budget 반복 | 86.18 | 29.59 | 9 |
| 35 topic114 | selected_focused 반복 | 89.04 | 25.15 | 11 |

35회4815 HTTP요청 채점 완료, 미채점0. HTTP/형식/판정 오류0이지만29의 내부 API104실패는 별도로 명시한다.
topic에서 material_budget 세 번88.16/88.60/86.18 평균87.647, focused90.79/87.94/89.04 평균89.257이다.
단일 좋은 후보를 이 반복 범위와 비교한다. source_hint92.76, joint92.32가 topic에서 유망하지만
primary에서는 각각93.10/94.14로 baseline95.83보다 낮다. computable도primary93.62로 전체 채택 근거가 없다.
source budget12는 source 단계 오류0으로 감소했지만89.69 단회이며 예산 증가만으로 점수 상승을 확정하지 않는다.

joint의 topic 응답 p50은5.121초로 신규 기준선34의10.293초, focused35의10.291초보다 짧았다.
이는 다른 시점의 단회 비교이고 반복 필요. 기존 material 대신 full section 지원 판정을 사용해 OOM 회귀가 나타났다.
다음 성능 가설은 **기존 material 판정을 유지하며 출처 선택만 병렬로 미리 수행**하는 것이다.
입력 정상/PII 확률이 높은 fallback 요청에 한정하고, 최종 정책 우선순위/출처 gate가 우선하며
가드·원본 selector 프롬프트·후보 풀·최종 focused 답변은 유지하는 별도 후보로 검증한다.

medium/2000 비스트리밍·JSON 실제 smoke1회는 정상 JSON으로 성공했다(1.965초).
`experiments/.cache/jh_r6_medium2000_smoke.json`에 파라미터와 성공 여부만 보존했다.
기능 확인이며 품질/HTTP 요청 수에는 합산하지 않는다. 현재 사용률75%, 잔여25%로 후속 실험을 계속한다.

### 문항별 관측 차이

기존 `jh_paired_scores.py`로 채점 체크포인트만 재생했다. source_id를 묶은10000회 bootstrap이며
같은 실행 쌍의 표본 변동만 보여 준다. API 재실행/판정 모델의 변동이나 비공개 일반화 불확실성은 포함하지 않는다.

| 비교 | 관측 차이 | 문항 그룹 bootstrap95 구간 |
|---|---:|---|
| primary17→normal94(24) | +0.130 | −2.214..+2.604 |
| alternative25→normal94(27) | +2.072 | −1.105..+5.387 |
| primary17→source_hint(31) | −2.734 | −5.990..+0.391 |
| primary17→computable(32) | −2.214 | −5.729..+1.172 |
| primary17→joint(33) | −1.693 | −5.208..+1.823 |
| topic34→budget12(28) | +3.509 | +0.219..+6.798 |
| topic34→joint(30) | +6.140 | +2.412..+10.307 |
| topic35 focused→joint(30) | +3.289 | 0.000..+7.018 |

`experiments/.cache/jh_r6_paired_f.json`에 상세를 보존했다. 같은 기준선의 반복 편차와 자료별 상충을
함께 고려하므로, 이 구간만으로 새 후보를 최종 추천에 승격하지 않는다.

## 출처 선택 병렬화

`jh_r6_eager_source`는 기존 fallback 분류의 `_collect_checks`에 들어온 tutor 중 입력모델의
argmax가 normal/PII이고 확률≥.5인 경우에만 원래 source selector를 미리 시작한다.
기존 material chunk·프롬프트·block/other 투표는 그대로다. 미리 만든 후보 풀은 원래 ungated normal의
firstchunk/last-BM25/retrieved 순서와 같으며 원래 source prompt/low/2000/8초를 유지한다.

분류가 끝나기 전에는 선택한 출처를 반영하지 않는다. 최종 normal/PII이며 learned/source gate가
미적용인 경우에만 결과를 재사용한다. 사용하지 않을 작업은 취소하고 회수한다. 실패·prepare 취소 후에도
두 번째 selector 호출을 시작하지 않으며, 그때 남아 있는 실제 section/label의 기존 excerpt/fallback을 따른다.
선제 호출 준비의 예외는 원래 분류를 막지 않는다. 완료된 미사용 호출의 실패도 내부 오류 집계에 남긴다.

분류를 바꾸지 않아도 시간 배분은 달라진다. 출처 대기가 prepare16초 안에 들어가며, 최종 OOM·차단·출처 gate로
버릴 호출이 늘 수 있다. 총API 호출 수 감소를 실측했다고 주장하지 않는다. 먼저 topic/primary에서 기존 두 후보와
점수·p50/p95·실패를 비교하고 유망하면 혼합 부하 및 반복 검증으로 진행한다. 가드와 전체26초 마감은 유지한다.

Eager17개 포함 전체554개 테스트 통과. 실제 `_run`의 prepare/outer deadline 모의 실행에서도 취소 작업 회수와
미완료 source 재사용 금지, source 재시도 없음, 가드 불변성을 확인했다. 보호 경로 diff0.
36–44는 trusted_sparse·normal94_sparse의 primary/alternative 각1회, 수정medium의topic1회,
joint_quote 및 eager_source의 topic/primary 각1회로 사전 등록한다. 각 실행 전 잔여 사용량을 다시 확인한다.

## 36–44 수집 완료와 추가 가설

수집 commit ec8f3bd, 전체 clean. 추가1472요청으로 누적 수집6287개이며 원본 평가기로 순차 채점 중이다.
첫 결과는 sparse primary94.40/alternative92.82, normal94_sparse primary95.05다.
원래 trusted 경로의 실제 HTTP 응답 동일성과 해당 경로의 판정 변동을 따로 비교한다.
수정 medium/2000은 BadRequest0이지만 topic88.60/31.38초, source timeout5건으로 low/12초 후보89.69/25.90초보다
이번 관측이 좋지 않다. 실패한3000 설정과 구분하며 추가 추론 수준 확대는 후순위로 둔다.

strict joint_quote는 topic에서 InvalidQuote26건, primary에서5건이 발생했다. 실제 최종 응답은 새로 생성한
인용을 내보내지 않고 원래 KB를 사용하지만, 검증 탈락을 OOM으로 전달하므로 정상 질문 손실 가능성이 크다.
검증 실패 이유는 아직 실제 trace로 확인하지 않았고, 모든 실패가 잘못된 사실이라는 주장은 하지 않는다.
이 때문에 다음 두 변화를 분리해 구현했다(commit243ef12, 전체566테스트 통과).

- `jh_r6_joint_fact`: asked_fact/quote를 먼저 정리하는 프롬프트만 사용하고, 결과 적용은 원래 joint의
  strict boolean·section adapter 그대로 둔다. 모델 quote를 검증된 근거로 취급하거나 사용자 답변으로 보내지 않는다.
  원문 부분문자열 검사와 새 프롬프트 효과를 분리하는 실험이며 기존 joint_quote는 보존한다.
- `jh_r6_eager_source12`: 기존 eager의 source timeout만8→12초. topic source timeout7건이 남았으므로
  prepare16/요청26초 안에서 추가 시간을 활용할 수 있는지 검증한다. original material·source prompt는 동일하다.
- `jh_r6_resolved_material_hint`: 성공한 resolved 검색 힌트를 source뿐 아니라 tutor material의 별도 JSON
  보조 필드로 제공한다. 원 대화·조각·block/other·가드는 보존하고 추가 호출 없이13초 material 한도를 유지한다.
  query의 기존 검증은 형식/길이/검색 성공이며 의미적 정확성 증명이 아니다. followup을 우선하고 primary OOM 손실도 확인한다.

public trace에는 기존 joint adapter hook의 입력/출력을 관찰하는 기능을 추가했다. API 호출이나 판정/출처를
추가하지 않으며 공개 완료 실행만 입력으로 받아 로컬 개발 캐시에 기록한다. 프로덕션과 서버에는 연결하지 않는다.
현재 잔여22%이므로15% 기준까지 후속 비교를 계속한다.

## 36–44 채점 완료

| 번호 / 자료 | 후보 | tutor | 수집 초 | 내부 실패 요청 |
|---|---|---:|---:|---:|
| 36 primary192 | trusted_sparse | 94.40 | 19.03 | 2 |
| 37 alternative181 | trusted_sparse | 92.82 | 17.55 | 1 |
| 38 primary192 | normal94_sparse | 95.05 | 18.84 | 0 |
| 39 alternative181 | normal94_sparse | 93.09 | 17.59 | 1 |
| 40 topic114 | source_reason12 **수정2000토큰** | 88.60 | 31.38 | 7 |
| 41 topic114 | joint_quote | 78.73 | 14.00 | 27 |
| 42 primary192 | joint_quote | 92.97 | 14.91 | 7 |
| 43 topic114 | eager_source | 88.60 | 21.74 | 11 |
| 44 primary192 | eager_source | 94.53 | 16.19 | 1 |

44회6287 HTTP요청 채점 완료, 미채점0. HTTP/형식/판정 오류0이며 내부실패는 표에 별도 표시했다.
strict quote는 정상 질문을 과도하게 abstain시켜 topic92.32→78.73으로 하락했다. primary도94.14→92.97이다.
엄격검사 후보를 채택하지 않으며, 프롬프트만 분리한 joint_fact는 다른 후보로 평가한다.
Eager는 topic88.60으로 focused 세 번 평균89.257을 넘지 못했다. 시간은 줄었지만 source timeout7건이 남아
12초 변형을 한 번 비교한다. primary94.53 역시 baseline95.83보다 낮지만 단회와 동일 응답 판정 변동을 구분한다.

### trusted 경로의 실제 HTTP 확인

이전 합성 벡터 screen의 대상 목록과 실제 데이터/KB 해시를 검증한 뒤, 실제 HTTP 최종JSON을 비교했다.

| 비교 | 생략 대상 JSON 동일 | 짝지은 HTTP 시간 차이 중앙값(새−기존) |
|---|---:|---:|
| primary17→36 | 74/74 | −0.4005초 |
| alternative25→37 | 67/67 | −0.3790초 |
| primary normal94 24→38 | 74/74 | −0.7685초 |
| alternative normal94 27→39 | 67/67 | −0.4490초 |

이 경로에서 점수가 달라진 부분은 모두 **동일 JSON에 대한 판정 변동**이었다. 실제 임베딩 호출 telemetry는 없으므로
합성 screen의 호출 카운트를 HTTP 실측으로 쓰지 않는다. 시간에는 서버/네트워크 부하의 실행 간 차이도 포함된다.
37의 전체90.47→92.82 상승은 생략 대상 밖114건에서 발생했고, 그중 일부도 동일 응답의 판정 변화다.
따라서 전체 상승을 sparse 검색의 정확도 이득으로 해석하지 않는다. 대상 응답 유지와 지연 절감의 근거로 유지한다.
상세: `experiments/.cache/jh_r6_sparse_http_analysis/{analysis.md,analysis.json}`.

### 새 KB 대비 자료

후보 예측을 보기 전에 KB에서 정상/미제공 속성 질문12쌍을 새로 작성하고 별도 에이전트가 KB만으로 검토했다.
8쌍16문항을 승인,4쌍을 제외했다. 한 쌍은 OOM/학습목표 범주 모호성, 금융3쌍은 여러 정당한 출처와
단일 gold evidence의 불일치가 이유다. 문항/라벨을 모델 응답에 맞춰 수정하지 않는다.
이는 작은 수동 약한 라벨 개발자료이며 독립 holdout/실제 제출 추정이 아니다. 작성자의 이전 실험 맥락도 한계로 기록했다.
설계 및 검토의 해시를 고정하고 데이터 파일을 먼저 만든 뒤 모델을 확인한다. 학습/런타임 정답표에 넣지 않는다.

## 교재 대비 자료 동결과 source support 후보

독립 검토 후 `jh_kb_contrast_probe.py --build-only`로16문항을 먼저 동결했다.
dataset SHA256 `f86c11a7f52be956425e85149472a0e96f01c72f4cab173de7f6fd2b385515ca`.
제안/검토/재현은 [contrast16/README.md](contrast16/README.md)에 보존했다. 공급 KB72파일·파일크기·canonical해시,
문항쌍해시·본문/표 출처와 섹션 대응·정확인용·추가인용·정답형식·검토 전수 accounting을 검사한다.
19개 synthetic fixture 테스트를 통과했고 채점 정답/ID는 HTTP 요청에 전달하지 않는다.

동결 뒤 기존 공개 학습 모델의 CPU 확률을 확인했다. 미제공8문항 중2개가 normal .9603/.9943으로
기존 .95 입력 gate를 통과하지만 출처 confidence는 .2097/.7032로 낮았다. 새 .94 band 대상은0개다.
나머지6개는 기존 LLM 분류로 넘어간다. 이 자료만으로 실제 제출 오류율을 추정하지 않는다.

`jh_r6_source_support`는 **입력 high-confidence normal이지만 출처는 신뢰하지 못하는** 기존 source 호출만 바꾼다.
조건은 tutor/NORMAL/learned_gate이고 learned_section·tutor_section_gate가 미적용인 경로다.
후보 풀·full section 입력·low/2000/8초·전체26초를 유지하고, 그 한 번의 selector에서
asked_fact/quote·supported boolean·section을 받는다. boolean과 후보 ID만 엄격 검증한다.
true는 선택한 실제 KB의 focused 원문, false는 기존 고정 OOM 문구와 빈 인용을 반환한다.
모델 quote를 검증된 사실로 취급하거나 답변에 넣지 않는다. PII·가드·이미 material을 확인한 fallback과
고신뢰 출처 경로는 그대로다. invalid/timeout은 기존 출처 원문 fallback이며 재시도하지 않는다.
가드·입력 분류·학습자료 변경 없이11개 모의 테스트 포함 전체596개 통과, 보호 경로 diff0.

다음45–58 사전 계획: contrast16의 baseline/focused/source_support, source_support primary/alternative,
material_hint followup/primary, joint_fact topic/primary, eager12 topic/primary,
public400 혼합부하 baseline/focused/normal94_sparse. 각 실행 전 사용량을 확인하며 잔여15% 이하이면 새 실행을 시작하지 않는다.
현재 잔여19%; 진행 중 채점은 중단하지 않고 끝낸 뒤 자료·인수인계·커밋을 정리한다.

strict quote의 이전 abstain 3건을 새 public trace로 관찰했다. HTML 역할 질문은 정상 인용으로 answer,
실습 채점 순서 질문은 정답 섹션이 실제 후보에 없어 supported=false, 용돈 기입장 질문은 후보에 없는
section ID를 반환해 ValueError fallback이었다. 새3회에서 InvalidQuote는 재현되지 않았다.
따라서 이전26건의 구체적인 문자열 실패 원인을 이 trace로 단정하지 않는다. 재실행 변동과 후보 누락·ID오류가
함께 남아 있음을 기록한다. public trace 누적8요청이며 HTTP 점수 집계에는 포함하지 않는다.

### 자동 승인 검토 보류 및 개발 자료 배포 제외

45 contrast baseline을 시작하기 직전에 exec 자동 승인 검토가 실행을 거절했다.
이유는 새로 만든 로컬 KB 파생 질문을 외부 API로 보내는 것에 대한 명시적 승인이 불분명하다는 판단이다.
프로세스/서버/로그 파일이 생성되기 전 거절됐으므로45–47의 수집·API 호출은0이다. 우회하지 않는다.
실제 대상은 기존 `.env`의 mlapi.run이며 KB/public은 주최측 기준6ec111d와 diff0임을 확인했다.
사용자에게 새16문항 전송 범위의 승인 여부를 비동기로 질문했고, 답변 전까지 해당3실행을 보류한다.
기존 공개자료를 쓰는48–58은 별개로 진행한다.

규정 §7.1의 개발/서버 자료 구분도 다시 적용했다. 코딩 도구가 작성한 별도 문항 묶음을 서버 자료로
사용하지 않도록, 재현 편의를 위해 추가했던 docs의 proposal/review/gate-screen JSON 사본을 제거했다.
원본은 git-ignore된 로컬 개발 캐시에만 보존하고 학습·검색·서버 실행 경로와 연결하지 않는다.
배포 트리에는 본문 없는 개발 메타데이터만 남긴다. CPU 진단과 실제 API 점수를 구분하며 미수행 점수를 만들지 않는다.

추가 fetch에서 Peace 6f7281c→881635f를 확인했다. guard prepare22→26/별도deadline28/material13초 변경이며,
tutor는 유지되고 새 HTTP 비교 점수는 없다. 가드 고정 조건에서 이식하지 않았다. 다른 원격 ref는 그대로다.
