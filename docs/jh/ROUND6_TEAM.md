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
