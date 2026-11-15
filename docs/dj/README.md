# dj — 석동진 작업 기록

각자 작업하며 남기는 설계 메모·실험 결과·아키텍처 비교 분석·인수인계를 둔다. 공용 자료는 `docs/ot/`, 전체 제출 기록은 `docs/submissions.md`.

- 기준 문서 브랜치: `expert/bsuk1` (`eb012e9` 이후 고도화 실험은 `feature/step6-dual-supervisor`에서 진행)
- 현 최고 런박스 점수: **`102cb2d` (Step 7) — 총점 `80.18` (튜터 `73.22`, 가드 `87.14`, `85.336s`)**
- 팀원 최고 `/guard`: **`bef5614` (`origin/experiment/seungwon` v4) — 총점 `79.19` (튜터 `70.25`, 가드 `88.14`, `109.045s`)**
- 채점 요청·답변 본문은 어디에도 기록하지 않는다 (`SPEC.md` §8).

| 문서 | 내용 |
| --- | --- |
| [01_설계_메모.md](01_%EC%84%A4%EA%B3%84_%EB%A9%94%EB%AA%A8.md) | 하이브리드 RAG 튜터·가드레일 아키텍처 및 초기 주요 설계 결정 (`e0c65a1` 기준) |
| [02_실험_결과.md](02_%EC%8B%A4%ED%97%98_%EA%B2%B0%EA%B3%BC.md) | 1차(`e0c65a1`)부터 Step 9(`cc1aa25`) 및 승원(`bef5614`)까지 전체 런박스 제출 표·변동 분석 |
| [03_고도화_및_팀비교_분석.md](03_%EA%B3%A0%EB%8F%84%ED%99%94_%EB%B0%8F_%ED%8C%80%EB%B9%84%EA%B5%90_%EB%B6%84%EC%84%9D.md) | Step 2~9 진화, Dual-Critic `/tutor` 실패·복구 교훈, `experiment/seungwon` v4 정밀 비교, 400문항·KB 감사 데이터 |
| [HANDOFF.md](HANDOFF.md) | 현재 브랜치 상태, 다음 유망 실험안(A/B), 실행·검증 명령어 |
