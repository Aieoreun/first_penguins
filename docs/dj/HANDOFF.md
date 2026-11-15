# HANDOFF — 석동진 남은 일

- 갱신: 2026-11-15
- 코드 브랜치: `expert/bsuk1` (`e0c65a1`, `dev`에 아직 병합되지 않음)
- 문서 위치: `docs/dj/` (`README.md`, `01_설계_메모.md`, `02_실험_결과.md`, `HANDOFF.md`)

---

## 1. 현재 상태 요약

- **완료된 것**
  - 하이브리드 RAG(`/tutor`) + 입력·출력 가드(`/guard`) 1차 파이프라인 구현 및 GitHub `origin/expert/bsuk1` 푸시 (`e0c65a1`).
  - 오프라인 단위 테스트(`11/11` 통과) 및 64 동시성 부하 테스트(`64/64` 통과, `2.278s`).
  - 5-fold OOF 교차검증을 통한 TF-IDF in-sample 누수(`81.03` vs `54.12`) 진단 및 `gpt-5-nano` + kNN 경로 전환.
  - 엘리스 런박스 1차 공식 제출 완료: **총점 74.19 (튜터 70.0, 가드 78.37)**, `67.980s`, 오류 0건.
  - 멘토링 질문지 작성 및 멘토 피드백 확보 (긴 응답시간 처리, `Supervisor` 패턴, `Search -> Review -> Write`, 출처의 명시, Evaluation, 하네스 엔지니어링).
- **미완료 / 확인 필요**
  - `expert/bsuk1`의 `dev` 병합 PR.
  - 런박스에서 `scripts/build_offline_assets.py` 실행 후 임베딩 사전 계산(`app/assets/kb_embeddings.npz`) 적용 여부 확인.
  - `docs/submissions.md` 공용 제출표에 1차 제출 시각 및 런박스 `commit` 해시 기록.

---

## 2. 우선순위별 다음 작업

1. **[긴급 버그 수정 — 즉시 점수 개선 대상]**
   - [ ] `app/tutor_pipeline.py` (`_run_guard`): `GuardClassifier.classify` 메서드명 불일치 수정 (현재 `/tutor` 사전 가드 우회 상태 해소).
   - [ ] `app/guard_verifier.py` (`verify`): `raw_grounded=False`인데 `unsupported` 스팬이 비었을 때 `grounded=True`로 강제 덮어쓰는 로직 수정.
2. **[하네스 엔지니어링 — 정량 평가 체계 구축]**
   - [ ] `harness/score_local.py` 결과를 `action` 일치율, 인용 F1, 범주별 정확도, `span_f1`, `safe` 일치율 등으로 분해 출력하는 평가 스크립트 작성 및 실험 이력 자동 누적 체계 구축.
3. **[아키텍처 고도화 — 멘토 피드백 반영]**
   - [ ] `Supervisor` 기반 `Search -> Review -> Write` 파이프라인 도입 검토 (데드라인 예산 기반 긴 응답시간 폴백, Review 단계에서 불필요 인용 제거 및 조건부 재검색으로 출처 정확도 개선).
   - [ ] 경계 범주(`out_of_material` vs `off_topic`, `answer_seeking` vs `academic_dishonesty`) 분류 기준 보강 후 2차 런박스 제출.

---

## 3. 실행 및 검증 명령어 (`expert/bsuk1` 기준)

```bash
# 1. 의존성 설치 및 환경변수 로드
pip install -r requirements.txt
set -a; source .env; set +a

# 2. (권장) KB 및 공개 세트 임베딩 사전 생성
python3 scripts/build_offline_assets.py

# 3. 서버 기동 (포트 8000)
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000

# 4. 로컬 채점기 및 64 동시성 테스트
python3 harness/score_local.py \
  --tutor-url http://localhost:8000/tutor \
  --guard-url http://localhost:8000/guard \
  --token "$TEAM_TOKEN" --public public --kb kb

python3 scripts/stress_test.py --base-url http://localhost:8000 --token "$TEAM_TOKEN" --concurrency 64 --total 64
```
