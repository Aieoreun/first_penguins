# 교재 속성 대비 16문항

공급 KB로 만든 수동 개발자료다. 12쌍을 후보 모델 예측 없이 설계하고 별도 검토에서 8쌍만 승인했다.
독립 비공개 검증이나 실제 제출 점수 추정용 자료가 아니다. 원본 제안·검토 파일을 byte 그대로 보존했다.
`proposal.json`에는 제외된 질문도 있으며, 승인 목록은 `review.json`이 결정한다.
생성 당시의 `pipeline_not_evaluated`는 설계 시점의 사실이다. 동결 후에는 아래 CPU 진단과 HTTP 비교에 사용한다.
파이프라인 학습/런타임 정답 조회에는 넣지 않는다.

## 동결 및 수집 재현

저장소 루트에서 실행한다. 이미 같은 캐시가 있다면 복사 단계는 생략한다.

```bash
mkdir -p experiments/.cache/jh_r6_kb_contrast_draft experiments/.cache/jh_r6_kb_contrast_review
cp docs/jh/contrast16/proposal.json experiments/.cache/jh_r6_kb_contrast_draft/proposal.json
cp docs/jh/contrast16/review.json experiments/.cache/jh_r6_kb_contrast_review/review.json
.venv/bin/python -B experiments/jh_kb_contrast_probe.py \
  --proposal experiments/.cache/jh_r6_kb_contrast_draft/proposal.json \
  --review experiments/.cache/jh_r6_kb_contrast_review/review.json \
  --name my_contrast_build --variant jh_r4_material_budget --build-only
```

수집하려면 새 `--name`을 쓰고 `--build-only`를 뺀다. `.env`를 읽고 원래 HTTP 수집기와
원래 채점기를 사용한다. 출력한 run 경로를 `experiments/jh_resume_transfer.py`에 전달해 채점한다.
variant만 바꿔 비교할 수 있으며 HTTP payload에는 messages/spec/session만 전달되고 gold/id는 들어가지 않는다.

동결 dataset SHA256: `f86c11a7f52be956425e85149472a0e96f01c72f4cab173de7f6fd2b385515ca`.
동결 뒤 시행한 `gate_screen.json`은 기존 공개 학습 분류기의 확률만 본 CPU 진단이다.
미제공8문항 중03/08 두 개의 raw normal 확률이 .9603/.9943이고, 출처 확률은 .2097/.7032였다.
새 normal .94 band 대상은0개다. HTTP 점수는 별도 ROUND6_TEAM.md/round6_results.json에 기록한다.
