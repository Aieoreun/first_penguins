# 로컬 개발용 교재 속성 대비 진단

공급 KB로 정상/미제공 질문12쌍을 작성하고 별도 검토로8쌍16문항을 승인했다.
수동 약한 라벨의 개발 진단이며 비공개 점수 추정이나 holdout이 아니다.

SPEC §7.1은 다른 LLM으로 만든 별도 예시/데이터셋을 서버에 넣어 쓰는 것을 금지한다.
따라서 문항·검토 원본은 git에 포함하지 않는 로컬 개발 캐시에만 보존하고, 배포할 파일에서는 제거했다.
서버의 학습·검색·런타임 정답표에는 연결하지 않는다. 이 기록은 문항 본문 없는 개발 결과 메타데이터다.

- 로컬 제안: `experiments/.cache/jh_r6_kb_contrast_draft/proposal.json`
- 로컬 독립 검토: `experiments/.cache/jh_r6_kb_contrast_review/review.json`
- 동결 자료: `experiments/.cache/jh_kb_contrast_frozen_v1/derived_inputs.jsonl`
- 동결 SHA256: `f86c11a7f52be956425e85149472a0e96f01c72f4cab173de7f6fd2b385515ca`
- 제안 SHA256: `bc9ea4acfa0b7f6a6f0d3313cd69f5e188d417f34a2db6b9e4a38e522fb4505f`
- 검토 SHA256: `c4930c25c2db8b21cc2c69dc5e39059838a7b524c93a63c60f0bab521584ec06`

동결 뒤 CPU 진단에서 미제공8문항 중2개가 기존 normal .95 gate를 통과했다.
새 normal .94 band 대상은0개다. HTTP/API 진단은 자동 승인 검토의 자료 전송 범위 확인 때문에 보류됐다.
기존 공개자료 실험과 이16문항 진단은 구분하며, 미수행한 API 진단을 점수로 보고하지 않는다.
