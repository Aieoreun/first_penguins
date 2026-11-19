# 튜터 MLP 비교 후보 실행

> **현재 추천:** 로컬 혼합 반복에서 안정적인 개선이 없어 최종 후보는 A `jh_cm_char_mlp50` 유지. B/C 명령은 추가 실험을 위한 것이며 교체 권고가 아니다.

이 문서는 실행 가능한 비교 후보를 설명한다. 튜터 교체 후보 jh_tm_replace의 실제 제출84.17이 보고됐고, 선별 후보는 아직 실제 제출 기록이 없다. 기존 실제 최고85.41은 **가드에만 MLP를 쓰는** `jh_cm_char_mlp50` 기록이며 새 후보 점수가 아니다. 최신 사용자 제출은 튜터 교체 후보의84.17이다. [로컬 비교 결과](TUTOR_MLP_RESULTS.md)를 먼저 확인한다.

기존 서버를 Ctrl+C로 종료하고 브랜치를 업데이트한다. 이번 변경에는 새 의존성이 없다.

```bash
cd ~/first_penguins_jh
git switch experiment/kimjaehak-peace-guard-fusion &&
git pull --ff-only origin experiment/kimjaehak-peace-guard-fusion
```

## C: 튜터 정책 범주를 신경망으로 교체

```bash
.venv/bin/python experiments/jh_server.py \
  --variant jh_tm_replace \
  --embedding-mode env --request-slots 64 \
  --host 0.0.0.0 --port 8000
```

시작 설정에서 `tutor_mlp_threshold` 약0.79508, `tutor_mlp_single_turn: false`, `tutor_mlp_fallback: "llm"`을 확인한다. `classifier_threshold` 약0.85567은 그대로인 **가드** 임계값이다. 문자 분류기이므로 `CLASSIFIER_EMBEDDING` 준비 로그는 출력하지 않는다. 교재 검색은 여전히 임베딩을 사용한다. `준비 끝: 흐름 jh_tm_replace`와 `Application startup complete`를 확인한다.

## B: 튜터 단일 턴에만 선별 적용

서버를 종료한 후:

```bash
.venv/bin/python experiments/jh_server.py \
  --variant jh_tm_selective \
  --embedding-mode env --request-slots 64 \
  --host 0.0.0.0 --port 8000
```

튜터 임계값 약0.95602, single_turn true, fallback original이다. 로컬 비교에서 실제 새 판정이 매우 적었던 후보다.

## A: 기존 실제 최고점 후보로 복귀

서버를 종료한 후:

```bash
.venv/bin/python experiments/jh_server.py \
  --variant jh_cm_char_mlp50 \
  --embedding-mode env --request-slots 64 \
  --host 0.0.0.0 --port 8000
```

## submission.json 커밋 갱신

새 코드로 업데이트했다면 별도 터미널에서 수행한다. 기존 URL·토큰·포트가 같은 조건이다.

```bash
cd ~/first_penguins_jh
.venv/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path
p = Path('/home/elicer/competition/submission.json')
data = json.loads(p.read_text())
data['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
print('커밋 갱신:', data['commit'])
PY
```

서버와 런박스 탭을 유지하고 제출한다. 같은 코드·주소·토큰에서 variant만 바꿀 때 파일 재작성은 필요 없지만 서버는 실제로 재시작해야 한다. 계정별 마지막 제출이 집계되고 과거 최고점은 자동 유지되지 않는다. 결과에는 variant·Score·튜터·가드·Duration·seed를 함께 기록한다. 비공개 요청/응답 본문은 수집하지 않는다.
