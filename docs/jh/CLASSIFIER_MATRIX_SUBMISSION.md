# 분류기 비교 후보 — 런박스 제출

브랜치: `experiment/kimjaehak-peace-guard-fusion`. 기존 후보도 같은 브랜치에 보존한다. `.env`는 런박스의 기존 파일을 사용하며 git으로 전송하지 않는다. 모델·임계값·결과의 근거는 [CLASSIFIER_MATRIX.md](CLASSIFIER_MATRIX.md)에 있다.

## 공통 준비

기존 서버 터미널에서 Ctrl+C로 종료한 뒤, 스크린샷에서 확인한 저장소 위치에서 실행한다.

```bash
cd ~/first_penguins_jh
git remote set-branches --add origin experiment/kimjaehak-peace-guard-fusion &&
git fetch origin &&
git switch experiment/kimjaehak-peace-guard-fusion &&
git pull --ff-only origin experiment/kimjaehak-peace-guard-fusion &&
git log -1 --oneline
```

기존 가상환경에 pip가 없었던 경우를 포함한 의존성 준비:

```bash
.venv/bin/python -m ensurepip --upgrade &&
.venv/bin/python -m pip install -r requirements-jh-research.txt
```

모든 후보는 `--embedding-mode env`, slots64, port8000으로 비교한다. Dense 분류기는 제공 `text-embedding-3-small`을 사용하며 검색용 임베딩 외에 마지막 사용자 발화의 분류용 임베딩을 추가 호출한다. 새 신경망 라이브러리나 외부 모델 파일 설치는 없다. 공개자료로 서버 준비 시 sklearn 모델을 학습한다.

## 1. 임베딩 + 로지스틱 회귀

primary196 가드 97.27(기존 95.02), 혼합388 전체 95.96/가드 96.61(기존 95.70/95.70)을 확인한 첫 제출 시험 후보다. alternative184에서는 기존보다 낮았으므로 확정 개선으로 보지 않는다. 모두 공개자료 기반 로컬 점수다.

```bash
cd ~/first_penguins_jh
.venv/bin/python experiments/jh_server.py \
  --variant jh_cm_dense_lr50 \
  --embedding-mode env \
  --request-slots 64 \
  --host 0.0.0.0 \
  --port 8000
```

`준비 끝: 흐름 jh_cm_dense_lr50`, `임베딩 있음`, `Application startup complete`를 확인한다. 이 variant의 실제 분류 임계값은 `classifier_threshold`이며 상속된 `guard_gate_threshold` 표시값과 구분한다.

다른 터미널에서 기존 성공한 제출 주소를 유지하고 토큰·커밋을 갱신한다. 이전과 같은 런박스 주소와 8000포트를 사용하는 기준이다.

```bash
cd ~/first_penguins_jh
.venv/bin/python - <<'PY'
import json
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('.env', override=True)
p = Path('/home/elicer/competition/submission.json')
d = json.loads(p.read_text())
d['token'] = os.environ['TEAM_TOKEN']
d['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
print('submission.json 갱신 완료, commit:', d['commit'])
PY
```

서버를 유지한 채 런박스 제출 버튼을 누르고, 채점 완료까지 서버·탭을 유지한다. 결과에 variant 이름을 함께 기록한다.

## 2. 문자 TF-IDF + 작은 신경망

신경망의 실제 제출 결과를 비교하기 위한 후보다. 공개 파생 진단에서 기존보다 일관되게 좋다는 증거는 없다. 첫 제출의 채점 완료 후 서버를 Ctrl+C로 종료하고 실행한다.

```bash
cd ~/first_penguins_jh
.venv/bin/python experiments/jh_server.py \
  --variant jh_cm_char_mlp50 \
  --embedding-mode env \
  --request-slots 64 \
  --host 0.0.0.0 \
  --port 8000
```

이번에는 `흐름 jh_cm_char_mlp50`을 확인한다. 코드·포트·토큰이 같으면 submission.json을 다시 수정할 필요 없이 제출 버튼을 다시 누르면 된다. 파일에는 variant 필드가 없으므로 서버를 실제로 바꿔야 한다.

## 나머지 두 조합

필요하면 같은 절차에서 각각 아래 명령으로 실행한다. 현재 제출 우선순위는 낮다.

문자 TF-IDF + 로지스틱(C3000, 공통 fallback 비교용):

```bash
.venv/bin/python experiments/jh_server.py \
  --variant jh_cm_char_lr50 --embedding-mode env \
  --request-slots 64 --host 0.0.0.0 --port 8000
```

제공 임베딩 + 작은 신경망(25% OOF 운영점):

```bash
.venv/bin/python experiments/jh_server.py \
  --variant jh_cm_dense_mlp --embedding-mode env \
  --request-slots 64 --host 0.0.0.0 --port 8000
```

## 기존 후보로 복귀

현재 실행한 서버를 종료하고:

```bash
.venv/bin/python experiments/jh_server.py \
  --variant jh_pg_fast_cancel --embedding-mode env \
  --request-slots 64 --host 0.0.0.0 --port 8000
```

이전 실제 제출 근거가 있는 `jh_r4_material_budget` 또는 `jh_r3_selected_focused`도 동일한 방식으로 variant만 바꾸면 된다. 계정별 마지막 제출을 집계하는 규정이므로 시험 후 최종 선택한 후보로 다시 제출해야 한다.

제출 결과는 Score·튜터·가드·Duration·오류 집계·seed와 실행 variant를 함께 비교한다. 요청 본문이나 응답 본문은 수집하지 않는다.
