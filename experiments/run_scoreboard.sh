#!/bin/sh
# 가드 개선 후보를 하나씩 켜서 공개 400문항 전체를 채점한다. 결과는 experiments/scoreboard.py로 모은다.
# 쓰는 법 (.env를 읽어 들인 뒤): sh experiments/run_scoreboard.sh base1 "" kb VX_CURRICULUM=1 ...   (이름과 스위치를 번갈아 적는다)
PY=${PY:-.venv/bin/python}
while [ $# -ge 2 ]; do
  name=$1; switch=$2; shift 2
  echo "== $name ($switch) 시작 $(date +%H:%M:%S)"
  env $switch $PY experiments/eval_full.py --pipeline vx --split all --name sb_$name --workers 16 > experiments/.cache/log_sb_$name.txt 2>&1
  echo "== $name 끝 $(date +%H:%M:%S): $(grep -m1 '^총점' experiments/.cache/log_sb_$name.txt)"
done
