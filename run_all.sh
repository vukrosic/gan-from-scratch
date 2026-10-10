#!/bin/sh
# Run every step in order, then the tests, then the exercise.
# The training runs are saved in files/, so the steps only read them (a few seconds each).
# The exercise trains one pair of networks (about 15 seconds). All of it: about half a minute. To measure every run again: delete files/ and run python3 runs.py first.
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
for f in step1_no_judge.py step2_judge.py step3_game.py step4_advice.py step5_digits.py step6_score.py tests.py exercise.py; do
  echo
  echo "=== $f ==="
  $PY $f
done
