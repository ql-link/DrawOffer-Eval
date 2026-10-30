#!/usr/bin/env bash
# Benchmark 2 in parallel: one process per (fold, model), at most $JOBS at a time, then one merged result.
#   DATASET=ednet JOBS=6 scripts/kt-parallel.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET=${DATASET:-assist09}
JOBS=${JOBS:-6}
FOLDS=${FOLDS:-5}
MODELS=${MODELS:-skill_mean,engine,engine_difficulty,engine_no_forgetting,engine_problem,irt,pfa,bkt,dkt}
CELLS=${CELLS:-runs/kt-cells-$DATASET}
mkdir -p "$CELLS"
# Read the raw files once; the cells load the pickled students.
uv run python -c "from pathlib import Path; from dreval.kt.run import students_of, ROOTS; students_of('$DATASET', Path.home() / 'Downloads/kt-data' / ROOTS['$DATASET'], 0)"
for f in $(seq 0 $((FOLDS - 1))); do
  for m in ${MODELS//,/ }; do
    [ -f "$CELLS/fold$f-$m.json" ] || echo "$f $m"
  done
done | xargs -P "$JOBS" -L 1 sh -c 'uv run python -m dreval.kt.run --dataset '"$DATASET"' --folds '"$FOLDS"' --only-folds "$0" --models "$1" --jobs 2 --cell-dir '"$CELLS"' > '"$CELLS"'/fold"$0"-"$1".log 2>&1 && echo "done fold $0 $1"'
uv run python -m dreval.kt.run --dataset "$DATASET" --folds "$FOLDS" --models "$MODELS" --merge "$CELLS" --out "runs/kt-$DATASET.json"
