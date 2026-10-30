#!/usr/bin/env bash
# Benchmark 2 in parallel: one process per (fold, model), at most $JOBS at a time, then one merged result.
#   JOBS=6 scripts/kt-parallel.sh
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-6}
FOLDS=${FOLDS:-5}
MODELS=${MODELS:-skill_mean,engine,engine_difficulty,engine_problem,irt,pfa,bkt,dkt}
CELLS=${CELLS:-runs/kt-cells}
mkdir -p "$CELLS"
for f in $(seq 0 $((FOLDS - 1))); do
  for m in ${MODELS//,/ }; do
    [ -f "$CELLS/fold$f-$m.json" ] || echo "$f $m"
  done
done | xargs -P "$JOBS" -L 1 sh -c 'uv run python -m dreval.kt.run --folds '"$FOLDS"' --only-folds "$0" --models "$1" --jobs 2 --cell-dir '"$CELLS"' > '"$CELLS"'/fold"$0"-"$1".log 2>&1 && echo "done fold $0 $1"'
uv run python -m dreval.kt.run --folds "$FOLDS" --models "$MODELS" --merge "$CELLS" --out runs/kt-assist09.json
