#!/bin/bash
# clean_eval.sh <merged_model_dir> <label>
# Score a model on the FIXED 40-task/band held-out set (deterministic seeds 60000+, disjoint
# from training) with the same protocol every time, so the base-vs-iter curve is real signal
# rather than the 8/band per-iter noise. Appends one JSON line to flywheel/clean_eval_results.jsonl.
set -u
cd /Users/dennison/develop/quantum-project/ecdsa-model
export PATH="$HOME/.local/bin:$PATH"
export ECDSA_BUILD_HEAVY=1
MODEL="$1"; LABEL="$2"
PERBAND="${3:-40}"
LOG="infra/cleaneval_${LABEL}.log"
echo "[cleaneval] $LABEL  model=$MODEL  per_band=$PERBAND  $(date)"
modal run train/tooleval_modal.py --model-dir "$MODEL" --per-band "$PERBAND" --max-steps 40 > "$LOG" 2>&1
RES=$(grep -E "TOOLEVAL_RESULT" "$LOG" | tail -1 | sed 's/^TOOLEVAL_RESULT //')
if [ -n "$RES" ]; then
  /usr/bin/python3 -c "
import json,sys
r=json.loads('''$RES'''.replace(chr(39),chr(34)))
row={'label':'$LABEL','model':'$MODEL','per_band':$PERBAND,'overall':r.get('overall_solve_rate'),'by_band':r.get('by_band')}
open('flywheel/clean_eval_results.jsonl','a').write(json.dumps(row)+'\n')
print('[cleaneval] $LABEL ->', row['overall'], {b:v.get('solve_rate') for b,v in r.get('by_band',{}).items()})
"
else
  echo "[cleaneval] $LABEL FAILED — no TOOLEVAL_RESULT (see $LOG)"; tail -5 "$LOG"
fi
