#!/bin/bash
# eval_policy.sh CKPT NAME TRIALS SEED  -> export to exports/act_part_flip_NAME, run TRIALS policy flips in the cell
set -e
cd "$(dirname "$0")/../.."
CKPT=$1; NAME=$2; TRIALS=${3:-6}; SEED=${4:-901}; DEMOS=${5:-53}
OUT=exports/act_part_flip_$NAME
TRAIN=$(ps -eo pid,args | awk '/physicalai fit/ && !/awk/ {print $1}' | head -1)
SIM=$(ps -eo pid,args | awk '/_owner_worker/ && !/awk/ {print $1}' | head -1)
[ -f $OUT/act.xml ] || nice -n 5 ${TRAIN_PYTHON:-python} -m v3.export_policy --ckpt $CKPT --out $OUT --demonstrations $DEMOS 2>&1 | grep -E "^exported" || true
[ -n "$TRAIN" ] && kill -STOP $TRAIN
[ -n "$SIM" ] && kill -CONT $SIM
sleep 3
timeout 2700 ${CELL_PYTHON:-python} -m v3.run_policy --trials $TRIALS --export $OUT --seed $SEED --record runs/policy_$NAME 2>&1 | grep -E "RESULT|SUMMARY|Error|Traceback" | sed -E 's/, "episode_dir".*//'
[ -n "$SIM" ] && [ -n "$TRAIN" ] && kill -STOP $SIM
[ -n "$TRAIN" ] && kill -CONT $TRAIN
true
