#!/bin/bash
set -eu

ITERS="${ITERS:-7000}"
ALPHA="${ALPHA:-0.1}"
SEED="${SEED:-0}"
ROUNDS="${ROUNDS:-3}"
INIT_TRAIN="${INIT_TRAIN:-10%}"
CALIB="${CALIB:-10%}"
TEST="${TEST:-20%}"
ADD_K="${ADD_K:-5}"
METHODS="${METHODS:-random fisher color visibility combined}"
SCENE="${SCENE:-tandt/train}"
AL_ROOT="${AL_ROOT:-output/active_learning/tandt_train}"

for method in $METHODS; do
    echo "Submitting scene=$SCENE method=$method seed=$SEED"
    sbatch --export=ALL,SCENE="$SCENE",AL_ROOT="$AL_ROOT",METHOD="$method",SEED="$SEED",ROUNDS="$ROUNDS",INIT_TRAIN="$INIT_TRAIN",CALIB="$CALIB",TEST="$TEST",ADD_K="$ADD_K",ITERS="$ITERS",ALPHA="$ALPHA" snellius_jobs/02_active_learning_loop.job
done
