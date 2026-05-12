#!/bin/bash
set -eu

ITERS="${ITERS:-30000}"
ALPHA="${ALPHA:-0.1}"
SCENE="${SCENE:-tandt/train}"
OUT="${OUT:-output/basic_tandt_train_${ITERS}}"

sbatch --export=ALL,SCENE="$SCENE",OUT="$OUT",ITERS="$ITERS",ALPHA="$ALPHA" snellius_jobs/01_full_pipeline.job
