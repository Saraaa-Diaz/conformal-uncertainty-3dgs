#!/bin/bash
set -eu

ITERS="${ITERS:-30000}"
ALPHA="${ALPHA:-0.1}"
SCENE="${SCENE:-db/playroom}"
OUT="${OUT:-output/basic_db_playroom_${ITERS}}"

sbatch --export=ALL,SCENE="$SCENE",OUT="$OUT",ITERS="$ITERS",ALPHA="$ALPHA" snellius_jobs/01_full_pipeline.job
