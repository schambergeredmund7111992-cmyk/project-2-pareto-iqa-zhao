#!/bin/bash
# Score the remaining no-reference metrics from the lecture at 224px.
#
# Skips anything already on disk, so an interrupted run picks up where it
# stopped rather than starting over.
set -u
GPU=${1:?usage: run_rest.sh <gpu> [metric ...]}
shift
METRICS=${@:-"maniqa topiq_nr arniqa liqe wadiqam_nr cnniqa nima paq2piq tres ilniqe"}
for m in $METRICS; do
  for split in "train.csv split_manifest.csv val" "heldout.csv heldout_manifest.csv heldout"; do
    set -- $split
    out="${m}_${3}_224.csv"
    if [ -f "$out" ]; then echo "--- $out exists, skipping"; continue; fi
    echo "=== $m on $3 ==="
    CUDA_VISIBLE_DEVICES=$GPU python run_pyiqa.py --metric "$m" --data "./$1" \
      --manifest "./$2" --max-side 224 --out "$out" 2>&1 | tee "${m}_${3}_224.log"
  done
done
