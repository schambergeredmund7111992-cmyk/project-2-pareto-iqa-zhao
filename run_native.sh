#!/bin/bash
set -u
export HF_ENDPOINT=https://hf-mirror.com
GPU=${1:?usage: run_native.sh <gpu> <metric> [metric ...]}
shift
for m in "$@"; do
  for split in "train.csv split_manifest.csv val" "heldout.csv heldout_manifest.csv heldout"; do
    set -- $split
    out="${m}_${3}.csv"
    if [ -s "$out" ]; then echo "--- $out exists, skipping"; continue; fi
    echo "=== $m on $3 === $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU python run_pyiqa.py --metric "$m" --data "./$1" \
      --manifest "./$2" --out "$out" 2>&1 | tee "${m}_${3}.log"
  done
done
echo "### queue finished $(date +%H:%M)"
