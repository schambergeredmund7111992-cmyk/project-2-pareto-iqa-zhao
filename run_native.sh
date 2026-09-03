#!/bin/bash
set -u
export HF_ENDPOINT=https://hf-mirror.com
GPU=${1:?usage: run_native.sh <gpu> <metric> [metric ...]}
shift
mkdir -p results/pyiqa
for m in "$@"; do
  for split in "data/train.csv data/split_manifest.csv val" "data/heldout.csv data/heldout_manifest.csv heldout"; do
    set -- $split
    out="results/pyiqa/${m}_${3}.csv"
    if [ -s "$out" ]; then echo "--- $out exists, skipping"; continue; fi
    echo "=== $m on $3 === $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$GPU python run_pyiqa.py --metric "$m" --data "./$1" \
      --manifest "./$2" --out "$out" 2>&1 | tee "results/pyiqa/${m}_${3}.log"
  done
done
echo "### queue finished $(date +%H:%M)"
