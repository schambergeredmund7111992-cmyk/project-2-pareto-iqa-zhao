#!/bin/bash
# Re-score every metric at a stated input convention.
#
# The earlier runs handed pyiqa the file and let it decide, which meant SPAQ's
# 4032x3024 phone captures went into networks whose receptive fields were sized
# for a small image. On 300 SPAQ images that cost DBCNN 0.3050 against 0.8739
# at 224px. Every row here is measured at one convention, stated in the log.
set -u
GPU=${1:?usage: rerun_all.sh <gpu> <metric> [metric ...]}
shift
mkdir -p results/pyiqa
for m in "$@"; do
  for split in "data/train.csv data/split_manifest.csv val" "data/heldout.csv data/heldout_manifest.csv heldout"; do
    set -- $split
    echo "=== $m on $3 ==="
    CUDA_VISIBLE_DEVICES=$GPU python run_pyiqa.py --metric "$m" --data "./$1" \
      --manifest "./$2" --max-side 224 --out "results/pyiqa/${m}_${3}_224.csv" 2>&1 | tee "results/pyiqa/${m}_${3}_224.log"
  done
done
