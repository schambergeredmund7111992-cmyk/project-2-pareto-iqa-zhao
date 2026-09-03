from __future__ import annotations

# ============================================================
# IMPORTANT:
# Must be set BEFORE importing transformers / pyiqa.
# This prevents Hugging Face from making online requests
# during the benchmark.
# ============================================================

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import warnings
import time

import numpy as np
import pandas as pd
import torch
import pyiqa

from PIL import Image
from scipy.stats import spearmanr, pearsonr


warnings.filterwarnings("ignore")


# ============================================================
# PATHS & CONFIG
# ============================================================

# Your project/data manifest directory is on zhao's path.
ROOT = (
    "/home/sergey/pareto-optimal/"
    "zhao/project-2-pareto-iqa"
)

HELDOUT_CSV = f"{ROOT}/heldout.csv"

OUTPUT_PREDICTIONS_CSV = (
    f"{ROOT}/qalign_heldout_predictions.csv"
)

OUTPUT_RESULTS_CSV = (
    f"{ROOT}/qalign_project02_results.csv"
)

OUTPUT_METADATA_JSON = (
    f"{ROOT}/qalign_project02_metadata.json"
)

# Actual image dataset root
DATA_ROOT = (
    "/home/sergey/iqa-data"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "qalign"
MODEL_ID = "Q-Align"

DEVICE = "cuda"


# ============================================================
# DATASETS
# ============================================================

DATASETS = [
    "clive",
    "csiq",
    "tid2013",
    "agiqa3k",
]

# AGIQA-3K is reported separately.
# It MUST NOT be included in the natural macro.
NATURAL_DATASETS = [
    "clive",
    "csiq",
    "tid2013",
]


# ============================================================
# BENCHMARK CONFIG
# ============================================================

# Q-Align PyIQA wrapper currently supports batch size 1.
THROUGHPUT_BATCH_SIZE = 1

# Warmup does not enter latency statistics.
WARMUP_ITERATIONS = 5

# Number of latency measurements.
LATENCY_ITERATIONS = 30

# For consistency, throughput is derived from batch-1 p50.
#
# throughput = 1000 / latency_p50_ms
#
# This means:
# 1 image per inference
# no batch=16
# no data loading included
THROUGHPUT_FROM_P50 = True


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 100)
print("Q-ALIGN — PROJECT 02 EXTERNAL BASELINE BENCHMARK")
print("=" * 100)

print()
print("Model        :", MODEL_ID)
print("PyIQA metric :", MODEL_NAME)
print("Checkpoint   : q-future/one-align")
print("Project role : External baseline")
print("HF offline   :", os.environ.get("HF_HUB_OFFLINE"))

print()


# ============================================================
# ENVIRONMENT
# ============================================================

if DEVICE == "cuda":

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but CUDA is not available."
        )

    device = torch.device("cuda")

else:

    device = torch.device(DEVICE)


print("=" * 100)
print("ENVIRONMENT")
print("=" * 100)

print()
print("PyTorch:", torch.__version__)
print(
    "PyIQA:",
    getattr(
        pyiqa,
        "__version__",
        "unknown",
    ),
)
print("Device:", device)


if device.type == "cuda":

    gpu_name = torch.cuda.get_device_name(device)

    cuda_capability = (
        torch.cuda.get_device_capability(device)
    )

    print("GPU:", gpu_name)
    print("CUDA capability:", cuda_capability)

else:

    gpu_name = str(device)
    cuda_capability = None

print()


# ============================================================
# HELPER: RESOLVE IMAGE PATH
# ============================================================

def resolve_path(path: str) -> str:

    path = str(path)

    if os.path.isabs(path):
        return path

    return os.path.join(
        DATA_ROOT,
        path,
    )


# ============================================================
# LOAD HELD-OUT DATA
# ============================================================

print("=" * 100)
print("LOADING HELD-OUT DATA")
print("=" * 100)

heldout = pd.read_csv(
    HELDOUT_CSV
)

print()
print(
    f"Total held-out images: "
    f"{len(heldout)}"
)


# ============================================================
# HELD-OUT INTEGRITY CHECK
# ============================================================

print()
print("=" * 100)
print("HELD-OUT INTEGRITY CHECK")
print("=" * 100)

# ------------------------------------------------------------
# Exact expected size
# ------------------------------------------------------------

EXPECTED_HELDOUT_ROWS = 8010

if len(heldout) != EXPECTED_HELDOUT_ROWS:

    raise RuntimeError(
        f"Expected {EXPECTED_HELDOUT_ROWS} held-out rows, "
        f"got {len(heldout)}"
    )


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

if "path" not in heldout.columns:
    raise RuntimeError(
        "heldout.csv is missing column: path"
    )

if "dataset" not in heldout.columns:
    raise RuntimeError(
        "heldout.csv is missing column: dataset"
    )


if "scaled_subjective_score" in heldout.columns:

    SCORE_COLUMN = "scaled_subjective_score"

elif "subjective_score" in heldout.columns:

    SCORE_COLUMN = "subjective_score"

else:

    raise RuntimeError(
        "heldout.csv must contain either "
        "'scaled_subjective_score' or "
        "'subjective_score'."
    )


# ------------------------------------------------------------
# Normalize dataset names
# ------------------------------------------------------------

def normalize_dataset_name(name):

    name = (
        str(name)
        .strip()
        .lower()
    )

    aliases = {
        "kadid": "kadid10k",
        "kadid10k": "kadid10k",

        "koniq": "koniq10k",
        "koniq10k": "koniq10k",

        "spaq": "spaq",

        "clive": "clive",

        "csiq": "csiq",

        "tid": "tid2013",
        "tid2013": "tid2013",

        "agiqa": "agiqa3k",
        "agiqa3k": "agiqa3k",
    }

    return aliases.get(
        name,
        name,
    )


heldout["dataset_norm"] = (
    heldout["dataset"]
    .map(normalize_dataset_name)
)


print()
print("Dataset counts:")
print(
    heldout["dataset_norm"]
    .value_counts()
)


# ------------------------------------------------------------
# Training-set leakage check
# ------------------------------------------------------------

TRAIN_DATASETS = {
    "kadid10k",
    "koniq10k",
    "spaq",
}

HELDOUT_DATASETS = {
    "clive",
    "csiq",
    "tid2013",
    "agiqa3k",
}


leakage = (
    set(heldout["dataset_norm"])
    & TRAIN_DATASETS
)

if leakage:

    raise RuntimeError(
        "Training-set leakage detected: "
        f"{sorted(leakage)}"
    )


unexpected = (
    set(heldout["dataset_norm"])
    - HELDOUT_DATASETS
)

if unexpected:

    raise RuntimeError(
        "Unexpected dataset(s) in heldout.csv: "
        f"{sorted(unexpected)}"
    )


missing = (
    HELDOUT_DATASETS
    - set(heldout["dataset_norm"])
)

if missing:

    raise RuntimeError(
        "Missing held-out dataset(s): "
        f"{sorted(missing)}"
    )


print()
print("Training-set leakage check: PASSED")
print(
    "Held-out datasets:",
    sorted(
        set(heldout["dataset_norm"])
    )
)

print(
    f"Held-out integrity check: PASSED "
    f"({len(heldout)} images)"
)


# ============================================================
# LOAD Q-ALIGN
# ============================================================

print()
print("=" * 100)
print("LOADING Q-ALIGN")
print("=" * 100)

metric = pyiqa.create_metric(
    MODEL_NAME,
    device=DEVICE,
)

metric.eval()

print()
print("Q-Align loaded successfully.")
print(
    "lower_better:",
    getattr(
        metric,
        "lower_better",
        None,
    ),
)


# ============================================================
# BASIC MODEL INFORMATION
# ============================================================

total_params = sum(
    p.numel()
    for p in metric.parameters()
)

trainable_params = sum(
    p.numel()
    for p in metric.parameters()
    if p.requires_grad
)

frozen_params = (
    total_params
    - trainable_params
)


if DEVICE.startswith("cuda"):

    try:

        precision = str(
            next(
                metric.parameters()
            ).dtype
        )

    except StopIteration:

        precision = "unknown"

else:

    precision = "cpu"


print()
print("Model parameters:", total_params)
print("Trainable parameters:", trainable_params)
print("Frozen parameters:", frozen_params)
print("Precision:", precision)


# ============================================================
# ACCURACY: DATASET-LEVEL SRCC / PLCC
# ============================================================

print()
print("=" * 100)
print("ACCURACY EVALUATION")
print("=" * 100)

results = {}
prediction_rows = []


for dataset_name in DATASETS:

    subset = heldout[
        heldout["dataset_norm"]
        == dataset_name
    ].copy()

    print()
    print("=" * 100)
    print(
        f"EVALUATING {dataset_name.upper()} "
        f"({len(subset)} images)"
    )
    print("=" * 100)

    gt = []
    pred = []

    for count, (_, row) in enumerate(
        subset.iterrows(),
        1,
    ):

        image_path = resolve_path(
            row["path"]
        )

        if not os.path.isfile(
            image_path
        ):

            print(
                f"\n[WARNING] missing: "
                f"{image_path}"
            )

            continue

        try:

            # ------------------------------------------------
            # Do NOT manually resize.
            #
            # Q-Align / PyIQA performs its own preprocessing.
            # ------------------------------------------------

            image = (
                Image.open(
                    image_path
                )
                .convert("RGB")
            )

            with torch.inference_mode():

                score = metric(
                    image,
                    task_="quality",
                )

            score_value = float(
                score.detach()
                .cpu()
                .reshape(-1)[0]
            )

            gt_value = float(
                row[SCORE_COLUMN]
            )

            if (
                not np.isfinite(
                    score_value
                )
                or not np.isfinite(
                    gt_value
                )
            ):

                continue

            gt.append(
                gt_value
            )

            pred.append(
                score_value
            )

            prediction_rows.append(
                {
                    "dataset": dataset_name,
                    "path": row["path"],
                    "ground_truth": gt_value,
                    "prediction": score_value,
                }
            )

        except Exception as exc:

            print(
                f"\n[WARNING] failed: "
                f"{image_path}"
            )

            print(
                f"           {exc!r}"
            )

        if (
            count % 100 == 0
            or count == len(subset)
        ):

            print(
                f"\rProcessed "
                f"{count:5d}/{len(subset):5d}",
                end="",
                flush=True,
            )

    print()

    if len(gt) < 2:

        srcc = np.nan
        plcc = np.nan

    else:

        srcc = float(
            spearmanr(
                gt,
                pred,
            ).statistic
        )

        plcc = float(
            pearsonr(
                gt,
                pred,
            ).statistic
        )

    results[dataset_name] = {
        "srcc": srcc,
        "plcc": plcc,
        "n": len(gt),
    }

    print(
        f"{dataset_name.upper():10s} "
        f"n={len(gt):5d} "
        f"SRCC={srcc:.6f} "
        f"PLCC={plcc:.6f}"
    )


# ============================================================
# SAVE PER-IMAGE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    prediction_rows
)

prediction_df.to_csv(
    OUTPUT_PREDICTIONS_CSV,
    index=False,
)

print()
print(
    "Per-image predictions saved to:"
)
print(
    OUTPUT_PREDICTIONS_CSV
)


# ============================================================
# NATURAL DATASET MACRO
# ============================================================

natural_srcc = {
    dataset: results[dataset]["srcc"]
    for dataset in NATURAL_DATASETS
    if np.isfinite(
        results[dataset]["srcc"]
    )
}

natural_plcc = {
    dataset: results[dataset]["plcc"]
    for dataset in NATURAL_DATASETS
    if np.isfinite(
        results[dataset]["plcc"]
    )
}


# IMPORTANT:
#
# Each dataset gets its own correlation coefficient first.
#
# Macro SRCC =
# (
#   CLIVE SRCC
#   +
#   CSIQ SRCC
#   +
#   TID2013 SRCC
# ) / 3
#
# AGIQA-3K is deliberately excluded.

natural_macro_srcc = (
    float(
        np.mean(
            list(
                natural_srcc.values()
            )
        )
    )
    if natural_srcc
    else np.nan
)

natural_macro_plcc = (
    float(
        np.mean(
            list(
                natural_plcc.values()
            )
        )
    )
    if natural_plcc
    else np.nan
)


# ============================================================
# WORST NATURAL SRCC
# ============================================================

if natural_srcc:

    worst_set = min(
        natural_srcc,
        key=natural_srcc.get,
    )

    worst_srcc = natural_srcc[
        worst_set
    ]

else:

    worst_set = "N/A"
    worst_srcc = np.nan


# ============================================================
# COST BENCHMARK
# ============================================================

print()
print("=" * 100)
print("COST BENCHMARK")
print("=" * 100)


# ------------------------------------------------------------
# Select one valid image
# ------------------------------------------------------------

test_image_path = None

for _, row in heldout.iterrows():

    candidate = resolve_path(
        row["path"]
    )

    if os.path.isfile(candidate):

        test_image_path = candidate

        break


if test_image_path is None:

    raise RuntimeError(
        "No valid image found "
        "for cost benchmark."
    )


test_image = (
    Image.open(
        test_image_path
    )
    .convert("RGB")
)


# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------

print()
print(
    f"Warmup: "
    f"{WARMUP_ITERATIONS} iterations"
)

for _ in range(
    WARMUP_ITERATIONS
):

    with torch.inference_mode():

        metric(
            test_image,
            task_="quality",
        )


if device.type == "cuda":

    torch.cuda.synchronize()


# ------------------------------------------------------------
# Peak memory:
#
# Reset AFTER model loading and warmup,
# then measure one inference.
# ------------------------------------------------------------

if device.type == "cuda":

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        device
    )

    torch.cuda.synchronize()


# ------------------------------------------------------------
# Batch-1 latency
# ------------------------------------------------------------

print()
print(
    f"Latency benchmark: "
    f"{LATENCY_ITERATIONS} iterations"
)

latencies = []


for _ in range(
    LATENCY_ITERATIONS
):

    if device.type == "cuda":

        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        metric(
            test_image,
            task_="quality",
        )

    if device.type == "cuda":

        torch.cuda.synchronize()

    end = time.perf_counter()

    latencies.append(
        (end - start) * 1000.0
    )


latencies = np.asarray(
    latencies,
    dtype=np.float64,
)


latency_p50 = float(
    np.percentile(
        latencies,
        50,
    )
)

latency_p95 = float(
    np.percentile(
        latencies,
        95,
    )
)


print()
print(
    f"Latency p50: "
    f"{latency_p50:.3f} ms"
)

print(
    f"Latency p95: "
    f"{latency_p95:.3f} ms"
)


# ------------------------------------------------------------
# Throughput
#
# Q-Align wrapper is batch=1.
#
# Therefore use:
#
# Throughput = 1000 / latency_p50
#
# Same inference granularity as latency benchmark.
# ------------------------------------------------------------

if (
    THROUGHPUT_FROM_P50
    and latency_p50 > 0
):

    throughput = (
        1000.0
        / latency_p50
    )

else:

    throughput = np.nan


print()
print(
    f"Throughput "
    f"(batch={THROUGHPUT_BATCH_SIZE}): "
    f"{throughput:.3f} images/s"
)


# ------------------------------------------------------------
# Peak VRAM
# ------------------------------------------------------------

if device.type == "cuda":

    torch.cuda.synchronize()

    peak_memory_mb = (
        torch.cuda
        .max_memory_allocated(
            device
        )
        / (1024 ** 2)
    )

else:

    peak_memory_mb = np.nan


print()
print(
    f"Peak VRAM: "
    f"{peak_memory_mb:.2f} MB"
)


# ============================================================
# FLOPs
# ============================================================
#
# IMPORTANT:
#
# This is NOT a universal "ground truth" FLOPs number.
#
# It is the FLOP count reported by PyTorch's
# torch.utils.flop_counter.FlopCounterMode for
# one Q-Align inference.
#
# Q-Align is a multimodal vision-language model,
# so this number should NOT be blindly compared with
# FLOPs reported by CNN-specific profilers unless the
# measurement protocol is identical.
# ============================================================

print()
print("=" * 100)
print("FLOPs BENCHMARK")
print("=" * 100)

flops = None

try:

    from torch.utils.flop_counter import (
        FlopCounterMode
    )

    if device.type == "cuda":

        torch.cuda.synchronize()

    with FlopCounterMode(
        display=False
    ) as flop_counter:

        with torch.inference_mode():

            metric(
                test_image,
                task_="quality",
            )

    flops = float(
        flop_counter.get_total_flops()
    )

    print()
    print(
        "FLOPs / image "
        "(PyTorch FlopCounterMode):"
    )

    print(
        f"{flops:.0f}"
    )

    print(
        "FLOPs / image (TFLOPs):"
    )

    print(
        f"{flops / 1e12:.6f}"
    )

except Exception as exc:

    print()
    print(
        "FLOPs / image: N/A"
    )

    print(
        "Reason:",
        repr(exc)
    )


# ============================================================
# FINAL RESULT
# ============================================================

final_result = {
    "model": MODEL_ID,

    "model_name": MODEL_NAME,

    "checkpoint": (
        "q-future/one-align"
    ),

    "design_space": (
        "External baseline"
    ),

    "heldout_images": int(
        len(heldout)
    ),

    # ---------------- Accuracy ----------------

    "CLIVE_SRCC": float(
        results["clive"]["srcc"]
    ),

    "CLIVE_PLCC": float(
        results["clive"]["plcc"]
    ),

    "CSIQ_SRCC": float(
        results["csiq"]["srcc"]
    ),

    "CSIQ_PLCC": float(
        results["csiq"]["plcc"]
    ),

    "TID2013_SRCC": float(
        results["tid2013"]["srcc"]
    ),

    "TID2013_PLCC": float(
        results["tid2013"]["plcc"]
    ),

    "AGIQA3K_SRCC": float(
        results["agiqa3k"]["srcc"]
    ),

    "AGIQA3K_PLCC": float(
        results["agiqa3k"]["plcc"]
    ),

    # AGIQA-3K deliberately excluded.
    "Natural_Macro_SRCC": float(
        natural_macro_srcc
    ),

    "Natural_Macro_PLCC": float(
        natural_macro_plcc
    ),

    "Natural_Worst_SRCC": float(
        worst_srcc
    ),

    "Natural_Worst_Set": worst_set,

    # ---------------- Cost ----------------

    "FLOPs": flops,

    "FLOPs_definition": (
        "PyTorch FlopCounterMode, "
        "single-image Q-Align inference"
    ),

    "Latency_P50_ms": float(
        latency_p50
    ),

    "Latency_P95_ms": float(
        latency_p95
    ),

    "Throughput_images_per_sec": float(
        throughput
    ),

    "Throughput_definition": (
        "batch=1; "
        "1000 / latency_p50_ms"
    ),

    "Peak_VRAM_MB": float(
        peak_memory_mb
    ),

    # ---------------- Model ----------------

    "Parameters": int(
        total_params
    ),

    "Trainable_Parameters": int(
        trainable_params
    ),

    "Frozen_Parameters": int(
        frozen_params
    ),

    "Device": DEVICE,

    "GPU": gpu_name,

    "Precision": precision,

    "HF_HUB_OFFLINE": (
        os.environ.get(
            "HF_HUB_OFFLINE"
        )
    ),
}


# ============================================================
# SAVE FINAL RESULT
# ============================================================

pd.DataFrame(
    [final_result]
).to_csv(
    OUTPUT_RESULTS_CSV,
    index=False,
)

print()
print("=" * 100)
print("FINAL PROJECT-02 RESULT")
print("=" * 100)

print(
    pd.DataFrame(
        [final_result]
    ).T.to_string(
        header=False
    )
)

print()
print(
    "Results saved to:"
)

print(
    OUTPUT_RESULTS_CSV
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model": MODEL_ID,

    "model_name": MODEL_NAME,

    "checkpoint": (
        "q-future/one-align"
    ),

    "design_space": (
        "External baseline"
    ),

    "heldout_csv": HELDOUT_CSV,

    "heldout_rows": int(
        len(heldout)
    ),

    "training_datasets": sorted(
        TRAIN_DATASETS
    ),

    "heldout_datasets": sorted(
        HELDOUT_DATASETS
    ),

    "natural_macro_datasets": (
        NATURAL_DATASETS
    ),

    "agiqa3k_in_macro": False,

    "device": DEVICE,

    "gpu": gpu_name,

    "precision": precision,

    "throughput_batch_size": (
        THROUGHPUT_BATCH_SIZE
    ),

    "warmup_iterations": (
        WARMUP_ITERATIONS
    ),

    "latency_iterations": (
        LATENCY_ITERATIONS
    ),

    "flops_method": (
        "PyTorch FlopCounterMode"
    ),

    "flops_scope": (
        "single-image Q-Align inference"
    ),

    "hf_offline": (
        os.environ.get(
            "HF_HUB_OFFLINE"
        )
    ),
}


with open(
    OUTPUT_METADATA_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print(
    "Metadata saved to:"
)

print(
    OUTPUT_METADATA_JSON
)

print()
print("DONE.")