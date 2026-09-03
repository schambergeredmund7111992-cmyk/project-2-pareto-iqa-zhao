from __future__ import annotations

import os
import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pyiqa

from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT CONFIG
# ============================================================

ROOT = "/home/sergey/pareto-optimal/zhao/project-2-pareto-iqa"

HELDOUT_CSV = f"{ROOT}/heldout.csv"

OUTPUT_PREDICTIONS_CSV = (
    f"{ROOT}/qrealign_heldout_predictions.csv"
)

OUTPUT_RESULTS_CSV = (
    f"{ROOT}/qrealign_project02_results.csv"
)

OUTPUT_METADATA_JSON = (
    f"{ROOT}/qrealign_project02_metadata.json"
)

DATA_ROOT = "/home/sergey/iqa-data"


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_NAME = "qrealign"

MODEL_ID = "q-future/Q-ReAlign-Mini-0.8B"

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

NATURAL_DATASETS = [
    "clive",
    "csiq",
    "tid2013",
]


# ============================================================
# BENCHMARK CONFIG
# ============================================================

# Project 02 official benchmark.py defaults to batch 16.
#
# IMPORTANT:
# Keep this FIXED when comparing models.
#
THROUGHPUT_BATCH_SIZE = 16

# Accuracy does not depend on this.
# Cost benchmark uses this batch size only for throughput.

WARMUP_ITERATIONS = 5

LATENCY_ITERATIONS = 30

# Same structure as official benchmark.py:
# max(5, iterations // 3)
THROUGHPUT_ITERATIONS = max(
    5,
    LATENCY_ITERATIONS // 3,
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 100)
print("Q-ReAlign Mini — PROJECT 02 EXTERNAL BASELINE")
print("=" * 100)

print()
print("Model :", MODEL_ID)
print("Metric:", MODEL_NAME)

print()
print(
    "IMPORTANT: Q-ReAlign is evaluated as an EXTERNAL VLM baseline."
)
print(
    "It is not a frozen CLIP/SigLIP Project 02 architecture."
)

print()


# ============================================================
# ENVIRONMENT
# ============================================================

if DEVICE == "cuda":

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False."
        )

    device = torch.device("cuda")

else:

    device = torch.device(DEVICE)


print("=" * 100)
print("ENVIRONMENT")
print("=" * 100)

print(
    "PyTorch:",
    torch.__version__,
)

print(
    "PyIQA:",
    getattr(pyiqa, "__version__", "unknown"),
)

print(
    "Device:",
    device,
)


if device.type == "cuda":

    gpu_name = torch.cuda.get_device_name(
        device
    )

    capability = torch.cuda.get_device_capability(
        device
    )

    print(
        "GPU:",
        gpu_name,
    )

    print(
        "CUDA capability:",
        capability,
    )

else:

    gpu_name = str(device)

print()


# ============================================================
# LOAD HELD-OUT DATA
# ============================================================

print("=" * 100)
print("LOADING HELD-OUT DATA")
print("=" * 100)


heldout = pd.read_csv(
    HELDOUT_CSV
)


print(
    f"Total held-out rows: {len(heldout)}"
)


# ------------------------------------------------------------
# Required row count
# ------------------------------------------------------------

assert len(heldout) == 8010, (
    f"Expected 8010 held-out rows, "
    f"got {len(heldout)}"
)


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required_columns = [
    "path",
    "dataset",
]


if "scaled_subjective_score" in heldout.columns:

    SCORE_COLUMN = "scaled_subjective_score"

elif "subjective_score" in heldout.columns:

    SCORE_COLUMN = "subjective_score"

else:

    raise RuntimeError(
        "heldout.csv must contain "
        "'scaled_subjective_score' or "
        "'subjective_score'."
    )


required_columns.append(
    SCORE_COLUMN
)


for column in required_columns:

    if column not in heldout.columns:

        raise RuntimeError(
            f"Missing required column: {column}"
        )


heldout["dataset"] = (
    heldout["dataset"]
    .astype(str)
    .str.lower()
)


# ============================================================
# HELD-OUT INTEGRITY
# ============================================================

print()
print("=" * 100)
print("HELD-OUT INTEGRITY CHECK")
print("=" * 100)


print()
print(
    heldout["dataset"].value_counts()
)


# ------------------------------------------------------------
# Expected datasets
# ------------------------------------------------------------

for dataset in DATASETS:

    n = int(
        (
            heldout["dataset"]
            == dataset
        ).sum()
    )

    assert n > 0, (
        f"Dataset {dataset} has zero rows."
    )


# ------------------------------------------------------------
# Training datasets must NOT appear
# ------------------------------------------------------------

FORBIDDEN_DATASETS = [
    "kadid10k",
    "koniq10k",
    "spaq",
]


for forbidden in FORBIDDEN_DATASETS:

    assert not (
        heldout["dataset"]
        == forbidden
    ).any(), (

        f"DATA LEAKAGE DETECTED: "
        f"{forbidden} appears in heldout.csv."
    )


# ------------------------------------------------------------
# Path duplication check
# ------------------------------------------------------------

duplicate_paths = int(
    heldout["path"].duplicated().sum()
)


if duplicate_paths > 0:

    print(
        f"[WARNING] Duplicate paths: "
        f"{duplicate_paths}"
    )

else:

    print(
        "Duplicate image paths: 0"
    )


# ------------------------------------------------------------
# Dataset-specific duplicate check
# ------------------------------------------------------------

duplicate_dataset_paths = int(
    heldout.duplicated(
        subset=["dataset", "path"]
    ).sum()
)


if duplicate_dataset_paths > 0:

    raise RuntimeError(
        "Duplicate dataset/path entries "
        "found in heldout.csv."
    )


print()
print(
    "Held-out integrity check: PASSED"
)

print(
    "GT score column:",
    SCORE_COLUMN
)

print()


# ============================================================
# PATH RESOLUTION
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
# CORRELATION
# ============================================================

def correlation(
    gt,
    pred,
):

    gt = np.asarray(
        gt,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )


    mask = (
        np.isfinite(gt)
        &
        np.isfinite(pred)
    )


    gt = gt[mask]
    pred = pred[mask]


    if len(gt) < 2:

        return (
            np.nan,
            np.nan,
        )


    srcc = spearmanr(
        gt,
        pred,
    ).statistic


    plcc = pearsonr(
        gt,
        pred,
    ).statistic


    return (
        float(srcc),
        float(plcc),
    )


# ============================================================
# LOAD Q-REALIGN
# ============================================================

print("=" * 100)
print("LOADING Q-REALIGN")
print("=" * 100)

print()
print(
    f"Loading {MODEL_ID}..."
)


metric = pyiqa.create_metric(
    MODEL_NAME,
    device=DEVICE,
)


metric.eval()


print(
    "Q-ReAlign loaded successfully."
)


print(
    "Metric class:",
    type(metric),
)


print(
    "lower_better:",
    getattr(
        metric,
        "lower_better",
        None,
    ),
)


assert getattr(
    metric,
    "lower_better",
    False,
) is False, (
    "Q-ReAlign must use higher-is-better scoring."
)


print()


# ============================================================
# MODEL PRECISION
# ============================================================

def get_model_dtype(model):

    for parameter in model.parameters():

        return parameter.dtype

    return None


MODEL_DTYPE = get_model_dtype(
    metric
)


print(
    "Model parameter dtype:",
    MODEL_DTYPE
)


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    total = 0
    trainable = 0

    for parameter in model.parameters():

        total += parameter.numel()

        if parameter.requires_grad:

            trainable += parameter.numel()


    return total, trainable


TOTAL_PARAMS, TRAINABLE_PARAMS = (
    count_parameters(metric)
)


FROZEN_PARAMS = (
    TOTAL_PARAMS
    - TRAINABLE_PARAMS
)


print()
print("=" * 100)
print("MODEL SIZE")
print("=" * 100)

print(
    f"Total parameters : "
    f"{TOTAL_PARAMS:,}"
)

print(
    f"Total parameters : "
    f"{TOTAL_PARAMS / 1e6:.2f} M"
)

print(
    f"Trainable parameters : "
    f"{TRAINABLE_PARAMS:,}"
)

print(
    f"Frozen parameters : "
    f"{FROZEN_PARAMS:,}"
)

print()


# ============================================================
# ACCURACY EVALUATION
# ============================================================

print("=" * 100)
print("ACCURACY EVALUATION")
print("=" * 100)


results = {}

prediction_rows = []


for dataset in DATASETS:

    sub = heldout[
        heldout["dataset"]
        == dataset
    ].copy()


    print()
    print("-" * 100)

    print(
        f"EVALUATING {dataset.upper()} "
        f"({len(sub)} images)"
    )

    print("-" * 100)


    gt = []
    pred = []

    missing_count = 0
    failed_count = 0


    for count, (_, row) in enumerate(
        sub.iterrows(),
        1,
    ):

        path = resolve_path(
            row["path"]
        )


        if not os.path.isfile(path):

            missing_count += 1

            print(
                f"\n[WARNING] Missing image: "
                f"{path}"
            )

            continue


        try:

            with torch.inference_mode():

                score = metric(
                    path,
                    task_="quality",
                )


                score_val = float(
                    score
                    .detach()
                    .cpu()
                    .reshape(-1)[0]
                )


                gt_score = float(
                    row[SCORE_COLUMN]
                )


                if (
                    not np.isfinite(
                        score_val
                    )
                    or
                    not np.isfinite(
                        gt_score
                    )
                ):

                    continue


                gt.append(
                    gt_score
                )

                pred.append(
                    score_val
                )


                prediction_rows.append({

                    "dataset": dataset,

                    "path": row["path"],

                    "ground_truth": gt_score,

                    "prediction": score_val,

                })



        except Exception as exc:

            failed_count += 1

            print(
                f"\n[WARNING] Failed: "
                f"{path}"
            )

            print(
                f" {exc!r}"
            )


        if (
            count % 100 == 0
            or count == len(sub)
        ):

            print(
                f"\rProcessed "
                f"{count:5d}/{len(sub):5d}",
                end="",
                flush=True,
            )


    print()


    srcc, plcc = correlation(
        gt,
        pred,
    )


    results[dataset] = {

        "srcc": srcc,

        "plcc": plcc,

        "n": len(gt),

        "missing": missing_count,

        "failed": failed_count,

    }


    print()

    print(
        f"{dataset.upper():10s} "
        f"n={len(gt):5d} "
        f"SRCC={srcc:.6f} "
        f"PLCC={plcc:.6f}"
    )


    if missing_count:

        print(
            f"Missing: {missing_count}"
        )


    if failed_count:

        print(
            f"Failed: {failed_count}"
        )


# ============================================================
# CHECK COMPLETENESS
# ============================================================

print()
print("=" * 100)
print("ACCURACY COMPLETENESS")
print("=" * 100)


for dataset in DATASETS:

    expected = int(
        (
            heldout["dataset"]
            == dataset
        ).sum()
    )

    evaluated = results[
        dataset
    ]["n"]

    print(
        f"{dataset.upper():10s} "
        f"expected={expected:5d} "
        f"evaluated={evaluated:5d}"
    )


print()


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


print(
    "Per-image predictions saved:"
)

print(
    OUTPUT_PREDICTIONS_CSV
)


# ============================================================
# NATURAL MACRO
# ============================================================

natural_srcc = {

    d: results[d]["srcc"]

    for d in NATURAL_DATASETS

    if np.isfinite(
        results[d]["srcc"]
    )

}


natural_plcc = {

    d: results[d]["plcc"]

    for d in NATURAL_DATASETS

    if np.isfinite(
        results[d]["plcc"]
    )

}


macro_srcc = (

    float(
        np.mean(
            list(
                natural_srcc.values()
            )
        )
    )

    if len(natural_srcc)
    == len(NATURAL_DATASETS)

    else np.nan
)


macro_plcc = (

    float(
        np.mean(
            list(
                natural_plcc.values()
            )
        )
    )

    if len(natural_plcc)
    == len(NATURAL_DATASETS)

    else np.nan
)


# ============================================================
# WORST NATURAL DATASET
# ============================================================

if len(natural_srcc) == len(
    NATURAL_DATASETS
):

    worst_set = min(
        natural_srcc,
        key=natural_srcc.get,
    )

    worst_srcc = (
        natural_srcc[
            worst_set
        ]
    )

else:

    worst_set = "N/A"

    worst_srcc = np.nan


# ============================================================
# FIND VALID BENCHMARK IMAGE
# ============================================================

sample_path = None


for _, row in heldout.iterrows():

    candidate = resolve_path(
        row["path"]
    )

    if os.path.isfile(candidate):

        sample_path = candidate

        break


if sample_path is None:

    raise RuntimeError(
        "No valid image found for "
        "cost benchmark."
    )


print()
print(
    "Cost benchmark image:"
)

print(
    sample_path
)


# ============================================================
# CUDA HELPERS
# ============================================================

def cuda_sync():

    if device.type == "cuda":

        torch.cuda.synchronize(
            device
        )


def reset_peak_memory():

    if device.type == "cuda":

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(
            device
        )


def peak_memory_mb():

    if device.type != "cuda":

        return np.nan


    return float(
        torch.cuda.max_memory_allocated(
            device
        )
        / (1024 ** 2)
    )


# ============================================================
# SINGLE-IMAGE INFERENCE
# ============================================================

def single_image_forward():

    with torch.inference_mode():

        metric(
            sample_path,
            task_="quality",
        )


# ============================================================
# FLOPs
# ============================================================

print()
print("=" * 100)
print("FLOPs")
print("=" * 100)

print()
print(
    "Attempting FLOPs measurement..."
)


flops_per_image = None


try:

    from torch.utils.flop_counter import (
        FlopCounterMode
    )


    reset_peak_memory()


    counter = FlopCounterMode(
        display=False
    )


    with counter:

        single_image_forward()


    flops_per_image = int(
        counter.get_total_flops()
    )


except Exception as exc:

    print(
        "[INFO] FLOPs counter could not "
        "trace Q-ReAlign."
    )

    print(
        f"Reason: {exc!r}"
    )

    flops_per_image = None


if flops_per_image is not None:

    print(
        f"FLOPs per image: "
        f"{flops_per_image:,}"
    )

    print(
        f"FLOPs per image: "
        f"{flops_per_image / 1e9:.3f} G"
    )

else:

    print(
        "FLOPs per image: N/A"
    )

print()


# ============================================================
# COST WARMUP
# ============================================================

print("=" * 100)
print("COST BENCHMARK")
print("=" * 100)

print()
print(
    f"Warmup iterations : "
    f"{WARMUP_ITERATIONS}"
)

print(
    f"Latency iterations : "
    f"{LATENCY_ITERATIONS}"
)

print(
    f"Throughput batch size : "
    f"{THROUGHPUT_BATCH_SIZE}"
)

print(
    f"Throughput iterations : "
    f"{THROUGHPUT_ITERATIONS}"
)

print()


# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------

print(
    "Running warmup..."
)


for _ in range(
    WARMUP_ITERATIONS
):

    single_image_forward()


cuda_sync()


# ============================================================
# BATCH THROUGHPUT NOTE
# ============================================================

print()
print(
    "NOTE:"
)

print(
    "Q-ReAlign/PyIQA image-path inference "
    "is naturally single-image."
)

print(
    "The official Project 02 throughput "
    "protocol uses fixed batch inference."
)

print(
    "Therefore this script attempts a true "
    "batched Q-ReAlign call only if the metric "
    "accepts a tensor batch."
)

print()


# ============================================================
# TRUE BATCH THROUGHPUT
# ============================================================

def try_get_image_tensor(path):

    """
    Reproduce the common PyIQA image input convention:
    RGB tensor, shape (1, 3, H, W), values in [0, 1].

    IMPORTANT:
    We do NOT resize here.

    The image is loaded at its original dimensions.
    Q-ReAlign/PyIQA remains responsible for model-specific
    preprocessing.

    This function is only used to test whether the metric's
    tensor interface can accept a batch.
    """

    from PIL import Image
    from torchvision.transforms.functional import (
        pil_to_tensor
    )

    image = Image.open(
        path
    ).convert("RGB")


    tensor = (
        pil_to_tensor(image)
        .float()
        / 255.0
    )


    return tensor


# ------------------------------------------------------------
# Try tensor batch interface
# ------------------------------------------------------------

batch_supported = False

batch_tensor = None


try:

    image_tensor = (
        try_get_image_tensor(
            sample_path
        )
    )


    batch_tensor = image_tensor.unsqueeze(
        0
    )


    # Test batch size 1 first.
    with torch.inference_mode():

        test_score = metric(
            batch_tensor.to(
                device
            ),
            task_="quality",
        )


        if (
            test_score
            is not None
            and
            int(
                test_score.numel()
            ) == 1
        ):

            batch_supported = True


except Exception as exc:

    print(
        "[INFO] Tensor batch interface "
        "is not available through this "
        "PyIQA Q-ReAlign wrapper."
    )

    print(
        f"Reason: {exc!r}"
    )


# ============================================================
# LATENCY
# ============================================================

reset_peak_memory()


print()
print(
    "Measuring batch-1 latency..."
)


latencies = []


for _ in range(
    LATENCY_ITERATIONS
):

    cuda_sync()


    start = time.perf_counter()


    single_image_forward()


    cuda_sync()


    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0


    latencies.append(
        elapsed_ms
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


# ============================================================
# THROUGHPUT
# ============================================================

throughput_img_s = np.nan

throughput_method = (
    "not measured"
)


if batch_supported:

    print()
    print(
        "True tensor-batch interface detected."
    )

    print(
        "Attempting fixed-batch throughput..."
    )


    try:

        batch_tensor = (
            batch_tensor
            .repeat(
                THROUGHPUT_BATCH_SIZE,
                1,
                1,
                1,
            )
            .to(device)
        )


        # Warmup batch.
        for _ in range(
            WARMUP_ITERATIONS
        ):

            with torch.inference_mode():

                metric(
                    batch_tensor,
                    task_="quality",
                )


        cuda_sync()


        reset_peak_memory()


        cuda_sync()


        start = time.perf_counter()


        for _ in range(
            THROUGHPUT_ITERATIONS
        ):

            with torch.inference_mode():

                metric(
                    batch_tensor,
                    task_="quality",
                )


        cuda_sync()


        elapsed = (
            time.perf_counter()
            - start
        )


        total_images = (
            THROUGHPUT_BATCH_SIZE
            *
            THROUGHPUT_ITERATIONS
        )


        throughput_img_s = (
            total_images
            / elapsed
        )


        throughput_method = (
            "true tensor batch"
        )


    except Exception as exc:

        print(
            "[WARNING] True batched "
            "throughput failed."
        )

        print(
            f"Reason: {exc!r}"
        )

        throughput_img_s = np.nan


else:

    print()
    print(
        "True tensor-batch inference "
        "is not supported by this "
        "PyIQA Q-ReAlign wrapper."
    )

    print(
        "Throughput is therefore NOT "
        "derived from 1 / latency."
    )

    print(
        "This is intentional: the official "
        "Project 02 throughput metric requires "
        "a fixed batch size."
    )


# ============================================================
# PEAK MEMORY
# ============================================================

peak_mem = peak_memory_mb()


# ============================================================
# FINAL COST RESULTS
# ============================================================

print()
print("=" * 100)
print("COST RESULTS")
print("=" * 100)

print()

if flops_per_image is not None:

    print(
        f"FLOPs / image : "
        f"{flops_per_image / 1e9:.3f} G"
    )

else:

    print(
        "FLOPs / image : N/A"
    )


print(
    f"Latency p50 : "
    f"{latency_p50:.3f} ms"
)


print(
    f"Latency p95 : "
    f"{latency_p95:.3f} ms"
)


if np.isfinite(
    throughput_img_s
):

    print(
        f"Throughput : "
        f"{throughput_img_s:.3f} img/s"
    )

    print(
        f"Throughput batch : "
        f"{THROUGHPUT_BATCH_SIZE}"
    )

    print(
        f"Throughput method : "
        f"{throughput_method}"
    )

else:

    print(
        "Throughput : N/A"
    )

    print(
        f"Required batch : "
        f"{THROUGHPUT_BATCH_SIZE}"
    )

    print(
        "Throughput method : "
        "NOT measured — wrapper does not expose "
        "true batched Q-ReAlign inference."
    )


if np.isfinite(
    peak_mem
):

    print(
        f"Peak VRAM : "
        f"{peak_mem:.1f} MB"
    )

else:

    print(
        "Peak VRAM : N/A"
    )


print(
    f"Parameters : "
    f"{TOTAL_PARAMS / 1e6:.2f} M"
)


print(
    f"Trainable params : "
    f"{TRAINABLE_PARAMS / 1e6:.2f} M"
)


print(
    f"Precision : "
    f"{MODEL_DTYPE}"
)


print(
    f"GPU : "
    f"{gpu_name}"
)


# ============================================================
# ACCURACY SUMMARY
# ============================================================

print()
print("=" * 100)
print("ACCURACY RESULTS")
print("=" * 100)

print()


for dataset in DATASETS:

    r = results[
        dataset
    ]

    print(
        f"{dataset.upper():10s} "
        f"SRCC={r['srcc']:.6f} "
        f"PLCC={r['plcc']:.6f} "
        f"N={r['n']}"
    )


print()

print(
    f"Natural macro SRCC : "
    f"{macro_srcc:.6f}"
)

print(
    f"Natural macro PLCC : "
    f"{macro_plcc:.6f}"
)

print(
    f"Worst natural SRCC : "
    f"{worst_srcc:.6f}"
)

print(
    f"Worst dataset : "
    f"{worst_set.upper()}"
)

print()

print(
    f"AGIQA-3K SRCC : "
    f"{results['agiqa3k']['srcc']:.6f}"
)

print(
    f"AGIQA-3K PLCC : "
    f"{results['agiqa3k']['plcc']:.6f}"
)


# ============================================================
# SAVE PROJECT 02 RESULT TABLE
# ============================================================

rows = []


for dataset in DATASETS:

    r = results[
        dataset
    ]


    rows.append({

        "model": MODEL_ID,

        "baseline_type":
        "external_vlm",

        "dataset":
        dataset,

        "srcc":
        r["srcc"],

        "plcc":
        r["plcc"],

        "n":
        r["n"],

        "missing":
        r["missing"],

        "failed":
        r["failed"],

    })


rows.extend([

    {

        "model": MODEL_ID,

        "baseline_type":
        "external_vlm",

        "dataset":
        "natural_macro",

        "srcc":
        macro_srcc,

        "plcc":
        macro_plcc,

        "n":
        np.nan,

        "missing":
        np.nan,

        "failed":
        np.nan,

    },


    {

        "model": MODEL_ID,

        "baseline_type":
        "external_vlm",

        "dataset":
        "worst_natural",

        "srcc":
        worst_srcc,

        "plcc":
        np.nan,

        "n":
        np.nan,

        "missing":
        np.nan,

        "failed":
        np.nan,

    },

])


# ------------------------------------------------------------
# Add cost information to every row
# ------------------------------------------------------------

cost_info = {

    "flops_per_image":
    (
        flops_per_image
        if flops_per_image is not None
        else np.nan
    ),

    "flops_G":
    (
        flops_per_image / 1e9
        if flops_per_image is not None
        else np.nan
    ),

    "latency_p50_ms":
    latency_p50,

    "latency_p95_ms":
    latency_p95,

    "throughput_img_s":
    throughput_img_s,

    "throughput_batch_size":
    THROUGHPUT_BATCH_SIZE,

    "peak_vram_mb":
    peak_mem,

    "total_params":
    TOTAL_PARAMS,

    "total_params_M":
    TOTAL_PARAMS / 1e6,

    "trainable_params":
    TRAINABLE_PARAMS,

    "frozen_params":
    FROZEN_PARAMS,

    "precision":
    str(MODEL_DTYPE),

    "device":
    str(device),

    "gpu":
    gpu_name,

    "throughput_method":
    throughput_method,

    "model_id":
    MODEL_ID,

    "project_role":
    "external_vlm_baseline",

}


results_df = pd.DataFrame(
    rows
)


for key, value in cost_info.items():

    results_df[key] = value


results_df.to_csv(
    OUTPUT_RESULTS_CSV,
    index=False,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "project":
    "IQA Summer School Project 02",

    "project_role":
    "external_vlm_baseline",

    "model":
    MODEL_ID,

    "pyiqa_metric":
    MODEL_NAME,

    "score_direction":
    "higher_is_better",

    "score_range":
    "[0, 1]",

    "heldout_csv":
    HELDOUT_CSV,

    "heldout_rows":
    int(len(heldout)),

    "score_column":
    SCORE_COLUMN,

    "datasets":
    DATASETS,

    "natural_datasets":
    NATURAL_DATASETS,

    "natural_macro_excludes":
    ["agiqa3k"],

    "throughput_batch_size":
    THROUGHPUT_BATCH_SIZE,

    "warmup_iterations":
    WARMUP_ITERATIONS,

    "latency_iterations":
    LATENCY_ITERATIONS,

    "throughput_iterations":
    THROUGHPUT_ITERATIONS,

    "precision":
    str(MODEL_DTYPE),

    "device":
    str(device),

    "gpu":
    gpu_name,

    "total_parameters":
    int(TOTAL_PARAMS),

    "trainable_parameters":
    int(TRAINABLE_PARAMS),

    "frozen_parameters":
    int(FROZEN_PARAMS),

    "flops_per_image":
    (
        int(flops_per_image)
        if flops_per_image is not None
        else None
    ),

    "latency_p50_ms":
    latency_p50,

    "latency_p95_ms":
    latency_p95,

    "throughput_img_s":
    (
        float(throughput_img_s)
        if np.isfinite(
            throughput_img_s
        )
        else None
    ),

    "peak_vram_mb":
    (
        float(peak_mem)
        if np.isfinite(
            peak_mem
        )
        else None
    ),

    "throughput_method":
    throughput_method,

    "manual_resize":
    False,

    "preprocessing":
    "Q-ReAlign/PyIQA internal preprocessing",

    "prediction_csv":
    OUTPUT_PREDICTIONS_CSV,

    "results_csv":
    OUTPUT_RESULTS_CSV,

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


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 100)
print("FINAL PROJECT 02 BASELINE SUMMARY")
print("=" * 100)

print()

print(
    f"Model : {MODEL_ID}"
)

print(
    "Project role : External VLM baseline"
)

print(
    f"Natural macro SRCC : {macro_srcc:.6f}"
)

print(
    f"Natural macro PLCC : {macro_plcc:.6f}"
)

print(
    f"Worst natural SRCC : {worst_srcc:.6f}"
)

print(
    f"Worst dataset : {worst_set.upper()}"
)

print(
    f"AGIQA-3K SRCC : "
    f"{results['agiqa3k']['srcc']:.6f}"
)

print(
    f"AGIQA-3K PLCC : "
    f"{results['agiqa3k']['plcc']:.6f}"
)

print()

print(
    f"Latency p50 : "
    f"{latency_p50:.3f} ms"
)

print(
    f"Latency p95 : "
    f"{latency_p95:.3f} ms"
)

if np.isfinite(
    throughput_img_s
):

    print(
        f"Throughput : "
        f"{throughput_img_s:.3f} img/s"
    )

else:

    print(
        "Throughput : N/A"
    )


print(
    f"Throughput batch : "
    f"{THROUGHPUT_BATCH_SIZE}"
)

print(
    f"Peak VRAM : "
    f"{peak_mem:.1f} MB"
    if np.isfinite(peak_mem)
    else
    "Peak VRAM : N/A"
)

print(
    f"Parameters : "
    f"{TOTAL_PARAMS / 1e6:.2f} M"
)

print(
    f"Precision : "
    f"{MODEL_DTYPE}"
)

print(
    f"GPU : "
    f"{gpu_name}"
)

print()

print(
    "Prediction CSV:"
)

print(
    OUTPUT_PREDICTIONS_CSV
)

print()

print(
    "Results CSV:"
)

print(
    OUTPUT_RESULTS_CSV
)

print()

print(
    "Metadata JSON:"
)

print(
    OUTPUT_METADATA_JSON
)

print()

print("=" * 100)
print("BENCHMARK FINISHED")
print("=" * 100)