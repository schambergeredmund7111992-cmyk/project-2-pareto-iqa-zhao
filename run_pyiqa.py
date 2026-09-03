"""Score the validation split with a pyiqa metric, and time what it costs.

    python run_pyiqa.py --metric dbcnn --data ./train.csv --manifest ./split_manifest.csv

Two halves, the same two the project reports for its own designs. Accuracy is
SRCC and PLCC per dataset, their macro and the worst of them — pooled across
releases would partly measure the offset between their scales. Cost is
batch-1 latency and throughput on the same device, with a warm-up discarded,
because the first design measured pays for backend initialisation.

Metrics are read at their native resolution: MUSIQ and others are trained on
the aspect ratio the camera produced, and resizing to a square changes what
they see. That makes batching awkward, so this runs one image at a time.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyiqa
import torch
from PIL import Image
from scipy import stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only N images, drawn at random — for a smoke test")
    ap.add_argument("--max-side", type=int, default=None,
                    help="downscale so the longer side is at most this, keeping the "
                         "aspect ratio; omit to hand the metric the file untouched")
    ap.add_argument("--timing-iterations", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None, help="write per-image scores here")
    args = ap.parse_args()

    rows = pd.read_csv(args.data)
    rows["cache_index"] = np.arange(len(rows))
    manifest = pd.read_csv(args.manifest)
    keep = set(manifest.loc[manifest["split"] == args.split, "cache_index"])
    rows = rows[rows["cache_index"].isin(keep)].reset_index(drop=True)
    if args.limit:
        rows = rows.sample(args.limit, random_state=0).reset_index(drop=True)

    device = torch.device(args.device)
    metric = pyiqa.create_metric(args.metric, device=device)
    lower_better = bool(getattr(metric, "lower_better", False))

    convention = f"longer side <= {args.max_side}px" if args.max_side else "native resolution"
    print(f"{args.metric} on {len(rows)} images from the {args.split} split, "
          f"{convention} ({'lower' if lower_better else 'higher'} is better)")

    # A metric can refuse an individual image - BRISQUE divides by the pixel
    # variance, and a heavily distorted frame can come out flat. Record the
    # failure and carry on: dropping one image is honest, losing the run is not.
    def load(path):
        """The image as the metric should see it.

        These metrics were trained at a fixed scale - roughly 224 for the CNN
        heads - and SPAQ ships 4032x3024 phone captures, fifteen to twenty-seven
        times the area of a KonIQ frame. Handing one of those to a network whose
        receptive field was sized for a small image measures the wrong thing:
        on 300 SPAQ images DBCNN scores 0.3050 at native resolution and 0.8739
        at 224, HyperIQA 0.1680 against 0.7902, CLIP-IQA 0.1376 against 0.6084,
        and MUSIQ runs out of memory trying. So the convention is stated per run
        rather than left to whatever the file happens to contain.
        """
        if args.max_side is None:
            return path
        image = Image.open(path).convert("RGB")
        image.thumbnail((args.max_side, args.max_side), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).permute(2, 0, 1)[None].to(device)

    scores, failures = [], []
    start = time.perf_counter()
    for i, path in enumerate(rows["path"]):
        try:
            with torch.no_grad():
                scores.append(float(metric(load(path))))
        except Exception as exc:
            scores.append(float("nan"))
            failures.append((path, str(exc).split("\n")[0]))
        if i % 500 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    if failures:
        print(f"\n  {len(failures)} images could not be scored and are excluded:")
        for path, why in failures[:5]:
            print(f"    {path}: {why}")
        if len(failures) > 5:
            print(f"    ... and {len(failures) - 5} more")
    wall = time.perf_counter() - start
    rows["prediction"] = scores

    # A metric where lower is better ranks the opposite way; flip it so every
    # correlation in the table reads the same direction.
    sign = -1.0 if lower_better else 1.0

    rows = rows[rows["prediction"].notna()].reset_index(drop=True)

    per_dataset = {}
    for name, group in rows.groupby("dataset"):
        if len(group) < 2 or group["scaled_subjective_score"].nunique() < 2:
            continue
        p = sign * group["prediction"].to_numpy()
        t = group["scaled_subjective_score"].to_numpy()
        per_dataset[name] = {
            "srcc": float(stats.spearmanr(p, t).correlation),
            "plcc": float(stats.pearsonr(p, t).statistic),
            "n": len(group),
        }

    per_reference = [
        stats.spearmanr(sign * g["prediction"], g["scaled_subjective_score"]).correlation
        for _, g in rows.groupby("reference")
        if len(g) >= 8 and g["scaled_subjective_score"].nunique() > 1
    ]

    # Cost, measured after the scoring loop so the model is warm.
    sample = load(rows["path"].iloc[0])
    for _ in range(5):
        with torch.no_grad():
            metric(sample)
    latencies = []
    for _ in range(args.timing_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            metric(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1e3)

    print()
    for name, row in sorted(per_dataset.items()):
        print(f"    {name:<14s} n {row['n']:>6d}   "
              f"SRCC {row['srcc']:.4f}   PLCC {row['plcc']:.4f}")
    # AGIQA-3K stays out of the macro: generated images fail in ways neither a
    # camera nor a codec produces, so averaging it in would hide both.
    in_macro = {k: v for k, v in per_dataset.items() if k != "agiqa3k"}
    srccs = [s["srcc"] for s in in_macro.values()]
    if len(in_macro) > 1:
        worst = min(in_macro, key=lambda k: in_macro[k]["srcc"])
        print(f"    {'macro':<14s} {'':>8s}   SRCC {np.mean(srccs):.4f}   "
              f"PLCC {np.mean([s['plcc'] for s in in_macro.values()]):.4f}   "
              f"worst {min(srccs):.4f} on {worst}"
              f"{'   (agiqa3k excluded)' if 'agiqa3k' in per_dataset else ''}")
    if per_reference:
        print(f"    {'within-ref':<14s} {'':>8s}   SRCC {np.mean(per_reference):.4f}"
              f"   ({len(per_reference)} references)")
    print()
    print(f"  latency batch 1     {np.percentile(latencies, 50):.1f} ms p50, "
          f"{np.percentile(latencies, 95):.1f} ms p95")
    print(f"  throughput          {len(rows) / wall:.1f} img/s "
          "(the scoring loop, one image at a time, including image loading)")
    params = sum(p.numel() for p in metric.parameters()) if hasattr(metric, "parameters") else 0
    if params:
        print(f"  parameters          {params:,}")

    if args.out:
        rows[["cache_index", "dataset", "reference", "path",
              "scaled_subjective_score", "prediction"]].to_csv(args.out, index=False)
        print(f"\nper-image scores -> {args.out}")


if __name__ == "__main__":
    main()
