"""Measure what a pyiqa metric costs to run, the same way benchmark.py does.

    python benchmark_pyiqa.py --metrics dbcnn hyperiqa musiq clipiqa brisque niqe

Same numbers as benchmark.py — FLOPs, batch-1 latency, throughput, peak memory,
parameters — so the rows can sit in one table. Run every metric in one
invocation: a cost column collected across machines, or against a different
background load, is not a cost column. The first metric measured pays for
backend initialisation, so each one gets a throwaway pass first.

Two things differ from benchmark.py, and both are the metrics' own doing.
Several read at native resolution, so their cost depends on the image rather
than being a constant of the design; --image-size fixes one input for every row
so the column compares designs and not datasets. And a metric that wraps its
preprocessing inside the forward may hide part of its work from the FLOP
counter, which is then reported as n/a rather than guessed at.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import pyiqa
import torch


def measure_flops(forward, x):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        counter = FlopCounterMode(display=False)
        with counter:
            forward(x)
        return int(counter.get_total_flops())
    except Exception:
        return None


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_one(metric, device, image_size, batch_size, iterations):
    def forward(x):
        with torch.no_grad():
            return metric(x)

    one = torch.rand(1, 3, image_size, image_size, device=device)
    batch = torch.rand(batch_size, 3, image_size, image_size, device=device)

    for _ in range(5):
        forward(one)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    flops = measure_flops(forward, one)

    latencies = []
    for _ in range(iterations):
        sync(device)
        start = time.perf_counter()
        forward(one)
        sync(device)
        latencies.append((time.perf_counter() - start) * 1e3)

    # Not every metric accepts a batch; fall back to one at a time and say so.
    try:
        for _ in range(3):
            forward(batch)
        sync(device)
        start = time.perf_counter()
        rounds = max(5, iterations // 3)
        for _ in range(rounds):
            forward(batch)
        sync(device)
        throughput = batch_size * rounds / (time.perf_counter() - start)
        batched = True
    except Exception:
        sync(device)
        start = time.perf_counter()
        for _ in range(iterations):
            forward(one)
        sync(device)
        throughput = iterations / (time.perf_counter() - start)
        batched = False

    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    params = sum(p.numel() for p in metric.parameters()) if hasattr(metric, "parameters") else 0

    return {"flops": flops, "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)), "throughput": throughput,
            "batched": batched, "peak_mb": peak, "params": params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/cost/cost_pyiqa.csv")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"every row at {args.image_size}px on {device}, "
          f"batch {args.batch_size} for throughput\n")

    rows = []
    for name in args.metrics:
        try:
            metric = pyiqa.create_metric(name, device=device)
        except Exception as exc:
            print(f"{name:<16s} could not be created: {exc}\n")
            continue
        # Throw the first measurement away — it pays for backend init.
        bench_one(metric, device, args.image_size, args.batch_size, 5)
        r = bench_one(metric, device, args.image_size, args.batch_size, args.iterations)
        r["metric"] = name
        r["image_size"] = args.image_size
        rows.append(r)

        flops = f"{r['flops'] / 1e9:.1f} G" if r["flops"] else "n/a"
        note = "" if r["batched"] else "  (one at a time: this metric refused a batch)"
        print(f"{name}")
        print(f"  FLOPs per image     {flops}")
        print(f"  latency batch 1     {r['p50']:.1f} ms p50, {r['p95']:.1f} ms p95")
        print(f"  throughput          {r['throughput']:.1f} img/s{note}")
        if r["peak_mb"]:
            print(f"  peak memory         {r['peak_mb']:.0f} MB")
        if r["params"]:
            print(f"  parameters          {r['params']:,}")
        print()
        del metric
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
