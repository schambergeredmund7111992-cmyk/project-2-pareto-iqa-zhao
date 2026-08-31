"""Measure what a design costs to run: FLOPs, latency, throughput, memory.

    python benchmark.py --backbone clip-base
    python benchmark.py --backbone clip-large --batch-size 16

This project is about the trade-off between accuracy and inference cost, so
the cost side needs measuring, not estimating. Two numbers that disagree on
purpose: FLOPs count arithmetic, latency counts what a server waits for, and
they rank designs differently — memory traffic, kernel launches and the
shape of the model all show up in one and not the other.

**Measure everything on one device at one precision.** A cost column
collected across machines is not a cost column. And the first design you
measure pays for backend initialization — kernel compilation, memory pools
— so the loop below runs a throwaway pass first.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn

from train import BACKBONES, QualityMLP, load_backbone


def measure_flops(forward, x) -> int | None:
    """FLOPs of one forward pass, or None if the counter cannot see this path."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with counter:
            forward(x)
        return int(counter.get_total_flops())
    except Exception:
        return None


def reset_peak_memory(device: torch.device) -> None:
    """Forget the high-water mark, so the number belongs to this design only."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def peak_memory_mb(device: torch.device) -> float | None:
    """Peak allocation in MB, on the accelerators that report one.

    CUDA and MPS both track it. On CPU there is no allocator to ask — the
    process resident size is the whole interpreter, not this model — so the
    column stays empty rather than carrying a number that means something else.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def benchmark(backbone, head, image_size, device, batch_size, iterations) -> dict:
    def forward(x):
        with torch.no_grad():
            return head(backbone(pixel_values=x).pooler_output.float())

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    reset_peak_memory(device)
    one = torch.randn(1, 3, image_size, image_size, device=device)
    batch = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for _ in range(5):
        forward(one)
        forward(batch)

    flops = measure_flops(forward, one)

    latencies = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        forward(one)
        sync()
        latencies.append((time.perf_counter() - start) * 1e3)

    sync()
    start = time.perf_counter()
    for _ in range(max(5, iterations // 3)):
        forward(batch)
    sync()
    elapsed = time.perf_counter() - start
    throughput = batch_size * max(5, iterations // 3) / elapsed

    return {
        "flops_per_image": flops,
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "throughput_img_s": float(throughput),
        "peak_memory_mb": peak_memory_mb(device),
        "params_backbone": sum(p.numel() for p in backbone.parameters()),
        "params_head": sum(p.numel() for p in head.parameters() if p.requires_grad),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="clip-base", choices=sorted(BACKBONES))
    ap.add_argument("--weights", default=None, help="local checkpoint directory")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    backbone, image_size, feature_dim = load_backbone(args.backbone, args.weights, device)
    head = QualityMLP(feature_dim, args.hidden_dim).to(device).eval()

    # The first measured design pays for backend initialization; throw it away.
    benchmark(backbone, head, image_size, device, args.batch_size, 5)
    row = benchmark(backbone, head, image_size, device, args.batch_size, args.iterations)

    print(f"{args.backbone} at {image_size}px on {device}")
    flops = f"{row['flops_per_image'] / 1e9:.1f} G" if row["flops_per_image"] else "n/a"
    print(f"  FLOPs per image     {flops}")
    print(f"  latency batch 1     {row['latency_ms_p50']:.1f} ms p50, "
          f"{row['latency_ms_p95']:.1f} ms p95")
    print(f"  throughput          {row['throughput_img_s']:.1f} img/s "
          f"at batch {args.batch_size}")
    if row["peak_memory_mb"]:
        print(f"  peak memory         {row['peak_memory_mb']:.0f} MB")
    print(f"  parameters          {row['params_backbone']:,} frozen + "
          f"{row['params_head']:,} trained")


if __name__ == "__main__":
    main()
