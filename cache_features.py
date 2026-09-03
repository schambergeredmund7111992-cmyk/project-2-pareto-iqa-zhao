"""Encode every image once with the frozen backbone and store the tokens.

The backbone never trains, so re-encoding the same 31,323 images every epoch
is work we already did. Caching turns the cost of an experiment from an hour
of JPEG decoding into a few seconds of matrix multiplies. It does not touch
the cost we report — benchmark.py still measures the full image -> encoder ->
head path, which is what a serving system would actually run.

    python cache_features.py --data ./data/train.csv --out ./features_clip-base.npy

Stored as float16: the head layer-norms its input immediately, so the extra
precision buys nothing and the file halves.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import IQADataset
from train import load_backbone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbone", default="clip-base")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, image_size, feature_dim = load_backbone(args.backbone, args.weights, device)
    family = "siglip" if args.backbone.startswith("siglip") else "clip"

    dataset = IQADataset(args.data, image_size=image_size, backbone=family)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    print(f"{args.backbone} at {image_size}px on {device}, {len(dataset)} rows")

    chunks = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            tokens = backbone(pixel_values=batch["image"].to(device)).last_hidden_state
            chunks.append(tokens.half().cpu().numpy())
            if i % 50 == 0:
                print(f"  {i * args.batch_size}/{len(dataset)}", flush=True)

    features = np.concatenate(chunks)
    np.save(args.out, features)
    print(f"saved -> {args.out}  shape {features.shape}  "
          f"{features.nbytes / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
