"""Train a head on cached frozen features.

    python cache_features.py --data ./data/train.csv --out ./features_clip-base.npy
    python train_cached.py --data ./data/train.csv --features ./features_clip-base.npy --head mlp

The encoder is frozen, so its output is a constant of the CSV. Reading it back
costs nothing and the epoch becomes a few seconds of matrix multiplies instead
of an hour of JPEG decoding. Nothing about the reported cost changes:
benchmark.py still times the full image -> encoder -> head path.

Heads, all at ~199k parameters so the comparison is structure and not capacity:

  mlp        CLS token -> MLP.  The baseline, reproduced off the cache.
  attnpool   pool the patch features by learned weights, then score once.
  iplnr      score every patch, then pool the scores. WaDIQaM and FAST-VQA's
             IP-NLR argue this way round: patches of different quality dilute
             each other in feature space, so a soft corner is averaged into a
             sharp frame before anything scores it.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset import split_by, make_sampler


class CachedFeatures(Dataset):
    """The CSV rows, with the encoder's output read from disk instead of computed."""

    def __init__(self, csv, features, score_column="scaled_subjective_score"):
        self.rows = csv.reset_index(drop=True) if isinstance(csv, pd.DataFrame) \
            else pd.read_csv(csv)
        self.features = features
        self.score_column = score_column
        if "cache_index" not in self.rows.columns:
            self.rows["cache_index"] = np.arange(len(self.rows))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return {
            "features": torch.from_numpy(
                self.features[int(row["cache_index"])].astype(np.float32)),
            "target": torch.tensor(float(row[self.score_column]), dtype=torch.float32),
            "reference": str(row["reference"]),
            "dataset": str(row.get("dataset", "")),
            "level": int(row["level"]) if pd.notna(row.get("level")) else -1,
        }

    def subset(self, rows):
        return CachedFeatures(rows, self.features, self.score_column)


class MLPHead(nn.Module):
    """The baseline: CLS token straight into the MLP."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        return self.net(tokens[:, 0]).squeeze(-1)


class AttnPoolHead(nn.Module):
    """Pool the patch features by learned weights, then score the pooled vector."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.score = nn.Linear(input_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        tokens = self.norm(tokens)
        weights = self.score(tokens).softmax(dim=1)
        return self.mlp((tokens * weights).sum(dim=1)).squeeze(-1)


class IPNLRHead(nn.Module):
    """Score every patch, then pool the scores. Same modules as AttnPoolHead,
    same parameter count, only the order of the two steps differs."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.score = nn.Linear(input_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens):
        tokens = self.norm(tokens)
        per_patch = self.mlp(tokens).squeeze(-1)
        weights = self.score(tokens).softmax(dim=1).squeeze(-1)
        return (per_patch * weights).sum(dim=1)


HEADS = {"mlp": MLPHead, "attnpool": AttnPoolHead, "iplnr": IPNLRHead}


def evaluate(head, loader, device):
    head.eval()
    predictions, targets, references, datasets = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            predictions.append(head(batch["features"].to(device)).cpu().numpy())
            targets.append(batch["target"].numpy())
            references.extend(batch["reference"])
            datasets.extend(batch["dataset"])
    head.train()
    frame = pd.DataFrame({"p": np.concatenate(predictions),
                          "t": np.concatenate(targets),
                          "ref": references, "dataset": datasets})

    per_dataset = {}
    for name, group in frame.groupby("dataset"):
        if len(group) < 2 or group["t"].nunique() < 2:
            continue
        per_dataset[name] = {
            "srcc": float(stats.spearmanr(group["p"], group["t"]).correlation),
            "plcc": float(stats.pearsonr(group["p"], group["t"]).statistic),
            "n": int(len(group)),
        }

    per_reference = [stats.spearmanr(g["p"], g["t"]).correlation
                     for _, g in frame.groupby("ref")
                     if len(g) >= 8 and g["t"].nunique() > 1]

    srccs = [s["srcc"] for s in per_dataset.values()]
    return {
        "per_dataset": per_dataset,
        "macro_srcc": float(np.mean(srccs)) if srccs else None,
        "macro_plcc": float(np.mean([s["plcc"] for s in per_dataset.values()])) if srccs else None,
        "worst_srcc": float(min(srccs)) if srccs else None,
        "worst_dataset": min(per_dataset, key=lambda k: per_dataset[k]["srcc"]) if srccs else None,
        "srcc_per_reference": float(np.mean(per_reference)) if per_reference else None,
        "n_references": len(per_reference),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--head", default="mlp", choices=sorted(HEADS))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--split", default="reference", choices=["reference", "random"])
    ap.add_argument("--score-column", default="scaled_subjective_score")
    ap.add_argument("--sampler", default="random",
                    choices=["random", "balanced", "by_level", "by_dataset"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default=None,
                    help="write the split out here, so the table can be reconstructed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = np.load(args.features, mmap_mode="r")
    dataset = CachedFeatures(args.data, features, args.score_column)
    if len(dataset) != len(features):
        raise SystemExit(
            f"{len(dataset)} rows against {len(features)} cached features — "
            "the cache was built from a different CSV")

    train_set, val_set = split_by(dataset, args.split, fraction=0.2, seed=args.seed)

    if args.manifest:
        pd.concat([
            train_set.rows.assign(split="train"),
            val_set.rows.assign(split="val"),
        ])[["cache_index", "dataset", "reference", "split"]].to_csv(args.manifest, index=False)
        print(f"manifest -> {args.manifest}")

    sampler = make_sampler(train_set, args.sampler, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              shuffle=sampler is None)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    feature_dim = features.shape[-1]
    head = HEADS[args.head](feature_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    print(f"head {args.head} on cached {features.shape[1]}x{feature_dim} features")
    print(f"train {len(train_set)}  held out {len(val_set)}  "
          f"(split by {args.split}, sampling {args.sampler}, seed {args.seed})")
    print(f"{sum(p.numel() for p in head.parameters()):,} trainable parameters")

    for epoch in range(args.epochs):
        losses = []
        for batch in train_loader:
            loss = loss_fn(head(batch["features"].to(device)),
                           batch["target"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scores = evaluate(head, val_loader, device)
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}", flush=True)
        for name, row in sorted(scores["per_dataset"].items()):
            print(f"    {name:<14s} n {row['n']:>6d}   "
                  f"SRCC {row['srcc']:.4f}   PLCC {row['plcc']:.4f}")
        if len(scores["per_dataset"]) > 1:
            print(f"    {'macro':<14s} {'':>8s}   SRCC {scores['macro_srcc']:.4f}   "
                  f"PLCC {scores['macro_plcc']:.4f}   "
                  f"worst {scores['worst_srcc']:.4f} on {scores['worst_dataset']}")
        if scores["srcc_per_reference"] is not None:
            print(f"    {'within-ref':<14s} {'':>8s}   SRCC {scores['srcc_per_reference']:.4f}"
                  f"   ({scores['n_references']} references)")

    if args.out:
        torch.save({"head": head.state_dict(), "kind": args.head,
                    "feature_dim": feature_dim}, args.out)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
