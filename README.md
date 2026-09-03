# Third-party NR metrics on the Project 2 splits

Fork of [dreminm/project-2-pareto-iqa](https://github.com/dreminm/project-2-pareto-iqa).
The upstream README describes the project; this one covers what was added here.

Every no-reference metric in [pyiqa](https://github.com/chaofengc/IQA-PyTorch),
scored on this project's validation split and its held-out sets, with the cost
of running each one. 51 checkpoints, 48 of them reportable.

Results: `pyiqa_all_rows.csv` (31 columns, same layout as the shared sheet).

---

## What was added

| file | |
| --- | --- |
| `run_pyiqa.py` | score a pyiqa metric over a split, print SRCC/PLCC per dataset |
| `benchmark_pyiqa.py` | FLOPs, latency, throughput, peak memory — aligned with `benchmark.py` |
| `cache_features.py` | encode every image once with the frozen backbone |
| `train_cached.py` | train a head on those cached features; three heads to choose from |
| `train_attnpool.py` | `train.py` with attention pooling over patch tokens |
| `run_native.sh` | queue of metrics; skips outputs already on disk, so it resumes |

Per-metric outputs are `<metric>_val.csv` / `<metric>_heldout.csv` (per-image
scores) and the matching `.log` (correlations and the run's conditions).

---

## Environment

Ubuntu 24, NVIDIA L40, driver 550.120 / CUDA 12.4.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
uv pip install pyiqa

# pyiqa pulls torch 2.13.0, which is built for CUDA 13 and will not see this
# driver - torch.cuda.is_available() returns False and everything silently
# falls back to CPU. Put it back:
uv pip install --reinstall "torch==2.6.0" torchvision \
  --index-url https://mirror.nju.edu.cn/pytorch/whl/cu124 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --index-strategy unsafe-best-match

export HF_ENDPOINT=https://hf-mirror.com   # weights time out without this
```

Four things cost a day between them and are worth knowing:

- **`uv pip install torch` without `--reinstall` does nothing.** It prints
  `Checked 1 package in 4ms` and leaves the existing version alone.
- **uv only looks in the first index that has a package.** With an extra index
  attached it will pick a higher version from the wrong one; `--index-strategy
  unsafe-best-match` makes it look at all of them.
- **transformers 5.16 needs torch ≥ 2.5** to enable its PyTorch backend at all,
  and ≥ 2.6 to load `.bin` checkpoints.
- **`tmux new -d` does not read `~/.bashrc`**, so `HF_ENDPOINT` has to be set
  inside the script. Without it the first download of any new checkpoint hangs
  and the whole queue entry produces an empty log.

---

## Running it

```bash
# one metric, both splits
python run_pyiqa.py --metric musiq --data ./train.csv \
  --manifest ./split_manifest.csv --out musiq_val.csv
python run_pyiqa.py --metric musiq --data ./heldout.csv \
  --manifest ./heldout_manifest.csv --out musiq_heldout.csv

# a queue of them on one card
./run_native.sh 3 liqe unique arniqa maniqa topiq_nr

# cost, every metric in one sitting
python benchmark_pyiqa.py --metrics liqe unique arniqa --out cost.csv
```

The manifest restricts scoring to the rows it marks `split=='val'` — 6,240 of
the 31,323 in `train.csv`. The training split is what our own heads are fitted
on, so a third-party metric scored there would not be measured on the same data.

---

## Two decisions worth stating

**Native resolution, no manual resizing.** pyiqa's `InferenceModel` reads the
file and adds a batch dimension; it does not resize. Each architecture handles
what it needs internally — HyperIQA uniform-crops to 224, DBCNN's `preprocess`
is normalisation only because bilinear pooling accepts any size. Resizing on
top of that is a second, wrong step.

An earlier pass did resize, to 224. On 300 SPAQ images that moved DBCNN from
0.3050 to 0.8739, HyperIQA from 0.1680 to 0.7902, and MUSIQ from an
out-of-memory error to a number. SPAQ ships 4032×3024 phone captures — fifteen
to twenty-seven times a KonIQ frame — so the effect is real and large. But it
is not a correction: MUSIQ exists to read native multi-scale input, and NIQE
measures per-pixel statistics that downsampling erases, and both score lower at
224. The `*_224.csv` files are kept for that comparison and are not the
reported numbers.

**AGIQA-3K is never in the macro.** It gets its own column. Generated images
fail in ways neither a camera nor a codec produces, so averaging them in hides
both. This follows `datasets.md`.

---

## What the numbers say

Held-out macro is CLIVE, CSIQ and TID2013 — the sets nothing here trained on.

| | in-domain | held out | GFLOPs | img/s |
| --- | --- | --- | --- | --- |
| LIQE-mix | 0.7472 | **0.8999** | 8.8 | 369 |
| UNIQUE | 0.7782 | 0.8926 | **7.4** | 1599 |
| LIQE | 0.6533 | 0.7767 | 8.8 | 2234 |
| QualiCLIP+ (CLIVE) | 0.6208 | 0.7487 | — | — |
| CLIP-IQA+ | 0.7087 | 0.7188 | 24.1 | 1303 |
| MANIQA (KonIQ) | 0.4482 | 0.6368 | 6055.4 | 3.0 |
| TReS (KonIQ) | 0.5635 | 0.5787 | 1995.2 | 5.0 |

**The cheapest are the best.** The frontier runs PIQE → BRISQUE → NIQE →
PaQ-2-PiQ → MetaIQA → UNIQUE → LIQE-mix, and every point on it is under 10
GFLOPs. MANIQA costs 688× what UNIQUE does, needs 23.8 GB of memory against
205 MB, runs at 3 img/s against 1599, and scores 0.26 lower on the held-out sets.

**Ranking depends on where you score it.** ARNIQA is 3rd in-domain and 9th
held-out; MANIQA (KonIQ) is 20th and 8th. The in-domain macro rewards having
trained on the target domain, and most of these checkpoints were trained on
KonIQ.

**TID2013 is the common floor.** Median SRCC across all 51 is 0.449 there,
against 0.726 on CLIVE. `datasets.md` predicted this: frozen features lose
ground as a dataset gets harder.

**Training on the wrong distribution is worse than not training.** NIQE reaches
0.4631 held-out having never seen a MOS score; BRISQUE, which fitted an SVR to
them on LIVE in 2012, reaches 0.4121. ARNIQA's seven checkpoints make the same
point within one architecture — the ones trained on synthetic distortion
(CSIQ 0.7050, TID 0.6957, KADID 0.6821) beat the ones trained on authentic
capture (KonIQ 0.6354, CLIVE 0.4750, SPAQ 0.4440) on a held-out set that is
mostly synthetic.

---

## Caveats in the table

- **KonIQ column.** DBCNN, HyperIQA, MUSIQ, CNNIQA, TReS and others ship
  KonIQ-trained weights. MUSIQ scores 0.9456 there against the 0.916 ± 0.002
  its own paper reports — fifteen standard deviations above, which is what
  training-set overlap looks like. Our split is by reference over the whole
  release and does not respect KonIQ's official train/test boundary.
- **Latency and throughput were measured in two sittings** on a shared machine.
  FLOPs and peak memory are properties of the graph and came out identical on
  repeat measurement; the two timing columns are only comparable within their
  batch. Repeat runs gave UNIQUE 4.0 ms then 13.3 ms.
- **AFINE-NR is not reportable.** It requires height and width divisible by 32,
  so 1,160 of 1,162 CLIVE images failed. Its apparent held-out macro of 0.8206
  comes from an SRCC of 1.0000 over the two that survived.
- **DMM and FGResQ** produced no output — interface incompatible with
  single-image scoring as called here.
- **ILNIQE** was not finished. NIQE alone took five hours over 6,240 images on
  this shared machine and ILNIQE is heavier.
- **BRISQUE and NIQE skip three AGIQA-3K images** (`DALLE2_normal_253`,
  `glide_normal_101`, `glide_normal_263`): they are flat enough that the pixel
  variance is zero and BRISQUE divides by it, while NIQE's SVD fails to
  converge.

---

## Feature caching

The backbone never trains, so re-encoding the same 25,083 images every epoch is
work already done. GPU utilisation sat at 6% while the CPU decoded JPEGs.

```bash
python cache_features.py --data ./train.csv --out ./features_clip-base.npy
python train_cached.py --data ./train.csv --features ./features_clip-base.npy \
  --head mlp --sampler by_dataset --epochs 5 --seed 42 --split reference \
  --manifest ./split_manifest.csv
```

Five epochs go from an hour and a half to seconds. The cache is 9.5 GB
(31,323 × 197 × 768, float16) and is gitignored.

This changes the cost of experimenting, not the cost of any design being
reported — `benchmark.py` still times the full image → encoder → head path.

Reproducing the baseline off the cache gives macro 0.7975 against 0.8061 from
`train.py`. The gap is the CLS token against `pooler_output`, which is that
token through one more layernorm.

`train_cached.py` also writes the split to a manifest, which `datasets.md` asks
for and `train.py` has no option for.

---

## Reproducing the splits

```bash
python prepare_data.py ~/iqa-data/{kadid10k,koniq10k,spaq} --out ./train.csv
python prepare_data.py ~/iqa-data/{clive,csiq,tid2013,agiqa3k} --out ./heldout.csv
```

31,323 and 8,010 rows. `split_by` is a function of the CSV and `--seed 42`; the
resulting rows are in `split_manifest.csv`. Use the same CSV file rather than
regenerating it — a different row order gives a different split even with the
same seed.
