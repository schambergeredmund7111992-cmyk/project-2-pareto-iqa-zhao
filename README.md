# Pareto-optimal IQA — CLIP-based architectures, third-party NR baselines

Fork of [dreminm/project-2-pareto-iqa](https://github.com/dreminm/project-2-pareto-iqa),
the quick-start repository for Project 02 of the
[IQA Summer School](https://dreminm.github.io/iqa-summer-school/project-2.html).
The upstream scaffold (train / benchmark / data tooling) is untouched here; this
fork adds **third-party no-reference metrics as external baselines**: every NR
metric in [pyiqa](https://github.com/chaofengc/IQA-PyTorch) scored on the
project's validation and held-out splits — 51 checkpoints, 48 reportable — plus
Q-Align and Q-ReAlign, two VLM-based metrics, each with its own script and
results. The original upstream README is kept at
[README_upstream.md](README_upstream.md).

## The project

An accurate quality metric is a large fine-tuned network; a cheap one loses
accuracy. Project 02 asks which architectures on top of a **frozen
CLIP-family encoder** are Pareto-optimal in quality and inference cost.
Quality is SRCC/PLCC on CLIVE, CSIQ and TID2013 (their mean is the headline
macro, the worst of the three is reported alongside); AGIQA-3K gets its own
column and never enters the macro. Cost is FLOPs, latency, throughput and
peak memory. Which datasets train, which are held out and how cost is
counted: [datasets.md](datasets.md).

The baselines in this fork score the *same splits under the same conventions*,
so the project's own designs can be read against what already exists.

## Layout map

| path | what it holds | how to read it |
| --- | --- | --- |
| repo root | the scripts (see below) and `datasets.md` | start with the Scripts table |
| [`data/`](data/) | `train.csv`, `heldout.csv`, `split_manifest.csv`, `heldout_manifest.csv` | the inputs every script consumes |
| [`results/pyiqa_all_rows.xlsx`](results/pyiqa_all_rows.xlsx) | the consolidated sheet: 51 rows × 31 columns, same layout as the shared spreadsheet | the one table to look at |
| [`results/pyiqa/`](results/pyiqa/) | per-image scores and correlation logs for every pyiqa metric | `<metric>_val.csv` / `<metric>_heldout.csv` = per-image scores; the matching `.log` = SRCC/PLCC per dataset, macro, worst, cost |
| [`results/qalign/`](results/qalign/), [`results/qrealign/`](results/qrealign/) | the two VLM baselines' predictions, results row and metadata | `*_project02_results.csv` is the headline row |
| [`results/cost/`](results/cost/) | FLOPs / latency / throughput / memory for every metric | `cost_pyiqa.csv` + the two batch runs; the log states the conditions |
| [`logs/`](logs/) | training and benchmark sessions of our own heads | `baseline.log`, `attnpool.log`, `cached_mlp.log`, `cache.log`, `baseline_bench*.log` |

## Scripts

| file | origin | what it does |
| --- | --- | --- |
| `train.py` | upstream | frozen CLIP + an MLP head, trained to predict quality |
| `benchmark.py` | upstream | what a design costs: FLOPs, latency, throughput, memory |
| `prepare_data.py`, `download_data.py`, `dataset.py` | upstream | labels → one CSV; fetch datasets; splits and samplers |
| `train_attnpool.py` | ours | `train.py` with attention pooling over patch tokens |
| `cache_features.py`, `train_cached.py` | ours | encode images once with the frozen backbone; train a head on the cached features |
| `run_pyiqa.py` | ours | score a pyiqa metric over a split, print SRCC/PLCC per dataset, macro, worst, cost |
| `benchmark_pyiqa.py` | ours | FLOPs, latency, throughput, peak memory for pyiqa metrics, aligned with `benchmark.py` |
| `run_native.sh` | ours | queue of metrics at native resolution; skips outputs already on disk, so it resumes |
| `rerun_all.sh`, `run_rest.sh` | ours | the same queue at a fixed 224px input (see Native vs 224px) |
| `qalign.py`, `qrealign.py` | ours | Q-Align / Q-ReAlign external VLM baselines on the held-out split |

## Results at a glance

Headline rows from [`results/pyiqa_all_rows.xlsx`](results/pyiqa_all_rows.xlsx)
— in-domain is the validation macro, held out is the CLIVE/CSIQ/TID2013 macro:

| | in-domain | held out | GFLOPs | img/s |
| --- | --- | --- | --- | --- |
| LIQE-mix (multi-dataset) | 0.7472 | **0.8999** | 8.8 | 369 |
| UNIQUE (multi-dataset) | 0.7782 | 0.8926 | **7.4** | 1599 |
| LIQE (KonIQ) | 0.6533 | 0.7767 | 8.8 | 2234 |
| QualiCLIP+ (CLIVE) | 0.6208 | 0.7487 | — | — |
| CLIP-IQA+ (KonIQ) | 0.7087 | 0.7188 | 24.1 | 1303 |
| MANIQA (KonIQ) | 0.4482 | 0.6368 | 6055.4 | 3.0 |
| TReS (KonIQ) | 0.5635 | 0.5787 | 1995.2 | 5.0 |

Where the numbers come from: SRCC is the macro line of the native-resolution
`*_val.log` / `*_heldout.log` in [`results/pyiqa/`](results/pyiqa/); GFLOPs and
img/s come from [`results/cost/`](results/cost/) (`flops` / `throughput`
columns). QualiCLIP+ has no cost row, hence the dash.

What the full sheet says:

- **The cheapest are the best.** The frontier runs PIQE → BRISQUE → NIQE →
  PaQ-2-PiQ → MetaIQA → UNIQUE → LIQE-mix, and every point on it is under 10
  GFLOPs. MANIQA (KonIQ) is the counterexample: 6055.4 GFLOPs against
  UNIQUE's 7.4, 35.3 GB of peak memory against 205 MB, 3 img/s against 1599 —
  and it scores 0.26 lower on the held-out sets.
- **Ranking depends on where you score it.** ARNIQA (KonIQ) is 4th in-domain
  and 23rd held out; MANIQA (KonIQ) is 43rd and 22nd. The in-domain macro
  rewards having trained on the target domain, and most of these checkpoints
  were trained on KonIQ.
- **TID2013 is the common floor.** Median SRCC across the 48 reportable rows
  is 0.452 there, against 0.736 on CLIVE. `datasets.md` predicted this: frozen
  features lose ground as a dataset gets harder.
- **Training on the wrong distribution is worse than not training.** NIQE
  reaches 0.4631 held out having never seen a MOS score; BRISQUE, which fitted
  an SVR to them on LIVE in 2012, reaches 0.4121. ARNIQA's seven checkpoints
  make the same point within one architecture — the ones trained on synthetic
  distortion (CSIQ 0.7050, TID 0.6957, KADID 0.6821) beat the ones trained on
  authentic capture (KonIQ 0.6354, CLIVE 0.4750, SPAQ 0.4440) on a held-out
  set that is mostly synthetic.

## Native vs 224px

Every file follows one naming rule:
`<metric>_{val,heldout}[_224].{csv,log}`. **No suffix is the reported
number** — the metric read images at native resolution, and the log's first
line says so (`... native resolution`). A `_224` suffix is a second run with
the longer side fixed at 224px, kept only as a controlled-input comparison.

The comparison matters because SPAQ ships 4032×3024 phone captures. On 300
SPAQ images, resizing to 224 moved DBCNN from 0.3050 to 0.8739, HyperIQA from
0.1680 to 0.7902, and MUSIQ from an out-of-memory error to a number. That is
not a correction: MUSIQ exists to read native multi-scale input, and NIQE
measures per-pixel statistics that downsampling erases, and both score lower
at 224. The `_224` files document the effect; they are not the reported
numbers.

One exception: ILNIQE has only `_224` files — native resolution never
finished — so for it the `_224` rows are the reportable ones, and the log
states the convention. The log's last line
(`per-image scores -> <file>.csv`) names the CSV it produced at run time;
that file now lives under `results/pyiqa/`.

## Reproducing it

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

Score one metric on both splits:

```bash
python run_pyiqa.py --metric musiq --data ./data/train.csv \
  --manifest ./data/split_manifest.csv --out results/pyiqa/musiq_val.csv
python run_pyiqa.py --metric musiq --data ./data/heldout.csv \
  --manifest ./data/heldout_manifest.csv --out results/pyiqa/musiq_heldout.csv

# a queue of them on one card (skips what is already on disk)
./run_native.sh 3 liqe unique arniqa maniqa topiq_nr

# cost, every metric in one sitting
python benchmark_pyiqa.py --metrics liqe unique arniqa --out results/cost/cost_pyiqa.csv
```

The manifest restricts scoring to the rows it marks `split=='val'` — 6,240 of
the 31,323 in `train.csv`. The held-out split is 8,010 rows. The splits are a
function of the CSV and `--seed 42`; use the committed `train.csv` rather
than regenerating it, because a different row order gives a different split
even with the same seed.

`qalign.py` and `qrealign.py` hard-code the machine paths at the top
(`ROOT`, `DATA_ROOT`) — edit them for your box. Their outputs land in
`results/qalign/` and `results/qrealign/`.

Four things cost a day between them:

- **`uv pip install torch` without `--reinstall` does nothing.** It prints
  `Checked 1 package in 4ms` and leaves the existing version alone.
- **uv only looks in the first index that has a package.** With an extra index
  attached it will pick a higher version from the wrong one;
  `--index-strategy unsafe-best-match` makes it look at all of them.
- **transformers 5.16 needs torch ≥ 2.5** to enable its PyTorch backend at all,
  and ≥ 2.6 to load `.bin` checkpoints.
- **`tmux new -d` does not read `~/.bashrc`**, so `HF_ENDPOINT` has to be set
  inside the script. Without it the first download of any new checkpoint hangs
  and the whole queue entry produces an empty log.

Feature caching (our own heads): the backbone never trains, so re-encoding the
same 31,323 images every epoch is work already done.

```bash
python cache_features.py --data ./data/train.csv --out ./features_clip-base.npy
python train_cached.py --data ./data/train.csv --features ./features_clip-base.npy \
  --head mlp --sampler by_dataset --epochs 5 --seed 42 --split reference \
  --manifest ./data/split_manifest.csv
```

Five epochs go from an hour and a half to seconds. The cache is 9.5 GB
(31,323 × 197 × 768, float16) and is gitignored. `benchmark.py` still times
the full image → encoder → head path, so caching changes the cost of
experimenting, not the cost of any design being reported.

## Conventions

- **Macro** = mean of CLIVE, CSIQ and TID2013 SRCC (or PLCC), each computed
  per dataset first; **worst** = the lowest of the three. AGIQA-3K never
  enters the macro and gets its own column.
- **Cost columns** come from `benchmark_pyiqa.py` at one fixed 224px input,
  one device: FLOPs (batch 1, `FlopCounterMode`), latency p50/p95 (batch 1,
  warm-up discarded), throughput (batch 16), peak VRAM. A metric whose
  preprocessing hides inside its forward gets n/a for FLOPs rather than a
  guess.
- The `throughput` printed at the end of a `*_val.log` is the scoring loop —
  batch 1, image loading included. It is not comparable with the cost sheet's
  batch-16 number; the table above uses the cost sheet.
- Lower-is-better metrics are flipped so every row reads higher = better.

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
- **DMM and FGResQ produced no output** — interface incompatible with
  single-image scoring as called here. The failed runs are
  `results/pyiqa/dmm_*.log` and `results/pyiqa/fgresq_*.log`.
- **ILNIQE ran at 224px only** — native resolution never finished (NIQE alone
  took five hours over 6,240 images on this shared machine and ILNIQE is
  heavier). Its `_224` files are complete and are its reported numbers.
- **BRISQUE and NIQE skip three AGIQA-3K images** (`DALLE2_normal_253`,
  `glide_normal_101`, `glide_normal_263`): they are flat enough that the pixel
  variance is zero and BRISQUE divides by it, while NIQE's SVD fails to
  converge.
- **CLIP-IQA+ at native resolution dropped 309 SPAQ images** — the shared
  card ran out of memory mid-split. The `_224` run is complete; treat the
  native row with that caveat.
