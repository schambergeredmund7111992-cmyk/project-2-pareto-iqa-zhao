# Pareto-optimal IQA — quick start

An accurate quality metric is a large fine-tuned network; a cheap one loses
accuracy. This project asks which architectures on top of a frozen encoder
are worth what they cost to run.

The full statement — research directions, how it is evaluated, what to hand
in — is at
[dreminm.github.io/iqa-summer-school/project-2.html](https://dreminm.github.io/iqa-summer-school/project-2.html).
This repository is where you start from.

Five scripts and a note to get you both numbers today:

```
download_data.py   fetch a dataset and unpack it
prepare_data.py    its labels -> one CSV, same columns for every dataset
dataset.py         a torch Dataset over that CSV, with splitting and sampling
train.py           frozen CLIP + an MLP, trained to predict quality
benchmark.py       what that design costs: FLOPs, latency, throughput, memory
datasets.md        what trains, what is held out, how cost is counted
```

## Run it

```
uv venv --python 3.12
source .venv/bin/activate        # .venv\Scripts\activate on Windows
uv pip install -e .

python download_data.py --list
python download_data.py kadid10k --data-root ~/iqa-data     # 2.9 GB, start here
python prepare_data.py ~/iqa-data/kadid10k
python train.py --data ~/iqa-data/kadid10k/labels.csv --epochs 5
python benchmark.py --backbone clip-base
```

Use `--limit 2000` while you are still wiring things up — it samples that many
training images at random, and leaves the held-out split whole.

Downloads run in parallel byte ranges, because the mirror throttles a single
sustained connection to a crawl. Pass `--connections 1` if a proxy dislikes
range requests.

## The two axes

`train.py` gives SRCC and PLCC on a held-out split; `benchmark.py` gives
FLOPs, batch-1 latency, throughput and parameter counts for the same design.
One point on the plane is one encoder-and-head pair, and the frontier is
whichever pairs nothing else beats on both axes.

Measure every row on one device at one precision, or the cost column is not
comparable. And discard the first measurement: the first design measured
pays for backend initialisation and comes out several times slower —
`benchmark.py` throws away a warm-up pass for exactly this reason.

## What prepare_data does

Every release ships its labels differently, so this reads whichever format
it finds and writes one table:

| column | |
| --- | --- |
| `path` | the image |
| `original_subjective_score` | the score as the release published it |
| `scaled_subjective_score` | the same, min-maxed to [0, 1], higher = better |
| `dataset` · `reference` | which set it came from, and of which pristine image |
| `distortion` · `level` | the type and severity the release recorded |
| `group` | that type folded into one of eight distortion groups |

Scores arrive on 1–5, 0–9 and 0–100 scales and one release counts backwards,
so training on several sets at once needs the scaled column. The three
training sets of this project, in one table:

```
python prepare_data.py ~/iqa-data/{kadid10k,koniq10k,spaq} --out ~/iqa-data/train.csv
python train.py --data ~/iqa-data/train.csv --sampler by_dataset --epochs 5
```

`--sampler by_dataset` keeps the largest set from deciding the batch, and
`train.py` prints SRCC and PLCC per dataset with their macro and the worst of
the three, which is what this project reports.

## Splitting

```python
from dataset import IQADataset, split_by, make_sampler

data = IQADataset("~/iqa-data/kadid10k/labels.csv", image_size=224)
train, val = split_by(data, "reference")        # or "random"
sampler = make_sampler(train, "balanced")       # or "random", "by_level", "by_dataset"
```

`split_by` keeps a pristine reference whole on one side, and takes its share
from every dataset separately. Both defaults matter. In KADID a hundred and
twenty-five rows are one photograph seen through twenty-five distortions, so
splitting them apart lets the model score the held-out ones by recognising
the picture — on frozen features that is worth up to 0.44 SRCC, larger than
the differences between the architectures you are comparing. And a reference
means different things in different releases, one photograph here and a
hundred and twenty-five rows there, so drawing the held-out share from the
pool would let one release decide the split. Use `"random"` for photographs.

## Where to go next

`train.py` and `benchmark.py` are short and meant to be edited.
`--backbone clip-large`, `siglip2-base` or `siglip2-large`, `QualityMLP` for a
different head — the head is the design space of this project — and `embed()`
if you want patch tokens instead of the pooled embedding. The backbone is
frozen, so caching features once changes the cost of experimenting, not the
cost of the design you report.

Which datasets train, which are held out and how cost is measured:
[datasets.md](datasets.md).
