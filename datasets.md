# Datasets and measurement: what trains, what is held out, how cost is counted

The question is which architectures on a frozen encoder are worth their
inference cost. The encoder stays frozen; what varies between rows is the
architecture on top of it and the encoder underneath. The trainer, the loss,
the schedule and the split must be held fixed, because a comparison in which
two things moved is not a comparison. The split is reproducible rather than
written down: `split_by` is a function of the CSV and `--seed`, so the same
seed over the same table gives the same rows. Write those rows out once as a
manifest and compare against the manifest afterwards — a table of results
whose split cannot be reconstructed is not a result.

## Training

| dataset | rows | why it is here |
| --- | --- | --- |
| **KADID-10k** | 10,125 | Synthetic distortions under control: the same picture clean and damaged, 25 ways, 5 severities. |
| **KonIQ-10k** | 10,073 | The opposite — photographs degraded by whatever happened when they were taken, labelled by crowd workers. |
| **SPAQ** | 11,125 | Smartphone captures. Overlaps KonIQ in kind but not in source, which keeps the model from fitting one photographic pipeline. |

The three arrive on different scales, so each is min-maxed into [0, 1].
That is all `prepare_data.py` does about it. A per-dataset scale and shift on
the prediction would absorb the rest — subjects in different studies use their
scales differently, and a shared loss that ignores this spends capacity
reconciling laboratories instead of judging pictures — and it is not in
`train.py`: two learned parameters per dataset, applied to the prediction
before the loss, are a change worth making early. Correlations are invariant
to that alignment, which is why it costs nothing to report.

## Held out

Scored once, at the end. Selection happens on the in-domain validation split
alone.

| dataset | rows | how it is reported |
| --- | --- | --- |
| **CLIVE** | 1,162 | in the headline macro |
| **CSIQ** | 866 | in the macro |
| **TID2013** | 3,000 | in the macro; a weak column here is a likely outcome, not a bug — frozen features lose ground as a dataset gets harder, and that number prices the decision to keep the encoder frozen |
| **AGIQA-3K** | 2,982 | its own column, never in the macro — generated images fail in ways neither a camera nor a codec produces |

The macro is the mean over the three natural sets, and the worst of the
three is reported beside it: a mean hides a collapse on one set, and a
metric that fails on one kind of image cannot be deployed.

## Measuring cost

Four numbers that disagree on purpose.

| | what it measures | what it misses |
| --- | --- | --- |
| **FLOPs** | arithmetic | memory traffic, kernel launches, scheduling |
| **Batch-1 latency** | what a serving path waits for | how well the work batches |
| **Throughput at a fixed batch** | what an offline pipeline gets | the single-image case, where overheads dominate |
| **Peak memory, parameters** | what it costs to keep running | everything about speed |

Peak memory is reported on CUDA and on MPS. On CPU there is no allocator to
ask, so the column is empty rather than filled with the resident size of the
whole interpreter.

One device, one precision, one input convention, for every row — a cost
collected across machines is not a cost. Throw away the first measurement:
the first design measured pays for backend initialisation, and on this
machine the same head reported 21 ms timed first and 6.6 ms timed second.
Report FLOPs and latency together; when they disagree, the disagreement is
the finding.

## The frontier

A design is on the frontier if nothing else tested is both more accurate and
cheaper — "among the pairs we ran", and worth writing that way, because the
frontier of a grid is not the frontier of the design space. Name at least
two operating points and say what each costs; a single recommendation hides
the shape of the trade-off, which is the whole result.

## Splits

Splits go by reference. In KADID a hundred and twenty-five rows are one
photograph; splitting them across the boundary lets the model score by
recognising the picture — up to 0.44 SRCC on frozen features, larger than
the differences between the architectures being compared. For photographs,
splitting by image is fine.

The held-out share is drawn from each of the three separately. A reference is
one photograph in KonIQ and a hundred and twenty-five rows in KADID, so a
single draw over the pool is decided by whichever release has the most
references, and a dataset can miss the held-out side altogether — measured on
this exact table, one of the three came out with nothing held out at all. For
the same reason `reference` is written with its dataset in front of it.
