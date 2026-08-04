# Checkpoints

Trained weights for representative transfer tasks. Each file stores the model
`state_dict` together with the `TrainConfig` needed to rebuild the network, so
`scripts/predict.py` can load it without any additional arguments.

| File | Dataset | Transfer | Classes | Parameters | Target accuracy |
|---|---|---|---|---|---|
| `unifiedgas_A_batch1to6.pt` | A | Batch 1 -> Batch 6 | 6 | 0.627 M | 95.48% |
| `unifiedgas_B_board1to2.pt` | B | Board 1 -> Board 2 | 4 | 0.239 M | 33.75% |

Accuracies are for a single seed (42) at the best target epoch, under the same
protocol as the paper. The tables in the paper average over three seeds, so
individual checkpoints will differ slightly from the reported means.

Each file stores the weights from the reported epoch, so re-running
`scripts/predict.py` on the corresponding target reproduces the accuracy in the
table above exactly.

Measured inference cost for the Dataset A checkpoint on a CPU, over repeated
runs on an otherwise loaded laptop: roughly 0.6-1.7 ms for a single sample and
0.17-0.55 ms per sample in batched mode (batch of 2,300). Absolute latency
depends on the hardware and on how busy the machine is, so reproduce it with
`--benchmark` on your own system rather than treating these as fixed numbers.

## Usage

```bash
python scripts/predict.py --checkpoint checkpoints/unifiedgas_A_batch1to6.pt \
    --input data/DataSetA/batch6.dat --benchmark
```

## A note on the Dataset B checkpoint

The Dataset B checkpoint is included deliberately, even though its accuracy is
low, because it makes the paper's central Dataset B claim inspectable rather
than merely asserted. Running it on Board 2 gives 33.75% against a four-class
chance level of 25%, and the predictions are heavily skewed across classes
(60 / 18 / 82 / 0 over CO / ethanol / ethylene / methane) rather than tracking
the balanced 40-per-class ground truth.

This is not an artifact of the checkpoint. As `scripts/reproduce_reviewer_check.py`
demonstrates from the raw recordings, classical classifiers fail the same way on
this representation — often degenerating to a single predicted class — while
reaching 90-99% within a single board. The steady-state descriptor encodes board
identity more strongly than gas identity: in the PCA score plots in `figures/`,
between-board scatter accounts for 56.2% of the variance on the leading plane
versus 2.1% for between-gas scatter.

With the raw transient time series instead of the steady-state summary, every
deep UDA method exceeds 95% on this dataset — using a separate Transformer
backbone together with the conditional-alignment components, which are outside
the scope of this release (see the main README). The limitation is in the input
representation, not the adaptation objective.
