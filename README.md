# UnifiedGas

**Official implementation of the paper "UnifiedGas: End-to-End Unsupervised
Domain Adaptation for Drift-Robust Gas Classification"** (Chenyu Wang, Yi Liu,
Yanshu Gu, Baoqing Li, Min Tu), under review at the *IEEE Sensors Journal*.

The repository releases the method itself and the checks the reviewers asked
for: the UnifiedGas model, its single-stage training loop, the deep UDA
baselines re-implemented on the same backbone, the Dataset-B preprocessing,
trained checkpoints, and the commands that reproduce the Dataset-A transfers
and the cross-board diagnostics. For the raw time-series protocol of Table VI,
the release includes the preprocessing (steady-state extraction and 256-step
downsampling); the separate Transformer backbone used by that protocol is not
part of this release.

**Questions about the method, the experiments, or the code are welcome — please
write to [wangchenyu0628.wcy@gmail.com](mailto:wangchenyu0628.wcy@gmail.com).**

UnifiedGas compensates sensor drift in electronic noses with a single-stage
unsupervised domain adaptation objective. Feature extraction, hierarchical
multi-kernel MMD alignment, a complementary CORAL covariance term, dual-level
center regularization, and fused multi-head classification are optimized
together by one optimizer under one alignment-weight schedule — there is no
separate pretraining stage and no manually specified phase transition. The
model has 0.627 M parameters and classifies a sample in roughly 0.2-0.6 ms in
batched CPU inference (see `checkpoints/README.md` for measured numbers).

---

## 1. Installation

```bash
git clone https://github.com/ChenyuWang0628/UnifiedGas.git
cd UnifiedGas
bash setup_env.sh                 # creates a conda env named "unifiedgas"
conda activate unifiedgas
```

`setup_env.sh` creates a Python 3.10 conda environment, installs
`requirements.txt`, and prints the resolved versions so you can confirm the
install. Pass a different name if you prefer (`bash setup_env.sh myenv`). If
conda is unavailable the script falls back to a `.venv` in the repository root.

To install by hand instead:

```bash
conda create -n unifiedgas python=3.10 -y
conda activate unifiedgas
pip install -r requirements.txt
```

Dependencies are PyTorch, NumPy, SciPy, scikit-learn, pandas and Matplotlib.
Everything runs on CPU; a GPU only makes the 400-epoch runs faster.

## 2. Data

**The Dataset B steady-state features are included in this repository**
(`data/DataSetB/`, about 62 kB), so the cross-board verification of Section 6 and
Dataset B training run immediately after installation, with no download. These
five CSV files are the 640 8-D feature vectors used in the paper — 160 samples
for each of Boards 1-3 and 80 for each of Boards 4-5 — derived from the UCI
recordings by `scripts/preprocess_datasetB.py`. Each row is eight sensor values
followed by the gas label (0 = CO, 1 = ethanol, 2 = ethylene, 3 = methane).

The underlying benchmarks are public and are **not** redistributed here; the raw
recordings in particular are 2.6 GB. Download them from the UCI Machine Learning
Repository:

* Dataset A — [Gas Sensor Array Drift](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset)
* Dataset B — [Twin Gas Sensor Arrays](https://archive.ics.uci.edu/dataset/361/twin+gas+sensor+arrays)

Place them so that the default layout applies (every script also accepts an
explicit `--data_dir`, so the data may live anywhere):

```
data/DataSetA/       batch1.dat ... batch10.dat   Dataset A batch files
data/DataSetB-raw/   B*_G*_F*_R*.txt              Dataset B raw recordings
data/DataSetB/       batch1.csv ... batch5.csv    Dataset B steady-state features (included)
```

Note that the UCI zip archives nest the files in a subdirectory (`Dataset/`
for Dataset A, `data1/` for Dataset B), so move the files up after unpacking —
e.g. `mv data/DataSetB-raw/data1/*.txt data/DataSetB-raw/` — or pass the
subdirectory itself as `--data_dir`.

To rebuild the included features from the raw recordings — which also verifies
that the steady-state definition in the paper is what the code implements:

```bash
python scripts/preprocess_datasetB.py --data_dir data/DataSetB-raw --out_dir data/DataSetB
```

[docs/DATA.md](docs/DATA.md) documents the exact steady-state definition, the
time-series downsampling rule, and the recording-length statistics.

## 3. Inference

Two checkpoints are included. The Dataset B one runs immediately after
installation, since its features ship with the repository:

```bash
python scripts/predict.py \
    --checkpoint checkpoints/unifiedgas_B_board1to2.pt \
    --input data/DataSetB/batch2.csv
```

Expected output: 33.75% on Board 2 — near the four-class chance level, which is
the cross-board collapse this dataset is included to document (Section 6).

Once Dataset A is downloaded (Section 2):

```bash
python scripts/predict.py \
    --checkpoint checkpoints/unifiedgas_A_batch1to6.pt \
    --input data/DataSetA/batch6.dat
```

Expected output: 95.48% on Batch 6, matching the accuracy recorded inside the
checkpoint. Add `--benchmark` to measure latency, or `--out predictions.csv` to
write per-sample predictions and class probabilities.

The input may be a benchmark file (`.dat` for Dataset A, `.csv` for Dataset B)
or a raw feature matrix (`.npy`). See [checkpoints/README.md](checkpoints/README.md)
for what each checkpoint contains.

## 4. Training

One transfer task, with the settings used in the paper:

```bash
# Dataset A, Setting 1 (Batch 1 is the fixed source): Batch 1 -> Batch 6
python scripts/train.py --dataset A --source 1 --target 6 \
    --epochs 400 --seeds 42,123,2024 \
    --save_checkpoint checkpoints/my_run.pt

# Dataset A, Setting 2 (sequential adaptation): Batch 5 -> Batch 6
python scripts/train.py --dataset A --source 5 --target 6 --epochs 400

# Dataset B, steady-state features: Board 1 -> Board 2 (five seeds, as in the paper)
python scripts/train.py --dataset B --source 1 --target 2 --epochs 400 \
    --seeds 42,123,2024,7,99
```

A single seed of one Dataset A task takes roughly 30-60 minutes on a laptop CPU
and a few minutes on a GPU. Use `--epochs 5` for a quick smoke test.

Useful options: `--lambda_coral` (CORAL weight, default 1.0), `--lambda_center`
(center regularizer, default 0.1), `--warmup_frac` (fraction of epochs before
alignment switches on, default 0.25), `--norm` (normalization protocol for
Dataset B), `--device`.

Target labels never enter a gradient or a hyperparameter search. They are
evaluated at the end of every epoch solely to record which epoch to report:
following the retrospective oracle-selection protocol applied uniformly to
every deep method in the paper,
the reported accuracy is the target accuracy at its best epoch, and the saved
checkpoint holds the weights from that epoch. `train.py` also prints the
final-epoch accuracy as the conservative alternative.

## 5. Baselines

The eight deep UDA baselines reported in Tables III-V are re-implemented on the
*same* backbone, classifier heads, logit fusion, classification loss, optimizer,
and schedule as UnifiedGas, so a comparison isolates the adaptation objective
rather than network capacity:

| Baseline | Adaptation objective | Reference |
|---|---|---|
| DANN | adversarial domain discriminator | Ganin et al., JMLR 2016 |
| Deep CORAL | second-order covariance alignment | Sun & Saenko, ECCVW 2016 |
| JAN | joint MMD over multiple layers | Long et al., ICML 2017 |
| CDAN | conditional adversarial alignment | Long et al., NeurIPS 2018 |
| MDD | margin disparity discrepancy | Zhang et al., ICML 2019 |
| MCC | minimum class confusion | Jin et al., ECCV 2020 |
| DSAN | local MMD over class subdomains | Zhu et al., TNNLS 2021 |
| SDAT | sharpness-aware adversarial training | Rangwani et al., ICML 2022 |

```bash
python scripts/train_baseline.py --list      # show available baselines

python scripts/train_baseline.py --method dsan --dataset A \
    --source 1 --target 6 --epochs 400 --seeds 42,123,2024
```

## 6. Verifying the Dataset B cross-board result

Using the features included in this repository, no download required:

```bash
python scripts/reproduce_reviewer_check.py --from_features data/DataSetB \
    --mode all --classifiers svm,rf,mlp
```

Or end to end from the raw recordings — steady-state feature extraction,
normalization, training a classifier on one board, testing on another — with no
reliance on any precomputed artifact:

```bash
python scripts/reproduce_reviewer_check.py --data_dir data/DataSetB-raw \
    --mode all --classifiers svm,rf,mlp --report_lengths
```

Both routes give identical SVM and random-forest numbers (the MLP can differ by
one sample, since the committed features are serialized in float32); the second
additionally verifies the feature extraction itself.

It prints the recording-length distribution, per-board sample counts, the
Board 1 → Board 2 transfer under three normalization protocols
(none / source-only / per-board), the within-board cross-validation accuracy,
and the full 5×5 cross-board matrix for each classifier.

Expected outcome — within-board classification succeeds while cross-board
transfer collapses to near the four-class chance level of 25%:

| Within-board 5-fold CV (SVM-rbf) | Board 1 | Board 2 | Board 3 | Board 4 | Board 5 |
|---|---|---|---|---|---|
| Accuracy | 90.6% | 97.5% | 98.1% | 98.8% | 95.0% |

Cross-board transfer with the same classifier averages **27.9%** over the 20
off-diagonal source-target pairs; random forests give 27.4% and MLPs 29.3%. The
collapse occurs under every normalization protocol, so it reflects a genuine
inter-board distribution shift in the 8-D steady-state space rather than a
parsing, labeling, or normalization error. With the raw transient time series
instead, the same deep UDA methods exceed 95% under the fixed-source setting.

The PCA score plots that visualize the same shift are produced by:

```bash
python scripts/plot_datasetB_pca.py --data_dir data/DataSetB-raw --out_dir figures
```

## 7. Reproduction notes

The paper averages over seeds `{42, 123, 2024}` on Dataset A and
`{42, 123, 2024, 7, 99}` on the Dataset B steady-state protocol; loop the
commands of Sections 4 and 5 over the source/target pairs of the setting you
want. Runs are seeded and use deterministic kernels where available, but exact
values may still shift slightly across PyTorch versions and hardware.

Two items in the paper are outside this release: the Transformer backbone and
conditional-alignment components used by the raw time-series block of Table VI
(its preprocessing is included), and the Holm-corrected Wilcoxon analysis of
Table V, which is a post-hoc test over per-task accuracies rather than a
training procedure.

## 8. Citation

```bibtex
@article{wang2026unifiedgas,
  author  = {Wang, Chenyu and Liu, Yi and Gu, Yanshu and Li, Baoqing and Tu, Min},
  title   = {{UnifiedGas}: End-to-End Unsupervised Domain Adaptation for
             Drift-Robust Gas Classification},
  journal = {IEEE Sensors Journal},
  year    = {2026},
  note    = {Under review}
}
```

## 9. Contact and license

For questions about the paper or this code, or to report a problem with
reproduction, contact **Chenyu Wang** at
[wangchenyu0628.wcy@gmail.com](mailto:wangchenyu0628.wcy@gmail.com). Issues and
pull requests on this repository are equally welcome.

The **code** is released under the MIT License; see [LICENSE](LICENSE). The
feature files in `data/DataSetB/` are derived from the UCI *Twin Gas Sensor
Arrays* dataset by Jordi Fonollosa (DOI
[10.24432/C5MW3K](https://doi.org/10.24432/C5MW3K)), which is distributed under
a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license; the
derivation (per-sensor steady-state means) is described in Section 2 and
implemented in `scripts/preprocess_datasetB.py`. The MIT license does not cover
these derived data.
