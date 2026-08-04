# Datasets

The raw benchmarks are not redistributed here; both are public on the UCI
Machine Learning Repository (the Dataset B steady-state *features* derived from
the raw recordings are committed under `data/DataSetB/` — see the license note
in the README). This page documents where to put the files and how the raw
recordings become the inputs reported in the paper.

## Dataset A — Gas Sensor Array Drift

* UCI page: <https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset>
* 13,910 samples, 6 gases, 10 batches spanning 36 months
* 128 features per sample: 8 features from each of 16 MOX sensors

Download the archive, unpack it, and place the ten batch files here:

```
data/DataSetA/
    batch1.dat   batch2.dat   ...   batch10.dat
```

Each line is `<label>;<concentration> 1:<v> 2:<v> ... 128:<v>`, with labels
numbered 1-6. `unifiedgas.data.load_dataset_a_batch` parses this format and
reshapes the 128 features into a 16 x 8 map so the 2-D backbone sees sensors
along one axis and features along the other. Features are z-scored per file,
which is the per-batch normalization protocol used throughout the paper.

## Dataset B — Twin Gas Sensor Arrays

* UCI page: <https://archive.ics.uci.edu/dataset/361/twin+gas+sensor+arrays>
* 640 recordings, 4 gases, 5 replicate 8-MOX boards
* Board 1-3 contribute 160 samples each, Boards 4-5 contribute 80 each

Download and unpack the raw recordings into a single directory. The UCI zip
nests them in a `data1/` subdirectory, so move them up (or point `--data_dir`
at `data/DataSetB-raw/data1`):

```
data/DataSetB-raw/
    B1_GCO_F010_R1.txt
    B1_GCO_F010_R2.txt
    ...
    B5_GMe_F100_R2.txt
```

The file name encodes `B<board>_<gas>_F<flow rate>_R<repetition>.txt`, with
gases `GCO` (CO), `GEa` (ethanol), `GEy` (ethylene), and `GMe` (methane).

Each file holds a timestamp column followed by the resistance (kOhm) of the eight
sensors, sampled at 100 Hz. A recording is nominally 60,000 samples long
(600 s at 100 Hz). A few runs terminate early: across the 640 files the
lengths range from 19,289 to 60,001 samples with a median of 59,977, and 19
files fall below 55,000 samples.

### Steady-state representation (8-D)

The steady-state feature of sensor *j* is the arithmetic mean of its response
over the **last 5,000 samples** of the recording, i.e. the final 50 s of the
exposure once the response has plateaued:

    x_j = (1/W) * sum_{t=L-W}^{L-1} s_j(t),    W = 5000

where *L* is that recording's own length. The descriptor is the average
plateau level, not the maximum or minimum of the transient. The eight
per-sensor values form the 8-dimensional vector.

Build the per-board CSV files with:

```bash
python scripts/preprocess_datasetB.py --data_dir data/DataSetB-raw --out_dir data/DataSetB
```

which writes `data/DataSetB/batch1.csv ... batch5.csv`, each row being
`f0,...,f7,label` with labels 0-3.

### Raw time-series representation (256 x 8)

For the time-series protocol, each recording is downsampled to 256 timesteps by
indexing 256 uniformly spaced positions **over that recording's own length**, so
every sample yields exactly 256 timesteps regardless of early termination:

```bash
python scripts/preprocess_datasetB.py --data_dir data/DataSetB-raw \
    --timeseries --T 256 --out_dir data/cache
```

This writes `data/cache/datasetB_ts_T256.npz` with `board{i}_X` of shape
`(N_i, 256, 8)` and `board{i}_y` of shape `(N_i,)`.

## Verifying the setup

Once `data/DataSetB-raw/` is populated, this reproduces the cross-board diagnostic
end to end and prints the recording-length distribution quoted above:

```bash
python scripts/reproduce_reviewer_check.py --data_dir data/DataSetB-raw \
    --mode all --classifiers svm,rf,mlp --report_lengths
```

Expected: within-board 5-fold cross-validation reaches 90-99%, while
cross-board transfer averages 27-29% over the 20 source-target pairs, i.e.
essentially the four-class chance level of 25%. Individual pairs scatter widely
around that mean (9-50%), which is what chance-level guessing on 80-160 samples
looks like; no normalization protocol lifts the average.
