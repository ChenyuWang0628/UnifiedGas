#!/usr/bin/env python3
"""Self-contained cross-board verification on Dataset B (UCI Twin Gas Sensor Arrays).

This script reproduces, end to end and in a single file, the cross-board
transfer result reported for Dataset B.  It covers the complete chain:

    raw .txt recordings
      -> steady-state feature extraction (8-D per sample)
      -> feature normalization (three protocols, incl. "no normalization")
      -> train an SVM on the source board
      -> test on the target board

Steady-state feature definition
-------------------------------
Each raw recording is a matrix of shape (L, 9): column 0 is the timestamp and
columns 1-8 are the eight MOX sensor resistances (kOhm).  Sampling is at 100 Hz for a
nominal 600 s exposure, i.e. L is nominally 60,000 (a small number of runs stop
early; see --report_lengths).  The steady-state feature of one recording is the
**per-sensor arithmetic mean over the last 5,000 samples** (the final 50 s of
the exposure, when the response has plateaued).  This yields one 8-dimensional
vector per recording -- it is neither the maximum nor the minimum of the
transient, but the average level of the plateau.  Formally, for sensor j:

    x_j = (1 / W) * sum_{t = L-W}^{L-1} s_j(t),     W = 5000.

Default experiment
------------------
Board 1 (Unit 1) is the training set and Board 2 (Unit 2) is the test set, as
requested, with an RBF-kernel SVM.  Other board pairs, the full 5x5 matrix,
within-board cross-validation, and random-forest / MLP classifiers are also
available via the command-line flags.

Usage
-----
    # The requested Unit 1 -> Unit 2 SVM check, starting from the raw recordings
    python reproduce_reviewer_check.py --data_dir /path/to/DataSetB-raw

    # Full 5x5 source->target matrix plus within-board 5-fold CV
    python reproduce_reviewer_check.py --data_dir /path/to/DataSetB-raw --mode all

    # Same check without downloading the 2.6 GB of raw data, using the
    # steady-state features committed to this repository.  Both routes give
    # identical numbers; the raw route additionally verifies the extraction.
    python reproduce_reviewer_check.py --from_features data/DataSetB

Expected outcome: within-board cross-validation is high (>90%), whereas
cross-board transfer collapses to near the four-class chance level (~25%),
under every normalization protocol.  The gap is therefore a property of the
8-D steady-state representation, not an artifact of parsing or normalization.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Four target gases; the label index follows the order used throughout the paper.
GAS_TO_LABEL = {"GCO": 0, "GEa": 1, "GEy": 2, "GMe": 3}
LABEL_TO_GAS = {v: k for k, v in GAS_TO_LABEL.items()}
GAS_FULL_NAME = {
    "GCO": "CO",
    "GEa": "Ethanol",
    "GEy": "Ethylene",
    "GMe": "Methane",
}

# File names look like  B1_GCO_F010_R1.txt  =  board / gas / flow rate / repetition.
FNAME_RE = re.compile(r"^B([1-5])_(GCO|GEa|GEy|GMe)_F(\d{3})_R(\d+)\.txt$")

STEADY_STATE_WINDOW = 5000  # last 5,000 samples = final 50 s at 100 Hz
N_SENSORS = 8
N_CLASSES = 4


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_steady_state(path: str | Path, window: int = STEADY_STATE_WINDOW) -> np.ndarray:
    """Return the 8-D steady-state feature of one raw recording.

    Column 0 of the file is the timestamp and is discarded; columns 1-8 are the
    eight sensor channels.  The feature is the per-sensor mean over the last
    ``window`` samples.
    """
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < N_SENSORS + 1:
        raise ValueError(f"unexpected shape {data.shape} in {path}")
    sensors = data[:, 1 : N_SENSORS + 1]  # (L, 8), drop the time column
    if sensors.shape[0] < window:
        raise ValueError(
            f"{path} has only {sensors.shape[0]} samples, fewer than the "
            f"{window}-sample steady-state window"
        )
    return sensors[-window:].mean(axis=0).astype(np.float64)


def load_boards(data_dir: str | Path, window: int = STEADY_STATE_WINDOW, verbose: bool = True):
    """Extract steady-state features for all five boards.

    Returns ``{board_index: (X, y)}`` with ``X`` of shape (N, 8) and ``y`` of
    shape (N,).
    """
    data_dir = Path(data_dir)
    files = sorted(glob.glob(str(data_dir / "*.txt")))
    if not files:
        raise FileNotFoundError(
            f"no .txt recordings found in {data_dir}. Download the UCI Twin Gas "
            f"Sensor Arrays dataset first (see docs/DATA.md), or pass "
            f"--from_features data/DataSetB to read the committed feature files "
            f"instead of extracting them from the raw recordings."
        )

    boards: dict[int, dict[str, list]] = {b: {"X": [], "y": []} for b in range(1, 6)}
    skipped = []
    for fp in files:
        m = FNAME_RE.match(os.path.basename(fp))
        if m is None:
            skipped.append(os.path.basename(fp))
            continue
        board = int(m.group(1))
        label = GAS_TO_LABEL[m.group(2)]
        boards[board]["X"].append(extract_steady_state(fp, window))
        boards[board]["y"].append(label)

    out = {}
    for b in range(1, 6):
        X = np.asarray(boards[b]["X"], dtype=np.float64)
        y = np.asarray(boards[b]["y"], dtype=np.int64)
        out[b] = (X, y)
        if verbose:
            counts = np.bincount(y, minlength=N_CLASSES)
            per_class = ", ".join(
                f"{LABEL_TO_GAS[c]}={counts[c]}" for c in range(N_CLASSES)
            )
            print(f"  Board {b}: X={X.shape}  y={y.shape}  ({per_class})")
    if skipped and verbose:
        print(f"  [warn] {len(skipped)} file(s) did not match the expected name pattern")
    return out


def load_boards_from_features(feature_dir: str | Path, verbose: bool = True):
    """Load the committed steady-state CSVs instead of extracting from raw data.

    This shortcut exists so the cross-board numbers can be checked without
    downloading the 2.6 GB of raw recordings.  It reads what
    ``preprocess_datasetB.py`` wrote; to verify the extraction step itself, run
    without ``--from_features`` so the features are recomputed from the raw
    ``.txt`` files.
    """
    feature_dir = Path(feature_dir)
    out = {}
    for b in range(1, 6):
        path = feature_dir / f"batch{b}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
        X, y = arr[:, :N_SENSORS], arr[:, N_SENSORS].astype(np.int64)
        out[b] = (X, y)
        if verbose:
            counts = np.bincount(y, minlength=N_CLASSES)
            per_class = ", ".join(
                f"{LABEL_TO_GAS[c]}={counts[c]}" for c in range(N_CLASSES)
            )
            print(f"  Board {b}: X={X.shape}  y={y.shape}  ({per_class})")
    return out


# --------------------------------------------------------------------------- #
# Normalization protocols
# --------------------------------------------------------------------------- #
def normalize(Xs: np.ndarray, Xt: np.ndarray, mode: str):
    """Apply one of the feature-normalization protocols to a source/target pair.

    ``none``        no normalization at all (raw resistance means).
    ``source_only`` fit a z-score scaler on the source, apply it to both
                    domains (the textbook supervised-pipeline choice).
    ``per_board``   z-score each board with its own statistics; this uses only
                    unlabeled target statistics and is the protocol used for the
                    trained models in the paper.
    """
    if mode == "none":
        return Xs, Xt
    if mode == "source_only":
        sc = StandardScaler().fit(Xs)
        return sc.transform(Xs), sc.transform(Xt)
    if mode == "per_board":
        return (
            StandardScaler().fit_transform(Xs),
            StandardScaler().fit_transform(Xt),
        )
    raise ValueError(f"unknown normalization mode: {mode}")


def make_classifier(name: str, seed: int):
    """Classifiers and hyperparameters used throughout the paper's Dataset-B checks."""
    if name == "svm":
        return SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
    raise ValueError(f"unknown classifier: {name}")


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
def cross_board(boards, src: int, tgt: int, clf_name: str, norm: str, seed: int):
    """Train on the source board, test on the target board."""
    Xs, ys = boards[src]
    Xt, yt = boards[tgt]
    Xs_n, Xt_n = normalize(Xs, Xt, norm)
    clf = make_classifier(clf_name, seed).fit(Xs_n, ys)
    pred = clf.predict(Xt_n)
    acc = float((pred == yt).mean())
    # A collapse onto a single class is the signature failure mode here, so we
    # report the prediction histogram alongside the accuracy.
    hist = np.bincount(pred, minlength=N_CLASSES).tolist()
    return acc, hist


def within_board_cv(boards, board: int, clf_name: str, norm: str, seed: int, folds: int = 5):
    """Supervised k-fold cross-validation inside a single board.

    The scaler is fitted on each training fold only, so no test-fold statistics
    leak into training.
    """
    X, y = boards[board]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, te_idx in cv.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        if norm != "none":
            sc = StandardScaler().fit(X_tr)
            X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
        clf = make_classifier(clf_name, seed).fit(X_tr, y[tr_idx])
        scores.append(float((clf.predict(X_te) == y[te_idx]).mean()))
    return float(np.mean(scores)), float(np.std(scores))


def report_lengths(data_dir: str | Path):
    """Print the recording-length distribution of the raw files."""
    files = sorted(glob.glob(str(Path(data_dir) / "*.txt")))
    if not files:
        raise FileNotFoundError(
            f"no .txt recordings found in {data_dir}. Download the UCI Twin Gas "
            f"Sensor Arrays dataset first and point --data_dir at the directory "
            f"holding the raw B*_G*_F*_R*.txt files (see docs/DATA.md)."
        )
    lengths = []
    for fp in files:
        with open(fp, "rb") as fh:
            lengths.append(sum(1 for _ in fh))
    lengths = np.asarray(lengths)
    print(f"\nRaw recording lengths over {len(lengths)} files:")
    print("  nominal (600 s x 100 Hz)     : 60000")
    print(f"  min / median / max           : {lengths.min()} / {int(np.median(lengths))} / {lengths.max()}")
    print(f"  >= 59,000 samples            : {(lengths >= 59000).sum()} files")
    print(f"  <  55,000 samples (truncated): {(lengths < 55000).sum()} files")
    print(f"  all >= steady-state window ({STEADY_STATE_WINDOW}): {bool((lengths >= STEADY_STATE_WINDOW).all())}")
    return lengths


# --------------------------------------------------------------------------- #
def main():
    # MLPClassifier at max_iter=500 does not always converge; that is expected
    # for this diagnostic and does not change the reported numbers, so the
    # per-fit ConvergenceWarnings are suppressed and noted here once instead.
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    ap = argparse.ArgumentParser(
        description="Cross-board verification on Dataset B (Twin Gas Sensor Arrays)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_dir", default=None,
                    help="directory holding the raw B*_G*_F*_R*.txt recordings")
    ap.add_argument("--from_features", default=None, metavar="DIR",
                    help="skip feature extraction and read the committed "
                         "batch1..5.csv in DIR (e.g. data/DataSetB); use this to "
                         "check the transfer numbers without the 2.6 GB raw data")
    ap.add_argument("--mode", default="pair", choices=["pair", "all"],
                    help="'pair' runs only the requested source->target check; "
                         "'all' adds the 5x5 matrix and within-board CV")
    ap.add_argument("--source", type=int, default=1, help="source board (Unit)")
    ap.add_argument("--target", type=int, default=2, help="target board (Unit)")
    ap.add_argument("--classifiers", default="svm",
                    help="comma-separated subset of svm,rf,mlp")
    ap.add_argument("--norms", default="none,source_only,per_board",
                    help="comma-separated subset of none,source_only,per_board")
    ap.add_argument("--window", type=int, default=STEADY_STATE_WINDOW,
                    help="steady-state window length in samples (100 Hz)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_lengths", action="store_true",
                    help="print the raw recording-length distribution")
    ap.add_argument("--out", default=None, help="optional path for a JSON dump of the results")
    args = ap.parse_args()

    if not args.data_dir and not args.from_features:
        ap.error("pass --data_dir <raw recordings> or --from_features data/DataSetB")

    clf_names = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    norms = [n.strip() for n in args.norms.split(",") if n.strip()]

    print("=" * 78)
    print("Dataset B cross-board verification (UCI Twin Gas Sensor Arrays)")
    print("=" * 78)
    print(f"Steady-state feature: per-sensor mean of the last {args.window} samples")
    print(f"                      (final {args.window / 100:.0f} s of each 600 s exposure at 100 Hz)")
    print(f"Feature dimension   : {N_SENSORS} (one value per MOX sensor)")
    print(f"Classes             : {N_CLASSES} "
          f"({', '.join(GAS_FULL_NAME[LABEL_TO_GAS[i]] for i in range(N_CLASSES))})")
    print()

    if args.report_lengths:
        if not args.data_dir:
            ap.error("--report_lengths needs --data_dir, since it measures the raw recordings")
        report_lengths(args.data_dir)
        print()

    if args.from_features:
        print(f"Reading committed steady-state features from {args.from_features}:")
        boards = load_boards_from_features(args.from_features)
    else:
        print("Extracting steady-state features:")
        boards = load_boards(args.data_dir, window=args.window)
    total = sum(len(y) for _, y in boards.values())
    print(f"  total: {total} samples\n")

    results: dict = {
        "steady_state_window": args.window,
        "sampling_rate_hz": 100,
        "n_samples_per_board": {b: int(len(boards[b][1])) for b in boards},
        "cross_board": {},
    }

    # ---- the requested check: source board -> target board -------------- #
    print("-" * 78)
    print(f"Cross-board transfer: train on Board {args.source} (Unit {args.source}), "
          f"test on Board {args.target} (Unit {args.target})")
    print("-" * 78)
    print(f"{'classifier':<12}{'normalization':<16}{'accuracy':>10}   predicted-class histogram")
    for clf_name in clf_names:
        for norm in norms:
            acc, hist = cross_board(boards, args.source, args.target, clf_name, norm, args.seed)
            key = f"B{args.source}->B{args.target}|{clf_name}|{norm}"
            results["cross_board"][key] = {"accuracy": acc, "pred_histogram": hist}
            print(f"{clf_name:<12}{norm:<16}{acc * 100:>9.2f}%   {hist}")
    print(f"\nchance level for {N_CLASSES} balanced classes: {100 / N_CLASSES:.2f}%")

    if args.mode == "all":
        # ---- within-board supervised cross-validation ------------------- #
        print("\n" + "-" * 78)
        print("Within-board supervised 5-fold cross-validation (sanity check on parsing)")
        print("-" * 78)
        results["within_board_cv"] = {}
        header = f"{'board':<8}" + "".join(f"{c:>14}" for c in clf_names)
        print(header)
        for b in range(1, 6):
            row = f"Board {b:<2}"
            for clf_name in clf_names:
                mean, std = within_board_cv(boards, b, clf_name, "source_only", args.seed)
                results["within_board_cv"][f"B{b}|{clf_name}"] = {"mean": mean, "std": std}
                row += f"{mean * 100:>10.2f}%   "
            print(row)

        # ---- full source -> target matrix ------------------------------- #
        for clf_name in clf_names:
            print("\n" + "-" * 78)
            print(f"Cross-board accuracy matrix ({clf_name}, per_board normalization)")
            print("-" * 78)
            # The header literal is kept outside the f-string expression: a
            # backslash inside one is a SyntaxError before Python 3.12.
            header = "src\\tgt"
            print(f"{header:<10}" + "".join(f"{'B' + str(t):>10}" for t in range(1, 6)))
            offdiag = []
            for s in range(1, 6):
                row = f"B{s:<9}"
                for t in range(1, 6):
                    if s == t:
                        row += f"{'--':>10}"
                        continue
                    acc, _ = cross_board(boards, s, t, clf_name, "per_board", args.seed)
                    results["cross_board"][f"B{s}->B{t}|{clf_name}|per_board"] = {"accuracy": acc}
                    offdiag.append(acc)
                    row += f"{acc * 100:>9.2f}%"
                print(row)
            mean_off = float(np.mean(offdiag))
            results.setdefault("cross_board_mean", {})[clf_name] = mean_off
            print(f"\nmean off-diagonal (cross-board) accuracy: {mean_off * 100:.2f}%")

    print("\n" + "=" * 78)
    chance = 1.0 / N_CLASSES
    pair_accs = [v["accuracy"] for v in results["cross_board"].values()]
    mean_acc = sum(pair_accs) / len(pair_accs)
    cross_collapsed = mean_acc <= chance + 0.10
    within_ok = all(v["mean"] >= 0.85 for v in results.get("within_board_cv", {}).values()) \
        if results.get("within_board_cv") else None
    if args.source == args.target:
        print("Note: source and target are the same board, so this run is not a")
        print("cross-board transfer; pass different --source/--target for that check.")
    elif cross_collapsed and within_ok:
        print("Conclusion: within-board classification succeeds while cross-board")
        print(f"transfer stays near chance (mean {mean_acc:.2%} against the {chance:.0%}")
        print("chance level), so the low cross-board accuracy reflects a genuine")
        print("inter-board distribution shift in the 8-D steady-state space rather")
        print("than a parsing, labeling, or normalization error.")
    elif cross_collapsed:
        print(f"Summary: cross-board accuracy stays near the {chance:.0%} chance level")
        print(f"(mean {mean_acc:.2%}) for the configurations run here (within-board CV")
        print("not run; add --mode all to include it).")
    else:
        print(f"Summary: mean cross-board accuracy {mean_acc:.2%} against a {chance:.0%}")
        print("chance level for the configurations run here.")
    print("=" * 78)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
