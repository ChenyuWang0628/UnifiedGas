#!/usr/bin/env python3
"""Two-dimensional PCA score plots of the Dataset-B steady-state features.

Produces two figures:

``fig_datasetB_pca_boards.pdf``
    One panel per board (Boards 1-3 hold 160 samples each, Boards 4-5 hold 80
    each), each panel showing the first two principal components of that
    board's own 8-D steady-state features, colored by gas class.  Fitting the
    PCA inside a board shows how separable the four gases are for a single
    device.

``fig_datasetB_pca_joint.pdf``
    A single PCA fitted on all 640 samples, with color encoding the gas and
    marker shape encoding the board.  This exposes the inter-board offset that
    the per-board panels cannot show.

Usage
-----
    python plot_datasetB_pca.py --data_dir /path/to/DataSetB-raw --out_dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from reproduce_reviewer_check import GAS_FULL_NAME, LABEL_TO_GAS, N_CLASSES, load_boards

plt.rcParams.update(
    {
        # Type 42 (TrueType) instead of matplotlib's default Type 3: IEEE PDF
        # eXpress rejects Type 3 fonts, and Type 3 text is not searchable.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)

CLASS_COLORS = ["#1b6ca8", "#d1495b", "#2e8b57", "#e5a13a"]
BOARD_MARKERS = ["o", "s", "^", "D", "v"]


def plot_per_board(boards, out_path: Path):
    """One PCA panel per board, colored by gas class."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3))
    for idx, b in enumerate(range(1, 6)):
        ax = axes[idx]
        X, y = boards[b]
        # Standardize within the board so that no single high-variance sensor
        # dominates the projection.
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(Xs)
        evr = pca.explained_variance_ratio_ * 100
        for c in range(N_CLASSES):
            sel = y == c
            ax.scatter(
                Z[sel, 0],
                Z[sel, 1],
                s=16,
                alpha=0.85,
                c=CLASS_COLORS[c],
                edgecolors="none",
                label=GAS_FULL_NAME[LABEL_TO_GAS[c]],
            )
        ax.set_title(f"Board {b} ($n$ = {len(y)})")
        ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1]:.1f}%)")
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.tick_params(direction="in", top=True, right=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=N_CLASSES,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_joint(boards, out_path: Path):
    """A single PCA over all boards; color = gas, marker = board."""
    X_all = np.concatenate([boards[b][0] for b in range(1, 6)], axis=0)
    y_all = np.concatenate([boards[b][1] for b in range(1, 6)], axis=0)
    board_id = np.concatenate(
        [np.full(len(boards[b][1]), b, dtype=int) for b in range(1, 6)], axis=0
    )

    Xs = StandardScaler().fit_transform(X_all)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)
    evr = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for b in range(1, 6):
        for c in range(N_CLASSES):
            sel = (board_id == b) & (y_all == c)
            ax.scatter(
                Z[sel, 0],
                Z[sel, 1],
                s=18,
                alpha=0.8,
                c=CLASS_COLORS[c],
                marker=BOARD_MARKERS[b - 1],
                edgecolors="none",
            )
    ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1f}%)")
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.tick_params(direction="in", top=True, right=True)

    gas_handles = [
        plt.Line2D([], [], marker="o", ls="", color=CLASS_COLORS[c],
                   label=GAS_FULL_NAME[LABEL_TO_GAS[c]])
        for c in range(N_CLASSES)
    ]
    board_handles = [
        plt.Line2D([], [], marker=BOARD_MARKERS[b - 1], ls="", color="0.35",
                   label=f"Board {b}")
        for b in range(1, 6)
    ]
    leg1 = ax.legend(handles=gas_handles, loc="upper left", frameon=False,
                     title="Gas", fontsize=7.5, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=board_handles, loc="lower right", frameon=False,
              title="Board", fontsize=7.5, title_fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")

    # Quantify how much of the leading variance is board-related rather than
    # gas-related: between-group scatter along PC1-PC2 for each factor.
    def between_group_ratio(labels):
        total = Z.var(axis=0).sum()
        gm = Z.mean(axis=0)
        between = 0.0
        for g in np.unique(labels):
            sel = labels == g
            between += sel.sum() * ((Z[sel].mean(axis=0) - gm) ** 2).sum()
        return float(between / len(Z) / total)

    print(f"  variance explained by PC1+PC2 : {evr.sum():.1f}%")
    print(f"  between-board  share (PC1-2)  : {between_group_ratio(board_id) * 100:.1f}%")
    print(f"  between-gas    share (PC1-2)  : {between_group_ratio(y_all) * 100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description="2-D PCA score plots for Dataset B")
    ap.add_argument("--data_dir", required=True,
                    help="directory holding the raw B*_G*_F*_R*.txt recordings")
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting steady-state features:")
    boards = load_boards(args.data_dir)
    print()

    plot_per_board(boards, out_dir / "fig_datasetB_pca_boards.pdf")
    plot_joint(boards, out_dir / "fig_datasetB_pca_joint.pdf")


if __name__ == "__main__":
    main()
