#!/usr/bin/env python3
"""
Tool 4: MC Truth Energy Histogram Visualizer

Reads reconstruction ROOT files under ./data/recon_data and plots a histogram
of MC truth energy per cluster (branch: true_eTot) from the 'cluster' TTree.

Recon type selection via --recon_type:
- 1: 3x3 reconstruction (./data/recon_data/3x3)
- 2: OPT reconstruction (./data/recon_data/opt)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import uproot
from typing import List

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def find_recon_dir(base: Path, recon_type: int) -> Path:
    if recon_type == 1:
        return base / "3x3"
    elif recon_type == 2:
        return base / "opt"
    else:
        raise ValueError("Invalid recon_type. Use 1 for 3x3 or 2 for optimal.")


def read_truth_energy_from_file(root_path: Path, tree: str = "cluster", branch_true: str = "true_eTot") -> np.ndarray:
    """
    Read MC truth total energy from one reconstruction ROOT file.
    Returns a 1D numpy array (may be empty if missing/invalid).
    """
    try:
        with uproot.open(root_path) as f:
            if tree not in f:
                logger.info(f"Skipping {root_path.name}: missing TTree '{tree}'")
                return np.array([], dtype=np.float64)
            t = f[tree]
            if branch_true not in t.keys():
                logger.info(f"Skipping {root_path.name}: missing branch '{branch_true}'")
                return np.array([], dtype=np.float64)
            arr = t[branch_true].array(library="np")
            vals = np.asarray(arr, dtype=np.float64)
            mask = np.isfinite(vals) & (vals > 0)
            return vals[mask]
    except Exception as e:
        logger.warning(f"Skipping {root_path.name}: {e}")
        return np.array([], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description='Tool 4: MC Truth Energy Histogram Visualizer')
    parser.add_argument('--recon_type', type=int, choices=[1, 2], default=1,
                        help='1: 3x3, 2: optimal (default: 1)')
    parser.add_argument('--root_file_number', type=int, default=1,
                        help='Number of ROOT files to use (default: 1)')
    parser.add_argument('--bins', type=int, default=60,
                        help='Number of histogram bins (default: 60)')
    parser.add_argument('--true_scale', type=float, default=1.0,
                        help='Scale factor applied to true_eTot to express in GeV (default: 1.0)')
    parser.add_argument('--output_dir', type=str, default='./output/mc_truth_hist',
                        help='Output directory for PNG (default: ./output/mc_truth_hist)')
    parser.add_argument('--save_to_file', action='store_true', default=False,
                        help='Save PNG instead of interactive display')
    parser.add_argument('--log_y', action='store_true', default=False,
                        help='Use logarithmic y-axis')

    args = parser.parse_args()

    if args.save_to_file:
        plt.switch_backend('Agg')
        logger.info("Using non-interactive backend for file saving")
    else:
        logger.info("Using interactive backend for plot display")

    base = Path("./data/recon_data")
    recon_dir = find_recon_dir(base, args.recon_type)

    if not recon_dir.exists():
        raise FileNotFoundError(f"Reconstruction directory not found: {recon_dir}")

    root_files = sorted(recon_dir.glob("*.root"))
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in {recon_dir}")

    selected = []
    truths: List[np.ndarray] = []
    for fpath in root_files:
        if len(selected) >= args.root_file_number:
            break
        vals = read_truth_energy_from_file(fpath)
        logger.info(f"Checked {fpath.name}: truth entries = {len(vals)}")
        if vals.size == 0:
            continue
        selected.append(fpath)
        truths.append(vals)

    if not truths:
        raise RuntimeError("No valid truth energy entries found in available ROOT files.")

    truth_all = np.concatenate(truths).astype(np.float64)
    truth_all = truth_all * float(args.true_scale)

    if truth_all.size:
        p = np.percentile(truth_all, [1, 50, 99])
        logger.info(f"Truth energy (GeV) percentiles: p1={p[0]:.3g}, p50={p[1]:.3g}, p99={p[2]:.3g}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(truth_all, bins=int(args.bins), color='tab:blue', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Cluster Energy (MC Truth) [GeV]", fontsize=14)
    ax.set_ylabel("Entries", fontsize=14)
    if args.log_y:
        ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    if args.save_to_file:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = f"mc_truth_hist_recon{args.recon_type}_{len(selected)}files.png"
        outpath = outdir / outfile
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved MC truth histogram to: {outpath}")
    else:
        logger.info("Showing interactive plot window...")
        plt.show()


if __name__ == "__main__":
    main()
