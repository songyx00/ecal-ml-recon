#!/usr/bin/env python3
"""
Tool 6: Relative Energy Resolution vs True Energy

Compares MC truth cluster energy to reconstructed cluster energy by computing
delta = 1 - e_rec/true_e (unitless), and plotting per-bin resolution (rms90 or std)
as a function of true energy (GeV), following the style of plotEnergyResolution.py.

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
from typing import List, Tuple

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


def read_truth_reco_from_file(
    root_path: Path,
    tree: str = "cluster",
    branch_true: str = "true_eTot",
    branch_e: str = "e",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read aligned (true_eTot, e) arrays from one reconstruction ROOT file.
    Returns two 1D numpy arrays (may be empty if missing/invalid).
    """
    try:
        with uproot.open(root_path) as f:
            if tree not in f:
                logger.info(f"Skipping {root_path.name}: missing TTree '{tree}'")
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
            t = f[tree]
            if branch_true not in t.keys() or branch_e not in t.keys():
                logger.info(f"Skipping {root_path.name}: missing branch '{branch_true}' or '{branch_e}'")
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
            a_true = t[branch_true].array(library="np").astype(np.float64)
            a_e = t[branch_e].array(library="np").astype(np.float64)
            n = min(len(a_true), len(a_e))
            if n == 0:
                return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
            a_true = a_true[:n]
            a_e = a_e[:n]
            mask = np.isfinite(a_true) & np.isfinite(a_e) & (a_true > 0)
            return a_true[mask], a_e[mask]
    except Exception as e:
        logger.warning(f"Skipping {root_path.name}: {e}")
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description='Tool 6: Relative Energy Resolution vs True Energy')
    parser.add_argument('--recon_type', type=str, default='1',
                        help="Reconstruction type: '1' (3x3), '2' (optimal), or '1+2' for both")
    parser.add_argument('--root_file_number', type=int, default=-1,
                        help='Number of ROOT files to use (default: -1 => use all)')
    parser.add_argument('--bins', type=int, default=60,
                        help='Number of histogram bins for delta-to-hist (unused in rms90; kept for compatibility)')
    parser.add_argument('--true_scale', type=float, default=1.0,
                        help='Scale factor applied to true_eTot to express in GeV (default: 1.0)')
    parser.add_argument('--rec_scale', type=float, default=1e-3,
                        help='Scale factor applied to reconstructed energy to express in GeV (default: 1e-3, MeV→GeV)')
    parser.add_argument('--method', type=str, choices=['rms90', 'std'], default='rms90',
                        help='Resolution estimator per bin (default: rms90)')
    parser.add_argument('--bin_edges', type=str, default='0,1,2,3,5,10,20,30,50,65,80',
                        help='Comma-separated true energy bin edges in GeV (default: variable-width 0–80 GeV)')
    parser.add_argument('--output_dir', type=str, default='./output/energy_resolution_hist',
                        help='Output directory for PNG (default: ./output/energy_resolution_hist)')
    parser.add_argument('--save_to_file', action='store_true', default=False,
                        help='Save PNG instead of interactive display')
    parser.add_argument('--log_y', action='store_true', default=False,
                        help='Use logarithmic y-axis')

    args = parser.parse_args()

    # Backend
    if args.save_to_file:
        plt.switch_backend('Agg')
        logger.info("Using non-interactive backend for file saving")
    else:
        logger.info("Using interactive backend for plot display")

    base = Path("./data/recon_data")
    # Determine which reconstruction types to run
    recon_types: List[int] = [1, 2] if args.recon_type.strip() == '1+2' else [int(args.recon_type)]
    label_map = {1: '3x3', 2: 'optimal'}
    color_map = {1: 'tab:blue', 2: 'tab:orange'}

    # Prepare bin edges (shared across curves)
    try:
        edges = np.array([float(x) for x in args.bin_edges.split(',')], dtype=float)
        if np.any(np.diff(edges) <= 0):
            raise ValueError
    except Exception:
        raise ValueError("Invalid --bin_edges. Provide comma-separated increasing numbers, e.g., 0,1,2,3,5")

    curves = []  # list of tuples (x_plot, y_plot, label, color, nfiles)

    for rtype in recon_types:
        recon_dir = find_recon_dir(base, rtype)
        if not recon_dir.exists():
            raise FileNotFoundError(f"Reconstruction directory not found: {recon_dir}")

        root_files = sorted(recon_dir.glob("*.root"))
        if not root_files:
            raise FileNotFoundError(f"No ROOT files found in {recon_dir}")

        selected = []
        all_true: List[np.ndarray] = []
        all_reco: List[np.ndarray] = []
        for fpath in root_files:
            if args.root_file_number > 0 and len(selected) >= args.root_file_number:
                break
            true_vals, reco_vals = read_truth_reco_from_file(fpath)
            logger.info(f"Checked {fpath.name}: true={len(true_vals)}, reco={len(reco_vals)}")
            if true_vals.size == 0 or reco_vals.size == 0:
                continue
            true_vals = true_vals * float(args.true_scale)
            reco_vals = reco_vals * float(args.rec_scale)
            selected.append(fpath)
            all_true.append(true_vals)
            all_reco.append(reco_vals)

        if not all_true:
            raise RuntimeError("No valid truth/reco pairs found in available ROOT files.")

        true_all = np.concatenate(all_true).astype(np.float64)
        reco_all = np.concatenate(all_reco).astype(np.float64)

        # Compute delta_e per entry (GeV)
        delta_e = reco_all - true_all

        # Helper: rms90
        def rms90(values: np.ndarray) -> float:
            v = np.sort(np.asarray(values))
            n = len(v)
            if n == 0:
                return np.nan
            k = max(1, int(np.floor(0.9 * n)))
            widths = v[k-1:] - v[:n-k+1]
            idx = int(np.argmin(widths))
            win = v[idx:idx+k]
            m = float(win.mean())
            return float(np.sqrt(np.mean((win - m) ** 2)))

        # Bin by true energy and compute resolution as percentage using true energy per cluster
        bin_idx = np.digitize(true_all, edges) - 1
        nb = len(edges) - 1
        xvals, yvals = [], []
        for i in range(nb):
            sel = (bin_idx == i)
            if not np.any(sel):
                continue
            te = true_all[sel]
            de = delta_e[sel]
            if de.size == 0:
                continue
            e_center = 0.5 * (edges[i] + edges[i+1])
            perc_vals = 100.0 * (np.abs(de) / te)
            x_rep = float(e_center)
            if args.method == 'std':
                val = float(np.std(perc_vals, ddof=1))
            else:
                val = rms90(perc_vals)
            xvals.append(x_rep)
            yvals.append(val)

        if not xvals:
            raise RuntimeError("No bins with entries. Consider adjusting --bin_edges or data selection.")

        order = np.argsort(np.array(xvals))
        x_plot = np.array(xvals)[order]
        y_plot = np.array(yvals)[order]
        curves.append((x_plot, y_plot, label_map.get(rtype, str(rtype)), color_map.get(rtype, None), len(selected)))

    # Render figure
    fig, ax = plt.subplots(figsize=(8, 5))
    for x_plot, y_plot, label, color, nsel in curves:
        ax.plot(x_plot, y_plot, 'o-', label=f"{label}", color=color)
    ax.set_xlabel("True energy (GeV)", fontsize=14)
    ax.set_ylabel("Relative energy resolution (%)", fontsize=14)
    # Use edges from last computed (all curves share args.bin_edges)
    ax.set_xlim(edges[0], edges[-1])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if args.save_to_file:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = f"energy_resolution_curve_recon{args.recon_type.replace('+','_')}.png"
        outpath = outdir / outfile
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved energy resolution plot to: {outpath}")
    else:
        logger.info("Showing interactive plot window...")
        plt.show()


if __name__ == "__main__":
    main()
