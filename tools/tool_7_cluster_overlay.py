#!/usr/bin/env python3
"""
Tool 7: Cluster Overlay on ECAL Geometry/PE Map (best-effort)

Goal: Visualize locations of reconstructed clusters on top of the ECAL layout,
optionally overlaying a PE map from an OutTrigd event. Because available
cluster-location branches can vary, this tool tries multiple strategies:

1) Membership overlay (best): if the cluster tree provides per-cluster cell
   membership (module IDs and cell IDs), highlight those exact cells.
2) Centroid markers (fallback): if only coordinates (x_mm, y_mm) are present,
   plot markers at cluster centroids.

Inputs:
- Recon files: ./data/recon_data/{3x3|opt}/*.root, tree: 'cluster'
- Geometry (and optional PE): OutTrigd_*.root under ./data/mc_data/single_particle

Notes:
- Event-level linking is often needed to show the correct event PE map. If
  explicit event indices are not available in the 'cluster' tree, this tool
  will draw only the geometry and overlay cluster positions (no event PE).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
import logging
import uproot
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def load_geometry(outtrigd_path: Path) -> Dict[str, np.ndarray]:
    with uproot.open(outtrigd_path) as f:
        cells = f["cells"].arrays(["x_mm", "y_mm", "dx_mm", "dy_mm"], library="np")
    return {
        "x": cells["x_mm"],
        "y": cells["y_mm"],
        "dx": cells["dx_mm"],
        "dy": cells["dy_mm"],
    }


def find_recon_dir(base: Path, recon_type: int) -> Path:
    if recon_type == 1:
        return base / "3x3"
    elif recon_type == 2:
        return base / "opt"
    raise ValueError("Invalid recon_type: use 1 (3x3) or 2 (optimal)")


def detect_membership_branches(keys: List[str]) -> Tuple[str, str]:
    """Try to guess membership branch names for module and cell IDs."""
    candidates = [
        ("cells_mod", "cells_id"),
        ("member_module_ids", "member_cell_ids"),
        ("module_ids", "cell_ids"),
    ]
    for m_key, c_key in candidates:
        if m_key in keys and c_key in keys:
            return m_key, c_key
    return "", ""


def detect_centroid_branches(keys: List[str]) -> Tuple[str, str]:
    """Try to guess centroid coordinate branch names in mm."""
    x_cands = ["x_mm", "center_x_mm", "x"]
    y_cands = ["y_mm", "center_y_mm", "y"]
    x_key = next((k for k in x_cands if k in keys), "")
    y_key = next((k for k in y_cands if k in keys), "")
    return x_key, y_key


def detect_id_branch(keys: List[str]) -> str:
    """Try to guess a cluster identifier branch name."""
    id_cands = ["cluster_id", "id", "index", "idx"]
    return next((k for k in id_cands if k in keys), "")


def draw_geometry(ax: plt.Axes, geom: Dict[str, np.ndarray]) -> None:
    x, y, dx, dy = geom["x"], geom["y"], geom["dx"], geom["dy"]
    for xi, yi, dxi, dyi in zip(x, y, dx, dy):
        rect = patches.Rectangle((xi - dxi/2, yi - dyi/2), dxi, dyi,
                                 facecolor='white', edgecolor='lightgray', linewidth=0.3)
        ax.add_patch(rect)
    ax.set_xlim(np.min(x - dx / 2), np.max(x + dx / 2))
    ax.set_ylim(np.min(y - dy / 2), np.max(y + dy / 2))
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')


def overlay_membership(ax: plt.Axes, geom: Dict[str, np.ndarray],
                       mod_ids: np.ndarray, cell_ids: np.ndarray,
                       color: str = 'tab:red', alpha: float = 0.4) -> None:
    """
    Highlight cells for which we only know (module_id, cell_id) within that module.
    Without a global cell index mapping, this function cannot pick exact rectangles
    unless there is a known translation from (module, cell) to a unique cell index.
    Therefore, this function currently logs a note and falls back to centroid plotting
    if we cannot compute a global index. Extend here if mapping is available.
    """
    logger.info("Membership arrays detected, but no module/cell -> global cell index mapping is implemented."
                " Overlay will proceed via centroids if available.")


def main():
    parser = argparse.ArgumentParser(description='Tool 7: Cluster Overlay on ECAL Layout')
    parser.add_argument('--recon_type', type=int, choices=[1, 2], default=1,
                        help='1: 3x3, 2: optimal (default: 1)')
    parser.add_argument('--root_file_number', type=int, default=1,
                        help='Number of recon ROOT files to sample (default: 1)')
    parser.add_argument('--cluster_limit', type=int, default=50,
                        help='Max clusters to overlay (default: 50)')
    parser.add_argument('--outtrigd_path', type=str, default='',
                        help='Path to an OutTrigd_*.root (optional). If empty, use first under single_particle.')
    parser.add_argument('--save_to_file', action='store_true', default=False,
                        help='Save PNG instead of interactive display')
    parser.add_argument('--print_only', action='store_true', default=False,
                        help='Only print the input files that would be used and exit')

    args = parser.parse_args()

    # Backend
    if args.save_to_file:
        plt.switch_backend('Agg')
        logger.info("Using non-interactive backend for file saving")
    else:
        logger.info("Using interactive backend for plot display")

    # Geometry
    if args.outtrigd_path:
        outtrigd = Path(args.outtrigd_path)
    else:
        sp = Path('./data/mc_data/single_particle')
        candidates = sorted(sp.glob('OutTrigd_*.root'))
        if not candidates:
            raise FileNotFoundError("No OutTrigd_*.root found under ./data/mc_data/single_particle")
        outtrigd = candidates[0]
    # Recon
    recon_dir = find_recon_dir(Path('./data/recon_data'), args.recon_type)
    recon_files = sorted(recon_dir.glob('*.root'))
    if not recon_files:
        raise FileNotFoundError(f"No ROOT files found in {recon_dir}")

    selected = recon_files[:max(1, args.root_file_number)]

    # Print only mode
    if args.print_only:
        print("=== Cluster Overlay Inputs ===")
        print(f"OutTrigd file: {outtrigd.resolve()}")
        print(f"Recon type: {args.recon_type}")
        print(f"Recon files used: {len(selected)}")
        for p in selected:
            try:
                print(p.resolve())
            except Exception:
                print(str(p))
        return

    # Load geometry for plotting
    geom = load_geometry(outtrigd)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_geometry(ax, geom)

    # Colors for cluster markers
    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink']

    clusters_plotted = 0
    for rf in selected:
        logger.info(f"Reading clusters from: {rf.name}")
        with uproot.open(rf) as f:
            if 'cluster' not in f:
                logger.info("No 'cluster' tree in file, skipping.")
                continue
            t = f['cluster']
            keys = list(t.keys())
            m_key, c_key = detect_membership_branches(keys)
            x_key, y_key = detect_centroid_branches(keys)
            id_key = detect_id_branch(keys)

            # Prefer centroid; if not available, just log available keys
            if x_key and y_key:
                fields = [x_key, y_key]
                if id_key:
                    fields.append(id_key)
                arr = t.arrays(fields, library='np')
                xs = np.asarray(arr[x_key], dtype=float)
                ys = np.asarray(arr[y_key], dtype=float)
                ids = None
                if id_key:
                    try:
                        ids = np.asarray(arr[id_key])
                    except Exception:
                        ids = None
                n = len(xs)
                take = min(n, args.cluster_limit - clusters_plotted)
                if take <= 0:
                    break
                xs = xs[:take]
                ys = ys[:take]
                if ids is not None and len(ids) >= take:
                    lab_ids = ids[:take]
                else:
                    # Fallback to running indices
                    lab_ids = np.arange(clusters_plotted, clusters_plotted + take)
                color = colors[clusters_plotted % len(colors)]
                ax.plot(xs, ys, 'o', ms=4, mfc='none', mec=color, mew=1.0, label=f'{rf.stem} (centroid)')
                # Annotate each centroid with its ID/index
                for xi, yi, cid in zip(xs, ys, lab_ids):
                    ax.text(xi + 2.0, yi + 2.0, str(int(cid)), color=color, fontsize=7)
                clusters_plotted += take
            elif m_key and c_key:
                # Placeholder: membership-based overlay requires mapping
                logger.info(f"Membership branches detected: {m_key}, {c_key} (not plotted; mapping unavailable)")
            else:
                logger.info("No centroid or membership branches found. Available keys: " + ", ".join(keys[:50]))

        if clusters_plotted >= args.cluster_limit:
            break

    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    if args.save_to_file:
        outdir = Path('./output/cluster_overlay')
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"cluster_overlay_recon{args.recon_type}.png"
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        logger.info(f"Saved overlay to: {outfile}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
