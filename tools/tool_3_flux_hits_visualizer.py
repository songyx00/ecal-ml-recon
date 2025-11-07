#!/usr/bin/env python3
"""
Flux Hits Map Visualizer (Tool 3)

Visualizes hit entry points from flux ROOT files by binning (entry_x, entry_y)
into ECAL cells using detector geometry. Output is a cell-colored intensity map
similar to the PE count visualizer.

Sample types supported via --sample_type:
- 1: Single particle data (./data/mc_data/single_particle)
- 2: Multiple particles data (./data/mc_data/mip_multiple_particles)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from pathlib import Path
import argparse
import logging
import uproot
from typing import Dict, Tuple, List

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def load_geometry(outtrigd_path: Path) -> Dict[str, np.ndarray]:
    """
    Load ECAL cells geometry from an OutTrigd_*.root file (cells tree).
    """
    logger.info(f"Loading geometry from: {outtrigd_path}")
    with uproot.open(outtrigd_path) as f:
        cells = f["cells"].arrays(["x_mm", "y_mm", "dx_mm", "dy_mm"], library="np")
    geom = {
        "x": cells["x_mm"],
        "y": cells["y_mm"],
        "dx": cells["dx_mm"],
        "dy": cells["dy_mm"],
    }
    logger.info(f"Loaded geometry for {len(geom['x'])} cells")
    return geom


def count_hits_per_cell(geom: Dict[str, np.ndarray], hx: np.ndarray, hy: np.ndarray) -> np.ndarray:
    """
    Bin hit coordinates into cells by rectangle containment.
    Returns an array of counts per cell.
    """
    x, y, dx, dy = geom["x"], geom["y"], geom["dx"], geom["dy"]
    counts = np.zeros_like(x, dtype=np.int64)

    for i, (xi, yi, dxi, dyi) in enumerate(zip(x, y, dx, dy)):
        mask = (
            (hx >= xi - dxi / 2) & (hx < xi + dxi / 2) &
            (hy >= yi - dyi / 2) & (hy < yi + dyi / 2)
        )
        counts[i] = int(mask.sum())
    return counts


def read_hits_from_flux(flux_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    with uproot.open(flux_file) as f:
        arr = f["tree"].arrays(["entry_x", "entry_y"], library="np")
    return arr["entry_x"], arr["entry_y"]


def read_hit_map_for_flux_file(geom: Dict[str, np.ndarray], flux_file: Path) -> Tuple[Dict, str]:
    logger.info(f"Reading flux hits: {flux_file.name}")
    hx, hy = read_hits_from_flux(flux_file)
    logger.info(f"Hits loaded: {len(hx)}")
    counts = count_hits_per_cell(geom, hx, hy)

    cell_data = {
        "x": geom["x"],
        "y": geom["y"],
        "dx": geom["dx"],
        "dy": geom["dy"],
        "hits": counts,
    }
    return cell_data, flux_file.name


def plot_hit_intensity_map(cell_data: Dict, filename: str, ax: plt.Axes, log_scale: bool = False) -> None:
    x = cell_data['x']
    y = cell_data['y']
    dx = cell_data['dx']
    dy = cell_data['dy']
    hits = cell_data['hits']

    if log_scale:
        vmin = np.min(hits[hits > 0]) if np.any(hits > 0) else 1
        vmax = np.max(hits) if hits.size else 1
        cmap = plt.cm.viridis
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        logger.info(f"Using log scale: vmin={vmin}, vmax={vmax}")
    else:
        vmin = np.min(hits[hits > 0]) if np.any(hits > 0) else 0
        vmax = np.max(hits) if hits.size else 1
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        logger.info(f"Using linear scale: vmin={vmin}, vmax={vmax}")

    logger.info(f"Drawing cells with hit count colors for {filename}...")
    for xi, yi, dxi, dyi, h in zip(x, y, dx, dy, hits):
        color = cmap(norm(h)) if h > 0 else "white"
        rect = patches.Rectangle(
            (xi - dxi / 2, yi - dyi / 2),
            dxi, dyi,
            facecolor=color,
            edgecolor='black',
            linewidth=0.3
        )
        ax.add_patch(rect)

    ax.set_xlim(np.min(x - dx / 2), np.max(x + dx / 2))
    ax.set_ylim(np.min(y - dy / 2), np.max(y + dy / 2))
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]", fontsize=24)
    ax.set_ylabel("y [mm]", fontsize=24)
    ax.set_title(f"{filename}", fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=20)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Hit count per cell" + (" - Log Scale" if log_scale else " - Linear Scale"), fontsize=20)
    cbar.ax.tick_params(labelsize=16)


def calculate_grid_layout(num_files: int) -> Tuple[int, int]:
    if num_files == 1:
        return 1, 1
    elif num_files == 2:
        return 1, 2
    else:
        cols = 3
        rows = (num_files + cols - 1) // cols
        return rows, cols


def main():
    parser = argparse.ArgumentParser(description='Flux Hits Map Visualizer (Tool 3)')
    parser.add_argument('--sample_type', type=int, choices=[1, 2], default=1,
                        help='1: single particle, 2: multiple particles (default: 1)')
    parser.add_argument('--root_file_number', type=int, default=1,
                        help='Number of flux ROOT files to process (default: 1)')
    parser.add_argument('--output_dir', type=str, default='./output/flux_maps',
                        help='Output directory (default: ./output/flux_maps)')
    parser.add_argument('--save_to_file', action='store_true', default=False,
                        help='Save PNG instead of interactive display')
    parser.add_argument('--log_scale', action='store_true', default=False,
                        help='Use log color scale')

    args = parser.parse_args()

    if args.save_to_file:
        plt.switch_backend('Agg')
        logger.info("Using non-interactive backend for file saving")
    else:
        logger.info("Using interactive backend for plot display")

    if args.sample_type == 1:
        data_dir = Path("./data/mc_data/single_particle")
        sample_name = "single_particle"
    else:
        data_dir = Path("./data/mc_data/mip_multiple_particles")
        sample_name = "multiple_particles"

    flux_dir = data_dir / "fluxFiles"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not flux_dir.exists():
        raise FileNotFoundError(f"Flux directory not found: {flux_dir}")

    outtrigd_files = sorted(list(data_dir.glob("OutTrigd_*.root")))
    if not outtrigd_files:
        raise FileNotFoundError(f"No OutTrigd_*.root found in {data_dir} for geometry")
    geom = load_geometry(outtrigd_files[0])

    flux_files = sorted(list(flux_dir.glob("flux_*.root")))
    if not flux_files:
        raise FileNotFoundError(f"No flux_*.root files found in {flux_dir}")

    logger.info(f"Sample type: {args.sample_type} ({sample_name})")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Flux directory: {flux_dir}")
    logger.info(f"Requested valid flux files: {args.root_file_number} (out of {len(flux_files)} available)")
    logger.info(f"Save to file: {args.save_to_file}")
    logger.info(f"Log scale: {args.log_scale}")

    hit_maps: List[Tuple[Dict, str]] = []
    skipped_zero_raw = 0
    skipped_zero_binned = 0
    for fpath in flux_files:
        if len(hit_maps) >= args.root_file_number:
            break
        try:
            hx, hy = read_hits_from_flux(fpath)
            raw_hits = int(len(hx))
            logger.info(f"Reading flux hits: {fpath.name} (raw hits: {raw_hits})")
            if raw_hits == 0:
                skipped_zero_raw += 1
                logger.info(f"⏭️  Skipping {fpath.name}: zero raw hits")
                continue

            cell_data, fname = read_hit_map_for_flux_file(geom, fpath)
            binned_hits = int(np.asarray(cell_data['hits']).sum())
            logger.info(f"Binned hits in geometry: {binned_hits}")
            if binned_hits == 0:
                skipped_zero_binned += 1
                logger.info(f"⏭️  Skipping {fpath.name}: zero binned hits (outside geometry)")
                continue

            hit_maps.append((cell_data, fname))
            logger.info(f"✅ Accepted: {fpath.name}")
        except Exception as e:
            logger.warning(f"⏭️  Skipping {fpath.name}: {e}")
            continue

    if not hit_maps:
        raise RuntimeError("No flux hit maps could be produced. Check your ROOT files and directories.")

    rows, cols = calculate_grid_layout(len(hit_maps))
    logger.info(f"Grid layout: {rows} rows x {cols} columns")

    fig_width, fig_height = 9, 6
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if len(hit_maps) == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if cols > 1 else [axes]
    else:
        axes = axes.flatten()

    for i, (cell_data, fname) in enumerate(hit_maps):
        if i < len(axes):
            plot_hit_intensity_map(cell_data, fname, axes[i], args.log_scale)
        else:
            logger.warning("More maps than subplot slots available")
            break

    for i in range(len(hit_maps), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if args.save_to_file:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = f"flux_hit_maps_{sample_name}_{len(hit_maps)}files.png"
        outpath = outdir / outfile
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        logger.info(f"Flux hits visualization saved to: {outpath}")
        logger.info("\n🎉 Flux hits visualization completed successfully!")
        logger.info(f"Sample type: {sample_name}")
        logger.info(f"Files processed: {len(hit_maps)}")
        logger.info(f"Output file: {outfile}")
    else:
        logger.info("\n🎉 Flux hits visualization completed successfully!")
        logger.info(f"Sample type: {sample_name}")
        logger.info(f"Files processed: {len(hit_maps)}")
        logger.info("Showing interactive plot window...")
        plt.show()

    logger.info(f"Skipped files with zero raw hits: {skipped_zero_raw}")
    logger.info(f"Skipped files with zero binned hits: {skipped_zero_binned}")


if __name__ == "__main__":
    main()
