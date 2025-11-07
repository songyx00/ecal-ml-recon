#!/usr/bin/env python3
"""
Photo Electronic Count Map Visualizer

This script visualizes photo electronic count maps from ROOT files.
It supports two sample types:
- 1: Single particle data
- 2: Multiple particles data

The script reads ROOT files and creates intensity maps showing PE counts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from pathlib import Path
import argparse
import logging
import uproot
import awkward as ak
from typing import Dict, Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Set matplotlib to interactive backend by default
# plt.switch_backend('Agg')  # Commented out to allow interactive plots

def read_pe_map_from_root(root_file_path: str) -> Tuple[Dict, str, List[str]]:
    """
    Read PE count map from ROOT file using the same logic as enhanced_pe_visualizer.py.
    
    Args:
        root_file_path: Path to the ROOT file
        
    Returns:
        Tuple of (cell data dictionary, ROOT filename, photon branch names)
    """
    try:
        logger.info(f"Opening ROOT file for PE count analysis: {root_file_path}")
        
        with uproot.open(root_file_path) as file:
            # Step 1: Load cell geometry from "cells" tree
            cells_tree = file["cells"]
            cells = cells_tree.arrays(["x_mm", "y_mm", "dx_mm", "dy_mm", "module_id_in_calo", "cell_id_in_module"])
        
        x = cells["x_mm"]
        y = cells["y_mm"]
        dx = cells["dx_mm"]
        dy = cells["dy_mm"]
        mod_id = cells["module_id_in_calo"]
        cell_id = cells["cell_id_in_module"]
        
        logger.info(f"Loaded {len(x)} cells for PE analysis")
        
        # Step 2: Load readout data from "tree" tree
        tree = file["tree"]
        all_branch_names = tree.keys()
        
        # Find meaningful photon branches
        nonempty_branches = []
        for name in all_branch_names:
            if name.endswith("_ph"):
                arr = tree[name].array(entry_start=0, entry_stop=1)
                if len(arr[0]) > 0:  # Non-empty vector
                    nonempty_branches.append(name)
        
        logger.info(f"Found {len(nonempty_branches)} meaningful photon branches")
        logger.info(f"Photon branch names: {nonempty_branches}")
        
        # Check if we have any meaningful branches
        if not nonempty_branches:
            logger.warning(f"No meaningful photon branches found in {Path(root_file_path).name}")
            raise ValueError("No meaningful photon branches found")
        
        # Read meaningful branches
        try:
            readout_arrays = tree.arrays(nonempty_branches)
        except Exception as e:
            logger.warning(f"Failed to read branches from {Path(root_file_path).name}: {e}")
            raise ValueError(f"Failed to read branches: {e}")
        
        # Step 3: Extract PE for each cell
        pe_list = []
        logger.info(f"Available fields: {readout_arrays.fields}")
        
        try:
            for mid, cid in zip(mod_id, cell_id):
                branch_name = f"mod{mid}_ph"
                if branch_name in readout_arrays.fields:
                    vec = readout_arrays[branch_name][0]  # arrays is length-1 ak.Array
                    pe = vec[cid] if cid < len(vec) else 0
                else:
                    pe = 0
                pe_list.append(pe)
        except Exception as e:
            logger.warning(f"Failed to extract PE data from {Path(root_file_path).name}: {e}")
            raise ValueError(f"Failed to extract PE data: {e}")
        
        pe_array = np.array(pe_list)
        non_zero_count = np.count_nonzero(pe_array)
        logger.info(f"Non-zero PE count: {non_zero_count}")
        logger.info(f"PE range: {np.min(pe_array)} to {np.max(pe_array)}")
        
        # Check if we have meaningful PE data
        if non_zero_count == 0:
            logger.warning(f"No valid photon count data found in {Path(root_file_path).name}")
            raise ValueError("No valid photon count data found")
        
        # Keep as awkward arrays like enhanced_pe_visualizer.py does
        cell_data = {
            'x': x,
            'y': y,
            'dx': dx,
            'dy': dy,
            'pe': pe_array,
            'mod_id': mod_id,
            'cell_id': cell_id
        }
        
        # Get filename for display
        root_filename = Path(root_file_path).name
        
        logger.info(f"Successfully loaded cell data from {root_filename}: {len(x)} cells")
        logger.info(f"PE map statistics: min={np.min(pe_array)}, max={np.max(pe_array)}, mean={np.mean(pe_array):.2f}")
        
        return cell_data, root_filename, nonempty_branches
            
    except Exception as e:
        logger.error(f"Error reading ROOT file {root_file_path}: {e}")
        raise


def plot_pe_intensity_map(cell_data: Dict, filename: str, ax: plt.Axes, log_scale: bool = False) -> None:
    """
    Plot PE intensity map for a single ROOT file.
    
    Args:
        cell_data: Dictionary containing cell geometry and PE data
        filename: ROOT filename for title
        ax: Matplotlib axes to plot on
    """
    x = cell_data['x']
    y = cell_data['y']
    dx = cell_data['dx']
    dy = cell_data['dy']
    pe = cell_data['pe']
    
    # Colormap setup - Non-zero values use colormap, zero values are white
    if log_scale:
        # For log scale, use LogNorm and handle zero values
        vmin = np.min(pe[pe > 0]) if np.any(pe > 0) else 0.1
        vmax = np.max(pe)
        cmap = plt.cm.viridis
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        logger.info(f"Using log scale: vmin={vmin:.2f}, vmax={vmax:.2f}")
    else:
        # Linear scale
        vmin = np.min(pe[pe > 0]) if np.any(pe > 0) else 0.1
        vmax = np.max(pe)
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        logger.info(f"Using linear scale: vmin={vmin:.2f}, vmax={vmax:.2f}")
    
    # Draw cells with PE count colors
    logger.info(f"Drawing cells with PE count colors for {filename}...")
    for xi, yi, dxi, dyi, pe_val in zip(x, y, dx, dy, pe):
        if pe_val > 0:
            color = cmap(norm(pe_val))
        else:
            color = "white"
        rect = patches.Rectangle(
            (xi - dxi / 2, yi - dyi / 2),
            dxi, dyi,
            facecolor=color,
            edgecolor='black',
            linewidth=0.3
        )
        ax.add_patch(rect)
    
    # Coordinate settings with larger fonts
    ax.set_xlim(np.min(x - dx / 2), np.max(x + dx / 2))
    ax.set_ylim(np.min(y - dy / 2), np.max(y + dy / 2))
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]", fontsize=24)  # Increased from default ~12
    ax.set_ylabel("y [mm]", fontsize=24)  # Increased from default ~12
    ax.set_title(f"{filename}", fontsize=28)  # Increased from default ~14
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=20)  # Increased from default ~10
    
    # Add colorbar with larger font
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    
    # Set colorbar label based on scale type
    if log_scale:
        cbar.set_label("PE count (photon-electrons) - Log Scale", fontsize=20)
    else:
        cbar.set_label("PE count (photon-electrons) - Linear Scale", fontsize=20)
    
    cbar.ax.tick_params(labelsize=16)  # Increase colorbar tick labels


def calculate_grid_layout(num_files: int) -> Tuple[int, int]:
    """
    Calculate grid layout for subplots.
    3 columns per row, fill columns first before moving to next row.
    
    Args:
        num_files: Number of files to plot
        
    Returns:
        Tuple of (rows, cols)
    """
    if num_files == 1:
        return 1, 1
    elif num_files == 2:
        return 1, 2
    else:
        cols = 3
        rows = (num_files + cols - 1) // cols  # Ceiling division
        return rows, cols


def main():
    """Main function to visualize PE count maps."""
    parser = argparse.ArgumentParser(description='Photo Electronic Count Map Visualizer')
    parser.add_argument('--sample_type', type=int, choices=[1, 2], default=1,
                       help='Sample type: 1 for single particle, 2 for multiple particles (default: 1)')
    parser.add_argument('--root_file_number', type=int, default=1,
                       help='Number of ROOT files to read (default: 1)')
    parser.add_argument('--output_dir', type=str, default='./output/pe_maps',
                       help='Output directory for visualizations (default: ./output/pe_maps)')
    parser.add_argument('--save_to_file', action='store_true', default=False,
                       help='Save plot to PNG file instead of showing interactive window (default: False - shows interactive plot)')
    parser.add_argument('--log_scale', action='store_true', default=False,
                       help='Use log scale for PE count colorbar (default: False - uses linear scale)')
    
    args = parser.parse_args()
    
    # Determine data directory based on sample type
    if args.sample_type == 1:
        data_dir = Path("./data/mc_data/single_particle")
        sample_name = "single_particle"
    else:
        data_dir = Path("./data/mc_data/mip_multiple_particles")
        sample_name = "multiple_particles"
    
    logger.info(f"Sample type: {args.sample_type} ({sample_name})")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Number of ROOT files to read: {args.root_file_number}")
    logger.info(f"Save to file: {args.save_to_file}")
    logger.info(f"Log scale: {args.log_scale}")
    
    # Set matplotlib backend based on save_to_file flag
    if args.save_to_file:
        plt.switch_backend('Agg')  # Non-interactive for file saving
        logger.info("Using non-interactive backend for file saving")
    else:
        logger.info("Using interactive backend for plot display")
    
    # Check if data directory exists
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    # Find ROOT files
    root_files = list(data_dir.glob("OutTrigd_*.root"))
    if not root_files:
        logger.error(f"No ROOT files found in {data_dir}")
        raise FileNotFoundError(f"No ROOT files found in {data_dir}")
    
    logger.info(f"Found {len(root_files)} total ROOT files in {data_dir}")
    
    # Sort files by ID number for consistent ordering
    def extract_file_id(filename):
        """Extract numeric ID from filename like 'OutTrigd_0.root'"""
        try:
            return int(filename.stem.split('_')[1])
        except (IndexError, ValueError):
            return 0
    
    root_files.sort(key=extract_file_id)
    
    # Limit number of files to read
    num_files_to_read = min(args.root_file_number, len(root_files))
    selected_files = root_files[:num_files_to_read]
    
    logger.info(f"Will process {len(selected_files)} ROOT files")
    
    # Read PE maps from ROOT files - try to get the requested number of valid files
    pe_maps = []
    skipped_files = []
    files_processed = 0
    all_branch_info = []  # Store branch information for each file
    
    # If we need more files than initially selected, expand the list
    all_files = root_files  # Use all available files
    files_to_try = all_files if len(all_files) >= args.root_file_number else all_files
    
    logger.info(f"Will try to read {args.root_file_number} valid PE maps from {len(files_to_try)} available files")
    
    for root_file in files_to_try:
        if len(pe_maps) >= args.root_file_number:
            break  # We have enough valid files
            
        try:
            pe_map, root_filename, branch_names = read_pe_map_from_root(str(root_file))
            pe_maps.append((pe_map, root_filename))
            all_branch_info.append((root_filename, branch_names))
            files_processed += 1
            logger.info(f"✅ Read PE map {len(pe_maps)}/{args.root_file_number}: {root_filename}")
        except ValueError as e:
            logger.info(f"⏭️  Skipping {root_file.name}: {e}")
            skipped_files.append(root_file.name)
            continue
        except Exception as e:
            logger.warning(f"❌ Failed to read {root_file.name}: {e}")
            skipped_files.append(root_file.name)
            continue
    
    if not pe_maps:
        logger.error("No PE maps were successfully read from ROOT files!")
        if skipped_files:
            logger.error(f"Skipped files: {', '.join(skipped_files)}")
        raise RuntimeError("No PE maps could be loaded. Check your ROOT files and data directory.")
    
    logger.info(f"Successfully loaded {len(pe_maps)} PE maps for visualization")
    if skipped_files:
        logger.info(f"Skipped {len(skipped_files)} files without valid photon data: {', '.join(skipped_files)}")
    
    # Calculate grid layout
    rows, cols = calculate_grid_layout(len(pe_maps))
    logger.info(f"Grid layout: {rows} rows x {cols} columns")
    
    # Create the visualization with standard 9:6 ratio (width:height)
    # Total figure: 9 inches width, 6 inches height
    fig_width = 9  # Total width in inches
    fig_height = 6  # Total height in inches
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    # No figure title
    
    # Handle single subplot case
    if len(pe_maps) == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each PE map
    for i, (cell_data, root_filename) in enumerate(pe_maps):
        if i < len(axes):
            plot_pe_intensity_map(cell_data, root_filename, axes[i], args.log_scale)
        else:
            logger.warning(f"More PE maps than subplot slots available")
            break
    
    # Hide unused subplots
    for i in range(len(pe_maps), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if args.save_to_file:
        # Create output directory and save
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"pe_count_maps_{sample_name}_{len(pe_maps)}files.png"
        output_path = output_dir / output_filename
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"PE count maps visualization saved to: {output_path}")
        
        # Print summary
        logger.info(f"\n🎉 PE count visualization completed successfully!")
        logger.info(f"Sample type: {sample_name}")
        logger.info(f"Files processed: {len(pe_maps)}")
        logger.info(f"Output file: {output_filename}")
    else:
        # Show interactive plot
        logger.info(f"\n🎉 PE count visualization completed successfully!")
        logger.info(f"Sample type: {sample_name}")
        logger.info(f"Files processed: {len(pe_maps)}")
        logger.info("Showing interactive plot window...")
        plt.show()
    
    # Print PE statistics for each file
    logger.info(f"\n📊 PE Statistics:")
    for i, (cell_data, root_filename) in enumerate(pe_maps):
        pe = cell_data['pe']
        non_zero_count = np.count_nonzero(pe)
        logger.info(f"  File {i+1} ({root_filename}):")
        logger.info(f"    - Total cells: {len(pe)}")
        logger.info(f"    - Non-zero PE cells: {non_zero_count}")
        logger.info(f"    - PE range: {np.min(pe):.1f} to {np.max(pe):.1f}")
        logger.info(f"    - Mean PE: {np.mean(pe):.2f}")
    
    # Print ROOT tree branch information table
    logger.info(f"\n🌳 ROOT Tree Branch Information:")
    logger.info("=" * 80)
    logger.info(f"{'File':<20} {'Branch Name':<20} {'Description':<35}")
    logger.info("=" * 80)
    
    for filename, branch_names in all_branch_info:
        for i, branch_name in enumerate(branch_names):
            if i == 0:
                # First branch shows filename
                logger.info(f"{filename:<20} {branch_name:<20} {'Reconstructed photon count':<35}")
            else:
                # Subsequent branches show empty filename
                logger.info(f"{'':<20} {branch_name:<20} {'Reconstructed photon count':<35}")
    
    logger.info("=" * 80)
    logger.info(f"Total files processed: {len(all_branch_info)}")
    logger.info(f"Total photon branches: {sum(len(branches) for _, branches in all_branch_info)}")
    logger.info("Note: These are reconstructed PE counts from detector modules, not MC truth")


if __name__ == "__main__":
    main()
