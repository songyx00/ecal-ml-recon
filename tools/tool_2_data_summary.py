#!/usr/bin/env python3
"""
Data Summary Script for Einstein Project
Analyzes Monte Carlo simulation data for ECAL electromagnetic calorimeter at LHCb detector
"""

import os
import glob
from pathlib import Path


def count_root_files(directory, pattern):
    """Count ROOT files matching a specific pattern in a directory."""
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    return len(files)


def analyze_mc_data():
    """Analyze Monte Carlo simulation data."""
    print("=== MONTE CARLO SIMULATION DATA ANALYSIS ===")
    print()
    
    # Base paths
    mc_base = "/data/data1/einsein_data/mc_data"
    single_particle_path = os.path.join(mc_base, "single_particle")
    mip_multiple_path = os.path.join(mc_base, "mip_multiple_particles")
    
    # Single Particle Analysis
    print("1. SINGLE PARTICLE CALIBRATION DATA")
    print("   Location:", single_particle_path)
    print("   Description: Single particle calibration data for ECAL")
    
    single_outtrigd_count = count_root_files(single_particle_path, "OutTrigd_*.root")
    print(f"   OutTrigd_*.root files: {single_outtrigd_count}")
    
    single_flux_path = os.path.join(single_particle_path, "fluxFiles")
    single_flux_count = count_root_files(single_flux_path, "flux_*.root")
    print(f"   flux_*.root files: {single_flux_count}")
    print()
    
    # MIP Multiple Particles Analysis
    print("2. MIP MULTIPLE PARTICLE DATA")
    print("   Location:", mip_multiple_path)
    print("   Description: MIP (Minimum Ionizing Particle) multiple particle simulation")
    print("   Purpose: Multiple hit simulation on ECAL detector")
    
    mip_outtrigd_count = count_root_files(mip_multiple_path, "OutTrigd_*.root")
    print(f"   OutTrigd_*.root files: {mip_outtrigd_count}")
    
    mip_flux_path = os.path.join(mip_multiple_path, "fluxFiles")
    mip_flux_count = count_root_files(mip_flux_path, "flux_*.root")
    print(f"   flux_*.root files: {mip_flux_count}")
    print()
    
    return {
        'single_particle': {
            'outtrigd_count': single_outtrigd_count,
            'flux_count': single_flux_count
        },
        'mip_multiple': {
            'outtrigd_count': mip_outtrigd_count,
            'flux_count': mip_flux_count
        }
    }


def analyze_recon_data():
    """Analyze reconstructed data."""
    print("=== RECONSTRUCTED DATA ANALYSIS ===")
    print()
    
    recon_base = "/data/data1/einsein_data/recon_data"
    
    # 3x3 Algorithm Analysis
    print("1. 3x3 CLUSTERING ALGORITHM")
    print("   Location:", os.path.join(recon_base, "3x3"))
    print("   Description: Heuristic-based clustering algorithm")
    print("   Method: 3x3 grid clustering for hit reconstruction on ECAL")
    
    recon_3x3_path = os.path.join(recon_base, "3x3")
    recon_3x3_count = count_root_files(recon_3x3_path, "*.root")
    print(f"   Reconstruction files: {recon_3x3_count}")
    print()
    
    # OPT Algorithm Analysis
    print("2. OPTIMAL (OPT) CLUSTERING ALGORITHM")
    print("   Location:", os.path.join(recon_base, "opt"))
    print("   Description: Optimal heuristic-based clustering algorithm")
    print("   Method: Advanced clustering optimization for ECAL hit reconstruction")
    
    recon_opt_path = os.path.join(recon_base, "opt")
    recon_opt_count = count_root_files(recon_opt_path, "*.root")
    print(f"   Reconstruction files: {recon_opt_count}")
    print()
    
    return {
        '3x3_count': recon_3x3_count,
        'opt_count': recon_opt_count
    }


def show_data_summary():
    """Main function to display comprehensive data summary."""
    print("EINSTEIN PROJECT - DATA SCHEME SUMMARY")
    print("=" * 50)
    print("ECAL Electromagnetic Calorimeter Data Analysis")
    print("LHCb Detector at CERN")
    print("=" * 50)
    print()
    
    # Analyze MC data
    mc_stats = analyze_mc_data()
    
    # Analyze reconstructed data
    recon_stats = analyze_recon_data()
    
    # Summary statistics
    print("=== SUMMARY STATISTICS ===")
    print(f"Total MC OutTrigd files: {mc_stats['single_particle']['outtrigd_count'] + mc_stats['mip_multiple']['outtrigd_count']}")
    print(f"Total MC Flux files: {mc_stats['single_particle']['flux_count'] + mc_stats['mip_multiple']['flux_count']}")
    print(f"Total Reconstruction files: {recon_stats['3x3_count'] + recon_stats['opt_count']}")
    print()
    
    print("Data collection completed successfully!")


def main():
    """Main entry point."""
    try:
        show_data_summary()
    except Exception as e:
        print(f"Error during data analysis: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
