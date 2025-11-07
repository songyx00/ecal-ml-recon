# DATASCHEME.md

## Einstein Project - ECAL Data Scheme Documentation

### Overview
This document describes the data structure and organization for the Einstein project, which analyzes Monte Carlo simulation data for the ECAL (Electromagnetic Calorimeter) detector at the LHCb experiment at CERN.

---

## 1. Monte Carlo Simulation Data

### Base Location
**Path:** `/data/data1/einsein_data/mc_data/`

The Monte Carlo simulation data consists of two main types of datasets, both generated specifically for ECAL electromagnetic calorimeter analysis at the LHCb detector.

### 1.1 Single Particle Calibration Data

**Location:** `/data/data1/einsein_data/mc_data/single_particle/`

**Description:** Single particle calibration data for ECAL detector calibration and performance studies.

**File Structure:**
- **Main Data Files:** `OutTrigd_*.root` files
  - **Count:** 119,613 files
  - **Purpose:** Each ROOT file represents a single simulation run
  - **Content:** Triggered output data from single particle simulations

- **Flux Files:** `fluxFiles/flux_*.root` files
  - **Count:** 120,000 files
  - **Purpose:** Flux measurement data corresponding to each simulation run
  - **Content:** Particle flux information for calibration purposes

### 1.2 MIP Multiple Particle Data

**Location:** `/data/data1/einsein_data/mc_data/mip_multiple_particles/`

**Description:** MIP (Minimum Ionizing Particle) multiple particle simulation data for multiple hit analysis on ECAL detector.

**File Structure:**
- **Main Data Files:** `OutTrigd_*.root` files
  - **Count:** 10,285 files
  - **Purpose:** Each ROOT file represents a simulation run with multiple particle interactions
  - **Content:** Triggered output data from multiple particle simulations

- **Flux Files:** `fluxFiles/flux_*.root` files
  - **Count:** 10,500 files
  - **Purpose:** Flux measurement data for multiple particle scenarios
  - **Content:** Particle flux information for multi-hit analysis

---

## 2. Reconstructed Data

### Base Location
**Path:** `/data/data1/einsein_data/recon_data/`

The reconstructed data contains processed simulation results using two different clustering algorithms for hit reconstruction on the ECAL detector.

### 2.1 3x3 Clustering Algorithm

**Location:** `/data/data1/einsein_data/recon_data/3x3/`

**Description:** Heuristic-based clustering algorithm using a 3x3 grid approach for hit reconstruction.

**Algorithm Details:**
- **Method:** 3x3 grid clustering for hit reconstruction on ECAL
- **Type:** Heuristic-based algorithm
- **Purpose:** Cluster detection and hit reconstruction using fixed grid patterns

**File Structure:**
- **Reconstruction Files:** `myReconstruction_photon_Run5_rotated_2023_baseline_*.root`
  - **Count:** 20 files
  - **Naming Convention:** `myReconstruction_photon_Run5_rotated_2023_baseline_[range]_seeding0.root`
  - **Content:** Reconstructed hit clusters using 3x3 algorithm

### 2.2 Optimal (OPT) Clustering Algorithm

**Location:** `/data/data1/einsein_data/recon_data/opt/`

**Description:** Optimal heuristic-based clustering algorithm for advanced hit reconstruction.

**Algorithm Details:**
- **Method:** Advanced clustering optimization for ECAL hit reconstruction
- **Type:** Optimal heuristic-based algorithm
- **Purpose:** Enhanced cluster detection with optimized clustering parameters

**File Structure:**
- **Reconstruction Files:** `myReconstruction_photon_Run5_rotated_2023_baseline_*.root`
  - **Count:** 20 files
  - **Naming Convention:** `myReconstruction_photon_Run5_rotated_2023_baseline_[range]_seeding5.root`
  - **Content:** Reconstructed hit clusters using optimal algorithm

---

## 3. Summary Statistics

### Monte Carlo Data Summary
- **Total OutTrigd Files:** 129,898 files
- **Total Flux Files:** 130,500 files
- **Single Particle OutTrigd Files:** 119,613 files
- **Single Particle Flux Files:** 120,000 files
- **MIP Multiple Particle OutTrigd Files:** 10,285 files
- **MIP Multiple Particle Flux Files:** 10,500 files

### Reconstructed Data Summary
- **Total Reconstruction Files:** 40 files
- **3x3 Algorithm Files:** 20 files
- **OPT Algorithm Files:** 20 files

---

## 4. Data Analysis Tools

### Python Analysis Script
**Location:** `/data/data3/work/einstein/tools/show_data_summary.py`

**Purpose:** Automated data analysis and statistics generation for the Einstein project.

**Features:**
- File counting and statistics
- Data structure analysis
- Comprehensive reporting
- PEP 8 compliant code structure

**Usage:**
```bash
python tools/show_data_summary.py
```

---

## 5. Technical Notes

### File Formats
- All data files are in ROOT format (`.root`)
- ROOT is the standard data format used in high-energy physics experiments
- Files contain structured data suitable for analysis with ROOT framework

### Data Organization
- Files are organized hierarchically by data type and algorithm
- Consistent naming conventions facilitate automated processing
- Flux files are stored in dedicated subdirectories for clarity

### Analysis Considerations
- Single particle data is primarily used for calibration purposes
- Multiple particle data enables multi-hit analysis capabilities
- Reconstruction algorithms provide different approaches to hit clustering
- Seeding parameters differ between algorithms (seeding0 vs seeding5)

---

*This document was generated automatically using the Einstein project data analysis tools.*
*Last updated: Generated from live data analysis*
