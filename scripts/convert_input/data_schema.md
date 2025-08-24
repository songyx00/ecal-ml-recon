# ECAL-ML Dataset Format Specification 

This document defines the training-ready data format produced by the conversion scripts for the ECAL wall simulation. It covers layout, file formats, schemas, keys, units, and recommended usage.

---

## 1) Directory Layout

```
ECAL_ML_Reconstruction/
  data/
    processed/
      geometry/
        cells.parquet            # static geometry (stored once)
      truth/
        events.parquet           # per-particle ground truth
      responses/
        modules.parquet          # per (event × module) sparse responses
```

> Geometry is identical across runs and is exported **once**.  
> Truth and responses are aligned **by index only** (no “id” branches are used).

---

## 2) Global Conventions

- **File format:** Apache Parquet (Arrow).
- **Keying and alignment**
  - `pair_idx:int32` — position of a file pair in the naturally sorted lists of truth (`flux_*.root`) and response (`OutTrigd_*.root`) files (0-based).
  - `entry_index:int64` — entry index inside a file (0…n-1).
  - `sample_key:string` — textual join key: `"{pair_idx}:{entry_index}"`.
  - **Alignment rule:** `(pair_idx, entry_index)` pairs **must** be one-to-one between `truth` and `responses`.
- **Module identifier in responses:** `module_idx:int32` is parsed from branch names `mod{NNN}_*` (e.g., `mod123_ph → 123`). No *id* branches are read or used.
- **Sparse semantics**
  - Single-particle injections yield mostly empty readouts. Responses are stored sparsely:
    - `ph_hits` keeps only `ph > 0`.
    - `t_hits` keeps only `t >= time_valid_min`.
- **Units (as in your simulation)**
  - Geometry positions/sizes: **mm**.
  - Momentum: as provided (commonly **MeV/c** or **GeV/c**).
  - Energy `eTot`: as provided (commonly **MeV** or **GeV**).
  - Times (`timing`, CFD timestamps `t`): simulation units (commonly **ns**).  
  Record actual units for your run in your experiment notes.

---

## 3) Schemas

### 3.1 Geometry — `geometry/cells.parquet`

Granularity: **one row per cell** (static; stored once).

| Column                  | Type     | Description |
|-------------------------|----------|-------------|
| `module_id_in_calo`     | int32    | Module index within the calorimeter. |
| `cell_id_in_module`     | int32    | Cell index within the module. |
| `module_type`           | int32    | Module type code. |
| `x_mm`,`y_mm`,`z_mm`    | float32  | Cell center position (mm). |
| `dx_mm`,`dy_mm`,`dz_mm` | float32  | Cell dimensions (mm). |
| `ax_deg`,`ay_deg`,`az_deg` | float32 | Cell orientation (deg). |
| `cell_global_id`        | int64    | Convenience global key (e.g., `module * BASE + cell`; `BASE ≥ max cell count`). |

> Geometry is **not** used for truth/response alignment; it supports spatial features and graph construction.

---

### 3.2 Truth — `truth/events.parquet`

Granularity: **one row per injected particle (event)**.

| Column        | Type     | Description |
|---------------|----------|-------------|
| `pair_idx`    | int32    | File-pair index (0-based). |
| `entry_index` | int64    | Entry index within file. |
| `sample_key`  | string   | `"{pair_idx}:{entry_index}"` (join key). |
| `entry_x`     | float32  | Injection x (mm). |
| `entry_y`     | float32  | Injection y (mm). |
| `entry_z`     | float32  | Injection z (mm). |
| `px`          | float32  | Momentum x (simulation units). |
| `py`          | float32  | Momentum y. |
| `pz`          | float32  | Momentum z. |
| `timing`      | float32  | Injection time (simulation units). |
| `eTot`        | float32  | Total energy (simulation units). |

**Constraints**
- `sample_key` is **unique** across the table.
- No columns with “id” from the input are used.

---

### 3.3 Responses — `responses/modules.parquet` (sparse)

Granularity: **one row per (event × module)**.

| Column         | Type                                    | Description |
|----------------|-----------------------------------------|-------------|
| `pair_idx`     | int32                                   | File-pair index. |
| `entry_index`  | int64                                   | Entry index. |
| `sample_key`   | string                                  | Join key to truth. |
| `module_idx`   | int32                                   | Module index parsed from `mod{NNN}_*` branch names. |
| `ph_hits`      | list<struct{ `idx:int32`, `val:float32` }> | Sparse PE hits: only entries with `ph > 0`. |
| `t_hits`       | list<struct{ `idx:int32`, `val:float32` }> | Sparse timestamps: only entries with `t >= time_valid_min`. |
| `pe_sum`       | float32                                 | Sum of all `ph` in this module. |
| `pe_count`     | int32                                   | Count of cells with `ph >= cfd_threshold_pe`. |
| `t_min`        | float32                                 | Minimum valid timestamp or `NaN` if none. |
| `t_mean`       | float32                                 | Mean valid timestamp or `NaN` if none. |

**Threshold parameters (from converter config)**
- `cfd_threshold_pe: float` — defines `pe_count`.
- `time_valid_min: float` — validity cutoff for `t_hits` (e.g., `0.0` to drop negative times).

**Optional dense columns** (present only if enabled during conversion)
- `ph: list<float32>` — dense per-cell PE (many zeros).
- `t:  list<float32>` — dense per-cell timestamps (invalid entries may be negative).

**Constraints**
- `(sample_key, module_idx)` is **unique** across the table.
- `ph_hits.idx` ranges `0..Ncells_in_module-1` and is strictly increasing within a row.
- `t_hits.idx` uses the same indexing as `ph_hits`.

---

## 4) Alignment & Joins

Align `truth` and `responses` **only** via `sample_key`.

```python
import pandas as pd
truth = pd.read_parquet("data/processed/truth/events.parquet")
resp  = pd.read_parquet("data/processed/responses/modules.parquet")

# Example: event-level PE
event_pe = resp.groupby("sample_key")["pe_sum"].sum().rename("pe_total")
df = truth.merge(event_pe, on="sample_key", how="left")
```

---

## 5) Working with Sparse Data

### 5.1 Sparse → dense (padding) for one module
```python
import numpy as np

sub = resp[resp["module_idx"] == 0]  # example: module 0
max_idx = 1 + max((h["idx"] for lst in sub["ph_hits"] for h in lst), default=-1)
dense_ph = np.zeros((len(sub), max_idx), dtype=np.float32)
for i, hits in enumerate(sub["ph_hits"]):
    for h in hits:
        dense_ph[i, h["idx"]] = h["val"]
```

### 5.2 Flatten hits per event (across modules)
```python
def flatten_hits(group):
    # -> list of (module_idx, cell_idx, ph, t_or_None)
    out = []
    for _, row in group.iterrows():
        ph_map = {h["idx"]: h["val"] for h in row["ph_hits"]}
        t_map  = {h["idx"]: h["val"] for h in row["t_hits"]}
        for i, pe in ph_map.items():
            out.append((row["module_idx"], i, pe, t_map.get(i)))
    return out

hits_per_event = resp.groupby("sample_key").apply(flatten_hits)
```

---

## 6) Data Validation Checklist

- **File pairing:** number of `flux_*.root` equals number of `OutTrigd_*.root`.
- **Entry counts:** for each `pair_idx`, entries in truth and responses **match**.
- **Key uniqueness:**  
  - `truth.sample_key` has no duplicates.  
  - `responses` has unique `(sample_key, module_idx)`.
- **Ranges and NaNs:**  
  - `ph >= 0`.  
  - `t_hits` contain only `t >= time_valid_min`; `t_min/t_mean` set to `NaN` if no valid times.
- **Sparse order:** `ph_hits.idx` strictly increasing within each row.

---

## 7) Notes & Extensions (Optional)

- **Cell mapping in responses:** If a branch with per-hit cell indices becomes available, `ph_hits`/`t_hits` can be upgraded to `list<struct{ cell_id_in_module:int32, val:float32 }>` for direct joins with geometry at cell level.
- **Event-graph export:** Provide a graph file (nodes=cells or modules; edges=geometric adjacency) keyed by `cell_global_id` for GNN workflows.
- **NPZ export:** For fixed-shape pipelines, a padded NPZ can be produced (top-K cells per module, top-M modules per event).

---

## 8) Quick Reference (Arrow-like schema)

```text
geometry/cells.parquet
  module_id_in_calo: int32
  cell_id_in_module: int32
  module_type: int32
  x_mm,y_mm,z_mm: float32
  dx_mm,dy_mm,dz_mm: float32
  ax_deg,ay_deg,az_deg: float32
  cell_global_id: int64

truth/events.parquet
  pair_idx: int32
  entry_index: int64
  sample_key: string
  entry_x,entry_y,entry_z: float32
  px,py,pz: float32
  timing: float32
  eTot: float32

responses/modules.parquet (sparse)
  pair_idx: int32
  entry_index: int64
  sample_key: string
  module_idx: int32
  ph_hits: list<struct<idx:int32, val:float32>>
  t_hits:  list<struct<idx:int32, val:float32>>
  pe_sum: float32
  pe_count: int32
  t_min: float32
  t_mean: float32
  # optional (if enabled):
  # ph: list<float32>
  # t:  list<float32>
```
