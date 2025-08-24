#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_responses.py — sparse, tolerant, debug-friendly (no stats; cache-only)

What's in this version
----------------------
- Keeps ONLY the original response content, stored sparsely (and optional dense):
    * ph_hits: list<struct{idx:int32, val:float32}>  (keep v>0)
    * t_hits : list<struct{idx:int32, val:float32}>  (keep v>=time_valid_min)
    * (optional) ph, t as dense lists<float32> when keep_dense_columns=True in config
- NO summary stats are computed (no pe_sum/pe_count/t_min/t_mean).
- Tolerant to truth/response mismatches (WARN only).
- Index-only mapping: sample_key = f"{pair_idx}:{entry_index}".
- Simple prefilter: read ALL *_ph/*_t at entry 0 and keep only non-empty branches/modules.
- Parallel-safe with -j workers.

Debug helpers
-------------
--limit N         : process only the first N files (after natural sort)
--sample N        : randomly sample N files (with --seed)
--sample-frac F   : randomly sample fraction F (0<F<=1) of files
--seed S          : RNG seed for sampling (default 123)
--dry-run         : print selected files and exit
--out-dir DIR     : output directory for parquet parts (default: data/processed/responses)

Examples
--------
# Dry-run first 50 files
python scripts/convert_input/convert_responses.py --limit 50 --dry-run

# Sample 200 files to a debug folder with 32 workers
python scripts/convert_input/convert_responses.py --sample 200 --seed 7 -j 32 --out-dir data/processed/responses_debug
"""

import os
import re
import glob
import yaml
import math
import argparse
import random
import awkward as ak
import uproot
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# -------------------- utils --------------------

def natural_key(s: str):
    """Natural sorting: OutTrigd_2 before OutTrigd_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_sparse_hits(values, *, positive_only=True, valid_min=None):
    """
    Dense Python list -> sparse list of {idx:int, val:float}.
      - If positive_only=True  : keep v > 0
      - Else with valid_min set: keep v >= valid_min
    """
    out = []
    if not values:
        return out
    for i, v in enumerate(values):
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if positive_only:
            if fv > 0.0:
                out.append({"idx": int(i), "val": fv})
        else:
            if (valid_min is not None) and (fv >= valid_min):
                out.append({"idx": int(i), "val": fv})
    return out


def build_schema(sparse_hits: bool, keep_dense: bool):
    """
    Build Arrow schema WITHOUT any stats columns.
    """
    if sparse_hits:
        hit_struct = pa.struct([("idx", pa.int32()), ("val", pa.float32())])
        fields = [
            pa.field("pair_idx", pa.int32()),
            pa.field("entry_index", pa.int64()),
            pa.field("sample_key", pa.string()),
            pa.field("module_idx", pa.int32()),
            pa.field("ph_hits", pa.list_(hit_struct)),
            pa.field("t_hits",  pa.list_(hit_struct)),
        ]
        if keep_dense:
            fields += [
                pa.field("ph", pa.list_(pa.float32())),
                pa.field("t",  pa.list_(pa.float32())),
            ]
        return pa.schema(fields)
    else:
        # dense only
        return pa.schema([
            pa.field("pair_idx", pa.int32()),
            pa.field("entry_index", pa.int64()),
            pa.field("sample_key", pa.string()),
            pa.field("module_idx", pa.int32()),
            pa.field("ph", pa.list_(pa.float32())),
            pa.field("t",  pa.list_(pa.float32())),
        ])


# -------------------- per-file worker --------------------

def process_one(pair_idx: int,
                fpath: str,
                truth_path: str | None,
                cfg_all: dict,
                out_dir: str,
                sparse_hits: bool,
                keep_dense: bool) -> str | None:
    """Process a single OutTrigd_*.root into one parquet part (cache-only; no stats)."""
    cfg = cfg_all["responses"]
    tree_name = cfg["tree_name"]
    time_valid_min = float(cfg.get("time_valid_min", 0.0))

    # open ROOT, get tree
    try:
        fs = uproot.open(fpath)
    except Exception as e:
        print(f"[WARN] Failed to open {fpath}: {e}. Skipping.")
        return None
    if tree_name not in fs:
        print(f"[WARN] Tree '{tree_name}' not in {fpath}. Skipping file.")
        return None
    tree = fs[tree_name]

    # candidate branches (endswith _ph or _t)
    all_branches = list(tree.keys())
    cand_names = [b for b in all_branches if b.endswith("_ph") or b.endswith("_t")]
    if not cand_names:
        return None

    # entry count (expect 1; tolerate otherwise)
    n_resp = tree.num_entries
    if n_resp == 0:
        return None
    if n_resp != 1:
        print(f"[WARN] Expected 1 entry but got {n_resp} in {fpath}. Proceeding.")

    # tolerant check with truth (WARN only)
    if truth_path is not None:
        try:
            with uproot.open(truth_path)[cfg_all["truth"]["tree_name"]] as ttruth:
                n_truth = ttruth.num_entries
            if n_truth != n_resp:
                print(f"[WARN] Entries mismatch at pair_idx={pair_idx}: responses={n_resp} vs truth={n_truth}. Proceeding.")
        except Exception as e:
            print(f"[WARN] Could not open/inspect truth file for pair_idx={pair_idx}: {e}")

    # ---------- Prefilter: read ALL cand branches for entry 0; keep non-empty vectors ----------
    try:
        first_entry = tree.arrays(cand_names, entry_start=0, entry_stop=1, library="ak")
    except Exception as e:
        print(f"[WARN] Prefilter read failed in {fpath}: {e}. Skipping file.")
        return None

    nonempty_branches = []
    for name in cand_names:
        try:
            val0 = first_entry[name][0]  # list at entry 0
            if len(ak.to_list(val0)) > 0:
                nonempty_branches.append(name)
        except Exception:
            # conservative: keep if unsure
            nonempty_branches.append(name)

    if not nonempty_branches:
        return None

    # derive kept module ids (if any ph/t exists)
    kept_mod_ids = sorted({
        int(m.group(1))
        for name in nonempty_branches
        for m in [re.match(r"^mod(\d+)_(?:ph|t)$", name)]
        if m
    })
    if not kept_mod_ids:
        return None

    # data for writing:
    #  - if n_resp == 1, reuse prefilter arrays to avoid second I/O
    #  - else, load all nonempty branches for all entries
    if n_resp == 1:
        data = first_entry
    else:
        data = tree.arrays(nonempty_branches, entry_start=0, entry_stop=n_resp, library="ak")

    # ---------- build Arrow table (per-file) ----------
    schema = build_schema(sparse_hits=sparse_hits, keep_dense=keep_dense)
    out_rows = {name: [] for name in schema.names}
    data_fields = set(data.fields)

    for entry_index in range(n_resp):
        sample_key = f"{pair_idx}:{entry_index}"
        for mid in kept_mod_ids:
            ph_br = f"mod{mid}_ph"
            t_br  = f"mod{mid}_t"
            ph_i = ak.to_list(data[ph_br][entry_index]) if ph_br in data_fields else []
            t_i  = ak.to_list(data[t_br][entry_index])  if t_br  in data_fields else []

            out_rows["pair_idx"].append(pair_idx)
            out_rows["entry_index"].append(entry_index)
            out_rows["sample_key"].append(sample_key)
            out_rows["module_idx"].append(mid)

            if "ph_hits" in out_rows:
                out_rows["ph_hits"].append(to_sparse_hits(ph_i, positive_only=True))
                out_rows["t_hits"].append(to_sparse_hits(t_i, positive_only=False, valid_min=time_valid_min))
                if "ph" in out_rows:  # keep_dense requested
                    out_rows["ph"].append([float(x) for x in ph_i])
                    out_rows["t"].append([float(x) for x in t_i])
            else:
                out_rows["ph"].append([float(x) for x in ph_i])
                out_rows["t"].append([float(x) for x in t_i])

    # cast types precisely to match schema (pa.Table will enforce, but be explicit)
    out_rows["pair_idx"]    = [int(x) for x in out_rows["pair_idx"]]
    out_rows["entry_index"] = [int(x) for x in out_rows["entry_index"]]
    out_rows["module_idx"]  = [int(x) for x in out_rows["module_idx"]]

    table = pa.table(out_rows, schema=schema)

    # write one part per file (parallel-friendly)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"part-{pair_idx:07d}.parquet")
    pq.write_table(table, out_path)
    return out_path


def process_one_star(args):
    return process_one(*args)


# -------------------- driver --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel workers (files processed concurrently).")
    parser.add_argument("--no-hits", action="store_true", help="Disable per-cell sparse hits (keep only dense ph/t).")
    parser.add_argument("--out-dir", type=str, default="data/processed/responses", help="Output directory for parquet parts.")
    # ---- DEBUG options ----
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, default=None, help="Process only the first N response files.")
    group.add_argument("--sample", type=int, default=None, help="Randomly sample N response files (uses --seed).")
    group.add_argument("--sample-frac", type=float, default=None, help="Randomly sample a fraction (0<F<=1) of response files.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for --sample / --sample-frac.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected files and exit without processing.")
    args = parser.parse_args()

    # load config
    with open("scripts/convert_input/config.yaml", "r") as f:
        cfg_all = yaml.safe_load(f)

    resp_files_full = sorted(glob.glob(cfg_all["responses"]["file_glob"]), key=natural_key)
    if not resp_files_full:
        raise FileNotFoundError(f"No ROOT files match: {cfg_all['responses']['file_glob']}")

    truth_files_full = sorted(glob.glob(cfg_all["truth"]["file_glob"]), key=natural_key)
    if truth_files_full and (len(truth_files_full) != len(resp_files_full)):
        print(f"[WARN] File count mismatch: truth={len(truth_files_full)} vs responses={len(resp_files_full)}. Proceeding anyway.")

    # ---- Select subset for DEBUG if requested ----
    total = len(resp_files_full)
    indices = list(range(total))

    if args.limit is not None:
        n = max(0, min(args.limit, total))
        indices = indices[:n]
    elif args.sample is not None:
        n = max(0, min(args.sample, total))
        rnd = random.Random(args.seed)
        indices = sorted(rnd.sample(indices, n)) if n > 0 else []
    elif args.sample_frac is not None:
        if not (0.0 < args.sample_frac <= 1.0):
            raise ValueError("--sample-frac must be in (0, 1].")
        n = max(1, min(total, math.ceil(total * float(args.sample_frac))))
        rnd = random.Random(args.seed)
        indices = sorted(rnd.sample(indices, n))

    selected = [(i, resp_files_full[i]) for i in indices] if indices else list(enumerate(resp_files_full))

    if args.dry_run:
        print(f"[DRY-RUN] Selected {len(selected)} / {total} response files")
        for i, (idx, path) in enumerate(selected[:20]):
            print(f"  [{i:02d}] pair_idx={idx:07d}  {path}")
        if len(selected) > 20:
            print(f"  ... (showing 20 of {len(selected)})")
        return

    out_dir = args.out_dir
    ensure_dir(out_dir)

    sparse_hits = not args.no_hits
    keep_dense = cfg_all.get("responses", {}).get("sparse", {}).get("keep_dense_columns", False)

    # build tasks
    tasks = []
    for idx, fpath in selected:
        truth_path = truth_files_full[idx] if idx < len(truth_files_full) else None
        tasks.append((idx, fpath, truth_path, cfg_all, out_dir, sparse_hits, keep_dense))

    results = []
    if args.jobs and args.jobs > 1:
        from multiprocessing import Pool
        with Pool(processes=args.jobs) as pool:
            for res in tqdm(pool.imap_unordered(process_one_star, tasks), total=len(tasks), desc="convert(parallel)"):
                results.append(res)
    else:
        for t in tqdm(tasks, desc="convert(serial)"):
            results.append(process_one_star(t))

    written = sum(1 for r in results if r)
    skipped = len(tasks) - written
    print(f"[DONE] Wrote {written} parquet parts to {out_dir} (skipped {skipped} files with no non-empty branches).")
    if args.limit or args.sample or args.sample_frac:
        print("[NOTE] DEBUG subset was used; run without --limit/--sample to process all files.")


if __name__ == "__main__":
    main()
