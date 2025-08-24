#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_responses.py — sparse, tolerant, debug-friendly (cache-only)
Optimized prefilter (PH-only + positive-only), no extra CPU threads.

What’s in this version
----------------------
(1) Prefilter ONLY on *_ph branches (NOT *_t), and default to a stricter non-emptiness:
    - Keep a module iff ak.any(modX_ph > 0, axis=1)[0] is True (positive-only).
      This typically reduces kept modules further than just "length > 0".
    - Fallback to length>0 if comparison fails for any branch (be conservative).
(2) No stats: ONLY cache original response content (sparse + optional dense).
(3) No extra CPU resources: we DO NOT add uproot decompression workers.
(4) Parallel-safe: one parquet per input file; -j N for multi-process.
(5) Debug subset + Timing:
    --limit / --sample / --sample-frac / --seed / --dry-run
    --timing / --timing-dir  → per-file JSON + aggregate summary (_summary.json)

Schema (sparse mode; default)
-----------------------------
pair_idx: int32
entry_index: int64
sample_key: string        (= f"{pair_idx}:{entry_index}")
module_idx: int32
ph_hits: list<struct<idx:int32, val:float32>>   (keep v>0)
t_hits : list<struct<idx:int32, val:float32>>   (keep v>=time_valid_min)

If keep_dense_columns=True in config, also write:
  ph: list<float32>, t: list<float32>

Usage examples
--------------
# Dry-run first 50 files
python scripts/convert_input/convert_responses.py --limit 50 --dry-run

# Sample 200 files into a debug dir with 32 workers and timing
python scripts/convert_input/convert_responses.py \
  --sample 200 --seed 7 -j 32 --out-dir data/processed/responses_debug --timing

# Full run (same behavior as before, but faster prefilter)
python scripts/convert_input/convert_responses.py -j 96
"""

import os
import re
import glob
import yaml
import math
import argparse
import random
import time
import json
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

def _tick():
    return time.perf_counter()

def _took(t0):
    return round((time.perf_counter() - t0) * 1000.0, 3)  # ms

TIMING_STEPS = [
    "open_root",
    "list_branches",
    "prefilter_read",
    "prefilter_eval",
    "data_read",
    "build_table",
    "write_parquet",
]


# -------------------- per-file worker --------------------

def process_one(pair_idx: int,
                fpath: str,
                truth_path: str | None,
                cfg_all: dict,
                out_dir: str,
                sparse_hits: bool,
                keep_dense: bool,
                enable_timing: bool,
                timing_dir: str | None):
    """
    Process one ROOT file -> one parquet part (cache-only, PH-only prefilter).
    Returns (out_path_or_None, metrics_dict).
    """
    metrics = {
        "pair_idx": pair_idx,
        "file": fpath,
        "n_resp": 0,
        "n_cand_branches": 0,
        "n_kept_mods": 0,
        "wrote": False,
        **{k: None for k in TIMING_STEPS},
    }

    cfg = cfg_all["responses"]
    tree_name = cfg["tree_name"]
    time_valid_min = float(cfg.get("time_valid_min", 0.0))
    mod_ph_regex = re.compile(r"^mod(\d+)_ph$")

    # open ROOT
    t0 = _tick()
    try:
        fs = uproot.open(fpath)
    except Exception as e:
        if enable_timing: metrics["open_root"] = _took(t0)
        print(f"[WARN] Failed to open {fpath}: {e}. Skipping.")
        _dump_timing(metrics, timing_dir, out_dir, enable_timing)
        return (None, metrics)
    if enable_timing: metrics["open_root"] = _took(t0)

    if tree_name not in fs:
        print(f"[WARN] Tree '{tree_name}' not in {fpath}. Skipping file.")
        _dump_timing(metrics, timing_dir, out_dir, enable_timing)
        return (None, metrics)
    tree = fs[tree_name]

    # list branches
    t0 = _tick()
    all_branches = list(tree.keys())
    # Prefilter only *_ph, not *_t (reduces I/O a lot)
    ph_names = [b for b in all_branches if b.endswith("_ph")]
    metrics["n_cand_branches"] = len(ph_names)
    if enable_timing: metrics["list_branches"] = _took(t0)
    if not ph_names:
        _dump_timing(metrics, timing_dir, out_dir, enable_timing)
        return (None, metrics)

    # entries (expect 1; tolerate otherwise)
    n_resp = tree.num_entries
    metrics["n_resp"] = int(n_resp)
    if n_resp == 0:
        _dump_timing(metrics, timing_dir, out_dir, enable_timing)
        return (None, metrics)
    if n_resp != 1:
        print(f"[WARN] Expected 1 entry but got {n_resp} in {fpath}. Proceeding.")

    # prefilter read (ALL *_ph at entry 0)
    t0 = _tick()
    try:
        first_entry_ph = tree.arrays(ph_names, entry_start=0, entry_stop=1, library="ak")
    except Exception as e:
        if enable_timing: metrics["prefilter_read"] = _took(t0)
        print(f"[WARN] Prefilter read failed in {fpath}: {e}. Skipping file.")
        _dump_timing(metrics, timing_dir, out_dir, enable_timing)
        return (None, metrics)
    if enable_timing: metrics["prefilter_read"] = _took(t0)

    # prefilter eval:
    #   Preferred: keep module if ANY(ph > 0) at entry 0 (strict, typical speed gain later)
    #   Fallback : keep if LENGTH(entry0) > 0 (conservative)
    t0 = _tick()
    kept_mod_ids = []
    for name in ph_names:
        try:
            anypos = ak.any(first_entry_ph[name] > 0, axis=1)[0]
            if bool(anypos):
                m = mod_ph_regex.match(name)
                if m:
                    kept_mod_ids.append(int(m.group(1)))
        except Exception:
            # fallback to length>0
            try:
                cnt = ak.num(first_entry_ph[name], axis=1)[0]
                if cnt > 0:
                    m = mod_ph_regex.match(name)
                    if m:
                        kept_mod_ids.append(int(m.group(1)))
            except Exception:
                # conservative: keep if unsure
                m = mod_ph_regex.match(name)
                if m:
                    kept_mod_ids.append(int(m.group(1)))
    kept_mod_ids.sort()
    metrics["n_kept_mods"] = len(kept_mod_ids)
    if enable_timing: metrics["prefilter_eval"] = _took(t0)
    if not kept_mod_ids:
        _dump_timing(metrics, timing_dir, out_dir, enable_timing)
        return (None, metrics)

    # data read:
    # - For ph: reuse first_entry_ph when n_resp==1; otherwise read kept ph branches for all entries.
    # - For t : read only kept t branches (if they exist in the file).
    t0 = _tick()
    if n_resp == 1:
        data_ph = first_entry_ph
    else:
        read_ph = [f"mod{mid}_ph" for mid in kept_mod_ids]
        data_ph = tree.arrays(read_ph, entry_start=0, entry_stop=n_resp, library="ak")

    kept_t = [f"mod{mid}_t" for mid in kept_mod_ids if f"mod{mid}_t" in all_branches]
    if kept_t:
        data_t = tree.arrays(kept_t, entry_start=0, entry_stop=n_resp, library="ak")
    else:
        # empty record with no fields
        data_t = ak.Array({})
    if enable_timing: metrics["data_read"] = _took(t0)

    # ---------- build Arrow table (per-file; cache-only) ----------
    t0 = _tick()
    schema = build_schema(sparse_hits=sparse_hits, keep_dense=keep_dense)
    out_rows = {name: [] for name in schema.names}
    fields_ph = set(data_ph.fields)
    fields_t  = set(data_t.fields)

    for entry_index in range(n_resp):
        sample_key = f"{pair_idx}:{entry_index}"
        for mid in kept_mod_ids:
            ph_br = f"mod{mid}_ph"
            t_br  = f"mod{mid}_t"

            # Extract lists (convert only here)
            ph_i = ak.to_list(data_ph[ph_br][entry_index]) if ph_br in fields_ph else []
            t_i  = ak.to_list(data_t[t_br][entry_index])   if t_br  in fields_t  else []

            out_rows["pair_idx"].append(pair_idx)
            out_rows["entry_index"].append(entry_index)
            out_rows["sample_key"].append(sample_key)
            out_rows["module_idx"].append(mid)

            if "ph_hits" in out_rows:
                out_rows["ph_hits"].append(to_sparse_hits(ph_i, positive_only=True))
                out_rows["t_hits"].append(to_sparse_hits(t_i, positive_only=False, valid_min=float(time_valid_min)))
                if "ph" in out_rows:  # keep_dense requested
                    out_rows["ph"].append([float(x) for x in ph_i])
                    out_rows["t"].append([float(x) for x in t_i])
            else:
                out_rows["ph"].append([float(x) for x in ph_i])
                out_rows["t"].append([float(x) for x in t_i])

    # enforce dtypes for scalar columns (paranoid but cheap)
    out_rows["pair_idx"]    = [int(x) for x in out_rows["pair_idx"]]
    out_rows["entry_index"] = [int(x) for x in out_rows["entry_index"]]
    out_rows["module_idx"]  = [int(x) for x in out_rows["module_idx"]]

    table = pa.table(out_rows, schema=schema)
    if enable_timing: metrics["build_table"] = _took(t0)

    # write parquet
    t0 = _tick()
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"part-{pair_idx:07d}.parquet")
    pq.write_table(table, out_path)
    metrics["wrote"] = True
    if enable_timing: metrics["write_parquet"] = _took(t0)

    _dump_timing(metrics, timing_dir, out_dir, enable_timing)
    return (out_path, metrics)


def _dump_timing(metrics: dict, timing_dir: str | None, out_dir: str, enable: bool):
    if not enable:
        return
    if timing_dir is None:
        timing_dir = os.path.join(out_dir, "_timings")
    ensure_dir(timing_dir)
    pair_idx = metrics.get("pair_idx", -1)
    path = os.path.join(timing_dir, f"part-{int(pair_idx):07d}.timing.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write timing json for pair_idx={pair_idx}: {e}")


def _aggregate_and_write_summary(all_metrics, timing_dir: str):
    """Compute average per-step timing & write _summary.json; also print a compact table."""
    ensure_dir(timing_dir)
    summary = {
        "files_seen": len(all_metrics),
        "files_wrote": int(sum(1 for m in all_metrics if m.get("wrote"))),
        "avg_n_resp": _avg([m.get("n_resp") for m in all_metrics if m.get("n_resp") is not None]),
        "avg_n_cand_branches": _avg([m.get("n_cand_branches") for m in all_metrics if m.get("n_cand_branches") is not None]),
        "avg_n_kept_mods": _avg([m.get("n_kept_mods") for m in all_metrics if m.get("n_kept_mods") is not None]),
        "steps": {},
    }
    # step-wise averages
    for step in TIMING_STEPS:
        vals = [m.get(step) for m in all_metrics if m.get(step) is not None]
        summary["steps"][step] = {
            "count": len(vals),
            "avg_ms": _avg(vals),
        }

    # write JSON
    out_path = os.path.join(timing_dir, "_summary.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write timing summary: {e}")

    # pretty print
    print("\n[TIMING SUMMARY]")
    print(f"  files_seen = {summary['files_seen']}, files_wrote = {summary['files_wrote']}")
    print(f"  avg_n_resp = {summary['avg_n_resp']:.3f}, avg_n_cand_branches = {summary['avg_n_cand_branches']:.3f}, avg_n_kept_mods = {summary['avg_n_kept_mods']:.3f}")
    print("  per-step average (ms):")
    for step in TIMING_STEPS:
        s = summary["steps"][step]
        avg_ms = f"{s['avg_ms']:.3f}" if s["count"] else "n/a"
        print(f"    - {step:16s}  avg={avg_ms:>8}  (n={s['count']})")
    print(f"  summary json: {out_path}\n")


def _avg(vals):
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def process_one_star(args):
    return process_one(*args)


# -------------------- driver --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel workers (files processed concurrently).")
    parser.add_argument("--no-hits", action="store_true", help="Disable per-cell sparse hits (keep only dense ph/t).")
    parser.add_argument("--out-dir", type=str, default="data/processed/responses", help="Output directory for parquet parts.")
    # DEBUG subset
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, default=None, help="Process only the first N response files.")
    group.add_argument("--sample", type=int, default=None, help="Randomly sample N response files (uses --seed).")
    group.add_argument("--sample-frac", type=float, default=None, help="Randomly sample a fraction (0<F<=1) of response files.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for --sample / --sample-frac.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected files and exit without processing.")
    # TIMING
    parser.add_argument("--timing", action="store_true", help="Record per-file timing JSONs and print an aggregate summary.")
    parser.add_argument("--timing-dir", type=str, default=None, help="Directory to write timing JSONs (default: <out-dir>/_timings).")
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

    # subset selection
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
        tasks.append((idx, fpath, truth_path, cfg_all, out_dir, sparse_hits, keep_dense, bool(args.timing), args.timing_dir))

    results = []
    if args.jobs and args.jobs > 1:
        from multiprocessing import Pool
        with Pool(processes=args.jobs) as pool:
            for res in tqdm(pool.imap_unordered(process_one_star, tasks), total=len(tasks), desc="convert(parallel)"):
                results.append(res)
    else:
        for t in tqdm(tasks, desc="convert(serial)"):
            results.append(process_one_star(t))

    # unpack results
    written = sum(1 for (out_path, _) in results if out_path)
    skipped = len(tasks) - written
    print(f"[DONE] Wrote {written} parquet parts to {out_dir} (skipped {skipped} files with no non-empty branches).")
    if args.limit or args.sample or args.sample_frac:
        print("[NOTE] DEBUG subset was used; run without --limit/--sample to process all files.")

    # aggregate timing & print summary
    if args.timing:
        all_metrics = [m for (_, m) in results if isinstance(m, dict)]
        timing_dir = args.timing_dir or os.path.join(out_dir, "_timings")
        _aggregate_and_write_summary(all_metrics, timing_dir)


if __name__ == "__main__":
    main()
