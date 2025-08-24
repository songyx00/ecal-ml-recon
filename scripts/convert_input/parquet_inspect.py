#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parquet_inspect.py

A tiny helper to quickly inspect ECAL-ML parquet outputs using PyArrow, covering:
1) Single-file schema & metadata
2) Row-group / column statistics
3) Lightweight preview of the first N rows (file or directory)

Update:
- When printing HEAD, show the FULL content with no ellipses ("...").

Usage examples
--------------
# Inspect one file (schema + row-group stats for module_idx + head=5)
python parquet_inspect.py --file data/processed/responses/part-0000001.parquet --col module_idx --head 5

# Inspect a directory as a dataset (directory schema + head=5 across files)
python parquet_inspect.py --dir data/processed/responses --head 5

# Do both in one go
python parquet_inspect.py --file data/processed/responses/part-0000001.parquet --dir data/processed/responses --col module_idx --head 10
"""

import argparse
import numpy as np
from pyarrow import parquet as pq
import pyarrow.dataset as ds
import pandas as pd


def inspect_file_schema(file_path: str):
    print("\n=== [1] Single-file schema & basic metadata ===")
    pf = pq.ParquetFile(file_path)
    print(f"- File: {file_path}")
    print(f"- Num row groups: {pf.metadata.num_row_groups}, Num rows: {pf.metadata.num_rows}")
    print(f"- Created by: {pf.metadata.created_by}")
    print("\n-- Arrow schema --")
    print(pf.schema_arrow)
    print("\n-- Column names --")
    print([f.name for f in pf.schema_arrow])
    return pf


def rowgroup_stats(pf: pq.ParquetFile, col_name: str | None):
    print("\n=== [2] Row-group & column statistics ===")
    if col_name is None:
        preferred = "module_idx"
        if pf.schema_arrow.get_field_index(preferred) >= 0:
            col_name = preferred
        else:
            col_name = pf.schema_arrow.names[0]
            print(f"[INFO] --col not provided. Using first column: {col_name}")

    col_idx = pf.schema_arrow.get_field_index(col_name)
    if col_idx < 0:
        print(f"[WARN] Column '{col_name}' not found. Available columns:")
        print(pf.schema_arrow.names)
        return

    md = pf.metadata
    print(f"- Column: {col_name} (index {col_idx})")
    print(f"- Row groups: {md.num_row_groups}, Total rows: {md.num_rows}")
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        col = rg.column(col_idx)
        stats = col.statistics
        print(f"  RG {i:02d}: rows={rg.num_rows}, bytes={rg.total_byte_size}")
        if stats is not None:
            print(f"    stats: min={getattr(stats, 'min', None)}  max={getattr(stats, 'max', None)}  "
                  f"nulls={getattr(stats, 'null_count', None)}  distinct={getattr(stats, 'distinct_count', None)}")
        else:
            print("    stats: <no statistics available>")


def preview_head(source_path: str, n: int, is_dir: bool):
    print("\n=== [3] Preview first rows (HEAD) — full print, no ellipses ===")
    if n <= 0:
        print("[INFO] --head <= 0; skipping preview.")
        return

    # Ensure *no* ellipses in printed output
    # - Pandas: disable all truncation
    # - Numpy: print full arrays/lists
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)            # auto-detect console width
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_seq_items", None)    # don't truncate long lists
    np.set_printoptions(threshold=np.inf)

    if is_dir:
        print(f"- Dataset: {source_path}")
        dataset = ds.dataset(source_path, format="parquet")
        print("-- Directory-level schema --")
        print(dataset.schema)
        tbl = dataset.head(n)
    else:
        print(f"- File: {source_path}")
        dataset = ds.dataset(source_path, format="parquet")
        print("-- File-level schema (via dataset) --")
        print(dataset.schema)
        tbl = dataset.head(n)

    try:
        df = tbl.to_pandas()
        print(f"\n-- head({n}) --")
        # to_string with no limits to avoid "..."
        print(df.to_string(max_rows=None, max_cols=None))
    except Exception as e:
        print(f"[WARN] Could not render Pandas preview: {e}")
        print(tbl)  # Fallback: Arrow table print (might still truncate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None, help="Path to a single parquet file to inspect.")
    ap.add_argument("--dir", type=str, default=None, help="Path to a directory of parquet parts (dataset).")
    ap.add_argument("--col", type=str, default=None, help="Column name for row-group stats (default: module_idx or first column).")
    ap.add_argument("--head", type=int, default=5, help="Number of rows to preview (default: 5).")
    args = ap.parse_args()

    if not args.file and not args.dir:
        ap.error("Please provide --file and/or --dir")

    # 1) Single-file schema & metadata
    pf = None
    if args.file:
        pf = inspect_file_schema(args.file)

    # 2) Row-group & column statistics
    if pf is not None:
        rowgroup_stats(pf, args.col)

    # 3) Preview head with full content
    if args.dir:
        preview_head(args.dir, args.head, is_dir=True)
    elif args.file:
        preview_head(args.file, args.head, is_dir=False)


if __name__ == "__main__":
    main()
