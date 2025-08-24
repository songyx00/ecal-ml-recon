#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 flux_*（truth），按自然排序生成 pair_idx，使用 entry_index（0..n-1），
构造 sample_key = "{pair_idx}:{entry_index}"。
不读取任何带 'id' 的分支。
"""
import os
import re
import glob
import yaml
import numpy as np
import pandas as pd
import uproot
from tqdm import tqdm

def natural_key(s: str):
    # 自然排序：flux_2 在 flux_10 前
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def load_cfg():
    with open("scripts/convert_input/config.yaml", "r") as f:
        return yaml.safe_load(f)["truth"]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    cfg = load_cfg()
    files = sorted(glob.glob(cfg["file_glob"]), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No ROOT files match: {cfg['file_glob']}")

    tree_name = cfg["tree_name"]
    branches = cfg["branches"]
    out_dir = "data/processed/truth"
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "events.parquet")

    parts = []
    for pair_idx, fpath in enumerate(tqdm(files, desc="truth files")):
        with uproot.open(fpath)[tree_name] as tree:
            n = tree.num_entries
            arrs = tree.arrays(list(branches.values()), library="np")
            df = pd.DataFrame({k: arrs[v] for k, v in branches.items()})
            df.insert(0, "entry_index", np.arange(n, dtype=np.int64))
            df.insert(0, "pair_idx", np.int32(pair_idx))
            df["sample_key"] = df["pair_idx"].astype(str) + ":" + df["entry_index"].astype(str)
            parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(out_path, index=False)
    print(f"[DONE] truth -> {out_path}, rows={len(out)}")

if __name__ == "__main__":
    main()
