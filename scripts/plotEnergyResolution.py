#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
energy_resolution_multi.py
- 从一个 YAML 配置中读取 >=1 个数据集（datasets），共享 binning 与坐标轴设置，
  在同一张图上绘制多条相对能量分辨率曲线。
- 每个数据集可单独配置 inputs / name / scales / method 等。
- 计算相对能量分辨率：
    delta = 1 - e/true_eTot
  每个 trueE bin 内用 rms90/std/gauss_fit 得到 sigma(delta)
- 依赖：uproot, numpy, matplotlib, pyyaml（可选 scipy）
  pip install uproot numpy matplotlib pyyaml scipy
"""

import argparse
import glob as _glob
import math
import os
import re
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import uproot
import yaml

# Optional SciPy for Gaussian fit
try:
    from scipy.optimize import curve_fit
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    warnings.warn("SciPy 未安装：method=gauss_fit 会回退为 std。")

# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="从 YAML 配置计算并绘制多曲线相对能量分辨率。")
    ap.add_argument("config", help="YAML 配置文件路径")
    return ap.parse_args()

# ---------------- YAML & Inputs ----------------

def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # ------ 顶层默认（可被 datasets 内覆写） ------
    cfg.setdefault("tree", "cluster")
    cfg.setdefault("branch_true", "true_eTot")
    cfg.setdefault("true_scale", 1.0)
    cfg.setdefault("branch_e", "e")
    cfg.setdefault("rec_scale", 1.0)
    cfg.setdefault("method", "rms90")
    cfg.setdefault("min_per_bin", 50)
    cfg.setdefault("nboot", 0)
    cfg.setdefault("boot_frac", 1.0)
    cfg.setdefault("nbins_hist", 60)
    cfg.setdefault("seed", 42)

    # 数据集列表（至少两个曲线时填写多个；也可只放一个）
    if "datasets" not in cfg or not cfg["datasets"]:
        raise ValueError("YAML 必须包含 datasets（>=1）。")

    # 校验每个数据集
    for i, ds in enumerate(cfg["datasets"]):
        if "name" not in ds or not ds["name"]:
            ds["name"] = f"dataset_{i+1}"
        if "inputs" not in ds or not ds["inputs"]:
            raise ValueError(f"datasets[{i}] 缺少 inputs。")
        # 给每个数据集填充默认值
        for k in ("tree", "branch_true", "true_scale", "branch_e", "rec_scale",
                  "method", "min_per_bin", "nboot", "boot_frac", "nbins_hist", "seed"):
            ds.setdefault(k, cfg.get(k))

    # binning（共享）
    if "binning" not in cfg:
        cfg["binning"] = {}
    binning = cfg["binning"]
    if "edges" not in binning:
        for k in ("nbins", "emin", "emax"):
            if k not in binning:
                raise ValueError("YAML 缺少 binning.edges 或 (nbins/emin/emax)。")

    # 输出（共享）
    if "outputs" not in cfg:
        cfg["outputs"] = {}
    outs = cfg["outputs"]
    outs.setdefault("out", "energy_resolution_multi.png")
    outs.setdefault("csv", "energy_resolution_multi.csv")  # 合并 CSV（含 dataset 列）
    outs.setdefault("csv_per_dataset", False)              # 可选：是否为每个数据集单独导出
    outs.setdefault("csv_pattern", "energy_resolution_{name}.csv")
    outs.setdefault("title", "Relative Energy Resolution vs True Energy")
    outs.setdefault("xlabel", "True energy")
    outs.setdefault("ylabel", "Relative energy resolution")
    outs.setdefault("percent", False)
    outs.setdefault("show", False)

    return cfg

def expand_inputs(patterns):
    """
    扩展 YAML 里的 inputs：
      - 普通字符串 => 作为 glob 模式用 glob.glob(..., recursive=True)
      - 以 're:' 开头 => 作为 regex，在当前工作目录递归 os.walk()，
        对完整路径（用 POSIX 分隔符）做 re.search 过滤。
    返回去重后的文件列表（按路径排序）。
    """
    files = []
    need_walk = any(is_regex(p) for p in patterns)
    walk_paths = []
    if need_walk:
        cwd = os.getcwd()
        for root, _, fnames in os.walk(cwd):
            for name in fnames:
                full = os.path.join(root, name)
                walk_paths.append(full.replace(os.sep, "/"))

    for p in patterns:
        if is_regex(p):
            regex_str = p[len("re:"):]
            try:
                rx = re.compile(regex_str)
            except re.error as e:
                raise ValueError(f"无效正则：{regex_str}  ({e})")
            matched = [x for x in walk_paths if rx.search(x)]
            files.extend(matched)
        else:
            matched = _glob.glob(p, recursive=True)
            matched = [m.replace(os.sep, "/") for m in matched]
            files.extend(matched)

    uniq = sorted({f for f in files if os.path.isfile(f)})
    if not uniq:
        raise FileNotFoundError("未匹配到任何输入 ROOT 文件，请检查 inputs。")
    return uniq

def is_regex(p):
    return isinstance(p, str) and p.startswith("re:")

# ----------------- Physics helpers -----------------

def load_arrays(files, tree, b_true, true_scale, b_e, rec_scale):
    arr_true, arr_e = [], []
    for f in files:
        with uproot.open(f) as rf:
            if tree not in rf:
                raise KeyError(f"{f} 中未找到 TTree '{tree}'")
            t = rf[tree]
            a_true = t[b_true].array(library="np")
            a_e    = t[b_e].array(library="np")
            arr_true.append(a_true)
            arr_e.append(a_e)
    true_e = np.concatenate(arr_true).astype(np.float64) * float(true_scale)
    e_rec  = np.concatenate(arr_e ).astype(np.float64) * float(rec_scale)
    mask = np.isfinite(true_e) & np.isfinite(e_rec) & (true_e > 0)
    return true_e[mask], e_rec[mask]

def rms90(values):
    v = np.sort(np.asarray(values))
    n = len(v)
    if n == 0:
        return np.nan
    k = max(1, int(math.floor(0.9 * n)))
    widths = v[k-1:] - v[:n-k+1]
    idx = int(np.argmin(widths))
    window = v[idx:idx+k]
    m = float(window.mean())
    return float(np.sqrt(np.mean((window - m)**2)))

def gauss_fit_sigma(delta, nbins_hist):
    if not SCIPY_OK:
        return float(np.std(delta, ddof=1)), np.nan
    hist, edges = np.histogram(delta, bins=nbins_hist)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu0 = float(np.mean(delta))
    sigma0 = float(np.std(delta)) if np.std(delta) > 1e-9 else 1e-6
    A0 = float(hist.max()) if hist.size else 1.0

    def gauss(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu)/sigma)**2)

    try:
        popt, pcov = curve_fit(gauss, centers, hist, p0=[A0, mu0, sigma0], maxfev=20000)
        sigma = abs(float(popt[2]))
        sigma_err = float(np.sqrt(pcov[2,2])) if (pcov is not None and pcov.shape == (3,3)) else np.nan
        return sigma, sigma_err
    except Exception:
        return float(np.std(delta, ddof=1)), np.nan

def _bootstrap_err(arr, nboot, frac, reducer, seed=42):
    if nboot and nboot > 0:
        rng = np.random.default_rng(seed=seed)
        m = max(1, int(round(frac * len(arr))))
        vals = []
        for _ in range(nboot):
            samp = rng.choice(arr, size=m, replace=True)
            vals.append(reducer(samp))
        return float(np.std(vals, ddof=1))
    return np.nan

def compute_resolution_per_bin(true_e, e_rec, bin_edges, method, min_per_bin,
                               nbins_hist, nboot=0, boot_frac=1.0, seed=42):
    bin_idx = np.digitize(true_e, bin_edges) - 1
    nbins = len(bin_edges) - 1
    xvals, yvals, yerrs, counts = [], [], [], []

    for i in range(nbins):
        sel = (bin_idx == i)
        if not np.any(sel):
            continue
        te = true_e[sel]
        er = e_rec[sel]
        if te.size < min_per_bin:
            continue

        delta = 1.0 - (er / te)               # 相对误差
        x_rep = float(np.median(te))          # 代表能量：中位数

        if method == "std":
            val = float(np.std(delta, ddof=1))
            err = _bootstrap_err(delta, nboot, boot_frac, reducer=lambda d: np.std(d, ddof=1), seed=seed)

        elif method == "rms90":
            val = rms90(delta)
            err = _bootstrap_err(delta, nboot, boot_frac, reducer=rms90, seed=seed)

        elif method == "gauss_fit":
            val, err = gauss_fit_sigma(delta, nbins_hist)
        else:
            raise ValueError("未知 method")

        xvals.append(x_rep)
        yvals.append(val)
        yerrs.append(err)
        counts.append(te.size)

    order = np.argsort(np.array(xvals))
    return (np.array(xvals)[order],
            np.array(yvals)[order],
            np.array(yerrs)[order],
            np.array(counts)[order])

# ---------------------- Main ----------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ---- 共享 bin 边界 ----
    binning = cfg["binning"]
    if "edges" in binning and binning["edges"] is not None:
        edges = np.array([float(x) for x in binning["edges"]], dtype=float)
        if np.any(np.diff(edges) <= 0):
            raise ValueError("binning.edges 必须严格递增")
    else:
        emin = float(binning["emin"])
        emax = float(binning["emax"])
        nbins = int(binning["nbins"])
        if not (emax > emin):
            raise ValueError("binning.emax 必须大于 binning.emin")
        edges = np.linspace(emin, emax, nbins + 1)

    outs = cfg["outputs"]
    percent = bool(outs.get("percent", False))

    # ---- 遍历各数据集，统一计算并绘图 ----
    all_rows = []  # 用于合并 CSV
    plt.figure(figsize=(7.6,5.4), dpi=140)

    for ds in cfg["datasets"]:
        name = str(ds["name"])

        # 收集输入文件
        files = expand_inputs(ds["inputs"])
        print(f"[info] 数据集 '{name}' 匹配到 {len(files)} 个文件。示例：")
        for s in files[:5]:
            print("   ", s)
        if len(files) > 5:
            print("   ...")

        # 读取数据
        true_e, e_rec = load_arrays(
            files,
            ds["tree"], ds["branch_true"], ds["true_scale"],
            ds["branch_e"], ds["rec_scale"]
        )
        if true_e.size == 0:
            print(f"[warn] 数据集 '{name}' 未读取到有效数据。跳过。", file=sys.stderr)
            continue

        # 计算
        x, y, yerr, n = compute_resolution_per_bin(
            true_e, e_rec, edges, ds["method"], int(ds["min_per_bin"]),
            int(ds["nbins_hist"]), nboot=int(ds["nboot"]), boot_frac=float(ds["boot_frac"]),
            seed=int(ds.get("seed", 42))
        )
        if x.size == 0:
            print(f"[warn] 数据集 '{name}' 所有 bin 被跳过（事件数不足或无有效 bin）。", file=sys.stderr)
            continue

        # 记录到汇总表
        for xi, yi, ei, ni in zip(x, y, yerr, n):
            row = {
                "dataset": name,
                "true_energy_repr": xi,
                "resolution": yi,
                "resolution_err": ei,
                "entries": int(ni)
            }
            if percent:
                row["resolution_percent"] = yi * 100.0
            all_rows.append(row)

        # 绘图（matplotlib 自动分配不同颜色/标记；也可在 YAML 加字段来自定义，需扩展）
        yy = y * (100.0 if percent else 1.0)
        yerr_plot = yerr * (100.0 if percent else 1.0) if np.any(np.isfinite(yerr)) else None
        label = f"{name} ({ds['method'].upper()})"
        if yerr_plot is not None and np.any(np.isfinite(yerr_plot)):
            plt.errorbar(x, yy, yerr=yerr_plot, fmt="o-", capsize=3, label=label)
        else:
            plt.plot(x, yy, "o-", label=label)

        # 可选：单独 CSV
        if outs.get("csv_per_dataset", False):
            csv_path = outs.get("csv_pattern", "energy_resolution_{name}.csv").format(name=sanitize_filename(name))
            with open(csv_path, "w") as f:
                if percent:
                    f.write("dataset,true_energy_repr,resolution,resolution_percent,resolution_err,entries\n")
                    for xi, yi, ei, ni in zip(x, y, yerr, n):
                        f.write(f"{name},{xi},{yi},{yi*100.0},{ei},{int(ni)}\n")
                else:
                    f.write("dataset,true_energy_repr,resolution,resolution_err,entries\n")
                    for xi, yi, ei, ni in zip(x, y, yerr, n):
                        f.write(f"{name},{xi},{yi},{ei},{int(ni)}\n")
            print(f"[保存] 单数据集 CSV -> {csv_path}")

    # 若无任何有效数据集
    if not all_rows:
        print("没有可绘制的数据（请检查 inputs 或放宽 min_per_bin）。", file=sys.stderr)
        sys.exit(2)

    # ---- 合并 CSV 输出 ----
    all_rows_sorted = sorted(all_rows, key=lambda r: (r["dataset"], r["true_energy_repr"]))
    with open(outs["csv"], "w") as f:
        if percent:
            f.write("dataset,true_energy_repr,resolution,resolution_percent,resolution_err,entries\n")
            for r in all_rows_sorted:
                f.write(f"{r['dataset']},{r['true_energy_repr']},{r['resolution']},{r['resolution_percent']},{r['resolution_err']},{r['entries']}\n")
        else:
            f.write("dataset,true_energy_repr,resolution,resolution_err,entries\n")
            for r in all_rows_sorted:
                f.write(f"{r['dataset']},{r['true_energy_repr']},{r['resolution']},{r['resolution_err']},{r['entries']}\n")
    print(f"[保存] 合并 CSV -> {outs['csv']}")

    # ---- 统一绘图属性（共享坐标轴/标题/单位）----
    plt.xlabel(outs["xlabel"])
    plt.ylabel(outs["ylabel"] + (" (%)" if percent else ""))
    plt.title(outs["title"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outs["out"], bbox_inches="tight")
    print(f"[保存] 图像 -> {outs['out']}")

    if outs.get("show", False):
        plt.show()

def sanitize_filename(s):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)

if __name__ == "__main__":
    main()
