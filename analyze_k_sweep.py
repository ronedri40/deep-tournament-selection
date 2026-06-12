#!/usr/bin/env python
"""Analyze a k-sweep: how does tournament size k affect each problem/instance?

    python analyze_k_sweep.py [--csv runs/k_sweep/k_sweep_summary.csv] [--selection dts]

Prints, per problem, a pivot of mean best-fitness by instance x k and the best k
per instance, reports whether the best k trends with instance SIZE (small vs large
problems), and writes an aggregated CSV + a "best-fitness vs k" plot per problem.
Everything is framed as maximization, so higher best-fitness is always better.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd


def parse_size(label):
    m = re.search(r"(\d+)", str(label))
    return int(m.group(1)) if m else np.nan


def main():
    p = argparse.ArgumentParser(description="Analyze a tournament-size (k) sweep.")
    p.add_argument("--csv", default=os.path.join("runs", "k_sweep", "k_sweep_summary.csv"))
    p.add_argument("--selection", default="dts", help="selection to analyze (dts/tournament)")
    p.add_argument("--outdir", default="figures_k_sweep")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["selection"] == args.selection].copy()
    if df.empty:
        raise SystemExit(f"no rows for selection={args.selection} in {args.csv}")
    df["size_num"] = df["size"].map(parse_size)

    agg = (
        df.groupby(["problem", "instance", "size_num", "k"])
        .agg(mean_fit=("best_fitness", "mean"),
             std_fit=("best_fitness", "std"),
             n=("best_fitness", "size"))
        .reset_index()
    )
    os.makedirs(args.outdir, exist_ok=True)
    agg_path = os.path.join(args.outdir, f"k_sweep_agg_{args.selection}.csv")
    agg.to_csv(agg_path, index=False)
    print(f"aggregated means -> {agg_path}\n")

    best_rows = []
    for prob in sorted(df["problem"].unique()):
        sub = agg[agg["problem"] == prob]
        print(f"===== {prob}  (mean best-fitness by instance x k; higher = better) =====")
        pivot = sub.pivot_table(index=["instance", "size_num"], columns="k", values="mean_fit")
        print(pivot.round(2).to_string(), "\n")
        for (inst, size_num), row in pivot.iterrows():
            best_rows.append(dict(problem=prob, instance=inst, size=size_num,
                                  best_k=int(row.idxmax())))

    best = pd.DataFrame(best_rows).sort_values(["problem", "size"])
    print("===== best k per instance (sorted by problem, then size) =====")
    print(best.to_string(index=False))

    print("\n===== does the best k trend with instance SIZE? "
          "(Spearman corr of best_k vs size) =====")
    for prob in sorted(best["problem"].unique()):
        s = best[best["problem"] == prob]
        if len(s) >= 3 and s["best_k"].nunique() > 1:
            corr = s["size"].corr(s["best_k"], method="spearman")
            trend = "larger problems prefer larger k" if corr > 0.3 else \
                    "larger problems prefer smaller k" if corr < -0.3 else "no clear trend"
            print(f"  {prob:14s} corr={corr:+.2f}  ({trend}, n={len(s)})")
        else:
            print(f"  {prob:14s} not enough variation to judge (n={len(s)})")

    # plot: mean best-fitness vs k, one line per instance, per problem
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probs = sorted(df["problem"].unique())
    ks = sorted(df["k"].unique())
    fig, axes = plt.subplots(1, len(probs), figsize=(6 * len(probs), 5), squeeze=False)
    for ax, prob in zip(axes[0], probs):
        sub = agg[agg["problem"] == prob]
        for inst in sorted(sub["instance"].unique()):
            d = sub[sub["instance"] == inst].sort_values("k")
            ax.plot(d["k"], d["mean_fit"], marker="o", label=inst)
        ax.set_title(prob)
        ax.set_xlabel("tournament size k")
        ax.set_ylabel("mean best fitness (higher = better)")
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Best fitness vs tournament size k ({args.selection})")
    fig.tight_layout()
    png = os.path.join(args.outdir, f"k_sweep_{args.selection}.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"\nplot -> {png}")


if __name__ == "__main__":
    main()
