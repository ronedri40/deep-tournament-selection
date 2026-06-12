#!/usr/bin/env python
"""Sweep the tournament size k across all problems (paper protocol: 15 runs).

For each k in --k-values, runs every problem x instance x run with the tournament
size set to k (this drives DTS's contextual tournament, and the tournament
baseline if --selection includes it). One JSON per run is written under

    <output>/k<k>/<problem>/<instance>/<selection>/run_<r>.json

plus a tidy ``k_sweep_summary.csv`` (one row per run) for analysis. The sweep is
RESUMABLE: a run whose JSON already exists is skipped, so you can run it in
chunks or across several SLURM jobs.

WARNING — this is large. 4 k-values x 3 problems x (all config instances) x 15
runs x full generations is days of GPU time. Start small to validate, e.g.:

    python run_k_sweep.py --problems graph_coloring --instances instances_myciel7.col.txt \
        --runs 3 --generations 1000 --device cuda

then scale up (drop --instances/--runs/--generations to use the paper defaults).
"""

import argparse
import csv
import os
import time
from dataclasses import replace

from deep_tournament_selection.config import DTSConfig
from deep_tournament_selection.experiments.runner_utils import (
    configure_logging,
    make_selection,
    run_one,
)
from run_experiments import PROBLEMS, resolve_instances


def main():
    p = argparse.ArgumentParser(description="Tournament-size (k) sweep across problems.")
    p.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 10, 20])
    p.add_argument("--selection", choices=["dts", "tournament", "both"], default="dts")
    p.add_argument("--problems", nargs="+", default=list(PROBLEMS), choices=list(PROBLEMS))
    p.add_argument("--instances", nargs="+", default=None,
                   help="instance filenames or 'all'; default = each problem's config list")
    p.add_argument("--runs", type=int, default=15)
    p.add_argument("--generations", type=int, default=None,
                   help="override generations (default: per-problem config)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=os.path.join("runs", "k_sweep"))
    p.add_argument("--no-diversity", action="store_true")
    args = p.parse_args()
    configure_logging(quiet=True)

    selections = ["dts", "tournament"] if args.selection == "both" else [args.selection]
    rows = []
    for k in args.k_values:
        dts_cfg = replace(DTSConfig(), tournament_size=k)
        for prob in args.problems:
            spec = PROBLEMS[prob]
            cfg = spec["cfg"]
            gens = args.generations or cfg.generations
            instances = resolve_instances(spec["data"], args.instances, cfg.instances)
            for inst_name, inst_path in instances:
                for sel in selections:
                    for run in range(args.runs):
                        out = os.path.join(
                            args.output, f"k{k}", prob, inst_name, sel, f"run_{run}.json"
                        )
                        if os.path.exists(out):
                            print(f"[skip cached] k{k}/{prob}/{inst_name}/{sel}/run{run}")
                            continue
                        ev, creator, operators, vocab, size_label = spec["build"](
                            inst_path, cfg, cfg.crossover_prob, cfg.mutation_prob
                        )
                        selection = make_selection(
                            sel, cfg.population_size, vocab, dts_cfg=dts_cfg,
                            device=args.device,
                        )
                        t0 = time.time()
                        res = run_one(
                            f"k{k}/{prob}/{inst_name}/{sel}/run{run}",
                            creator, ev, operators, selection,
                            population_size=cfg.population_size, generations=gens,
                            elitism=cfg.elitism, output_path=out, quiet=True,
                            diversity_fn=None if args.no_diversity else spec["diversity"],
                        )
                        dt = time.time() - t0
                        score = spec["score"](res["best_fitness"])
                        rows.append(dict(
                            k=k, problem=prob, instance=inst_name, size=size_label,
                            selection=sel, run=run, best_fitness=res["best_fitness"],
                            score=score, score_name=spec["score_name"], seconds=round(dt, 1),
                        ))
                        print(f"[k={k} {prob} {inst_name} {sel} run{run}] "
                              f"{spec['score_name']}={score:.1f}  ({dt:.1f}s)")

    if rows:
        os.makedirs(args.output, exist_ok=True)
        csv_path = os.path.join(args.output, "k_sweep_summary.csv")
        # Append if the file exists (so resumed chunks accumulate), else write header.
        keys = ["k", "problem", "instance", "size", "selection", "run",
                "best_fitness", "score", "score_name", "seconds"]
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if not exists:
                w.writeheader()
            w.writerows(rows)
        print(f"\nSummary appended to {csv_path}  (+{len(rows)} runs this session)")


if __name__ == "__main__":
    main()
