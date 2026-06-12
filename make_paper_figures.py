#!/usr/bin/env python
"""Reproduce the paper's per-generation fitness convergence figure, locally.

Runs the three *representative* benchmark instances from the paper
(Fig. "Fitness of the best individual per generation") with Deep Tournament
Selection and the Tournament baseline, a few runs each, then plots
``log(fitness)`` vs generation — one panel per domain, mirroring
``figures/fitness_graph_dts_all_domains.pdf``.

What the paper plots (from notebooks/res_analysis.py of the original repo):
  * representative instances: Graph Coloring = myciel7, Set Cover = scp65, TSP = berlin52
  * y-axis = log(-best_fitness) = log(cost); lower is better
  * averaged over runs (paper: 15; here: --runs, default 3)

Results are cached: each run writes a JSON via the FileLogger under --output, and
re-running skips runs whose JSON already exists (use --force to recompute). So you
can plot instantly after the runs finish, or resume an interrupted sweep.

Examples
--------
    # Reduced demo that still shows the published shape (~4-5h on CPU):
    python make_paper_figures.py --mode demo --runs 3

    # Faithful reproduction (paper gens: GC/SC 6000, TSP 1000; ~20h on CPU!):
    python make_paper_figures.py --mode paper --runs 3

    # Fast sanity check (~30 min):
    python make_paper_figures.py --mode quick --runs 2

    # Re-plot from already-saved runs without recomputing:
    python make_paper_figures.py --plot-only
"""
import argparse
import json
import os
from dataclasses import replace

import numpy as np

from run_experiments import PROBLEMS
from deep_tournament_selection.config import DTSConfig
from deep_tournament_selection.experiments.runner_utils import make_selection, run_one
from deep_tournament_selection.problems import DATA_DIR
from deep_tournament_selection.selection.custom_reward import custom_reward

# Representative instance per domain (exactly the ones in the paper's figure).
# NOTE: the paper's convergence text/diversity figure use scp65 for Set Cover
# (an older analysis script used scp52) — change here to compare the other one.
REPRESENTATIVE = {
    "graph_coloring": "instances_myciel7.col.txt",
    "set_cover": "scp65.txt",
    "tsp": "berlin52.tsp",
}
# Panel order + pretty titles for the figure.
PANELS = [
    ("graph_coloring", "Graph Coloring (myciel7)"),
    ("set_cover", "Set Cover (scp65)"),
    ("tsp", "TSP (berlin52)"),
]
SELECTIONS = ["dts", "tournament"]
SEL_STYLE = {"dts": dict(color="C3", label="DTS"),
             "tournament": dict(color="C0", label="Tournament k=5")}

# Per-domain generations for each mode. "paper" matches the article (Set Cover is
# ~20 h on CPU!); "demo" is the reduced run that still shows the published shape;
# "quick" is a fast sanity check. --generations overrides all of these.
MODES = {
    "paper": {"graph_coloring": 6000, "set_cover": 6000, "tsp": 1000},
    "demo":  {"graph_coloring": 2000, "set_cover": 1000, "tsp": 1000},
    "quick": {"graph_coloring": 600,  "set_cover": 300,  "tsp": 400},
}


def gens_for(domain, args):
    return args.generations or MODES[args.mode][domain]


def instance_path(domain, fname):
    return os.path.join(DATA_DIR, PROBLEMS[domain]["data"], fname)


def selected_panels(args):
    """Panels to run/plot, filtered by --domains (default: all three)."""
    chosen = getattr(args, "domains", None) or [d for d, _ in PANELS]
    return [(d, t) for (d, t) in PANELS if d in chosen]


def run_all(args):
    """Run (or reuse cached) the runs needed for every panel; return nothing."""
    # Optional ablation overrides. epsilon is shared (DTSConfig); the rest are
    # per-domain (applied via dataclasses.replace so the frozen config is intact).
    dts_cfg = DTSConfig()
    if args.epsilon is not None:
        dts_cfg = replace(dts_cfg, epsilon_greedy=args.epsilon, min_epsilon=args.epsilon)
    for domain, _title in selected_panels(args):
        spec = PROBLEMS[domain]
        cfg = spec["cfg"]
        overrides = {}
        if args.crossover_prob is not None:
            overrides["crossover_prob"] = args.crossover_prob
        if args.elitism is not None:
            overrides["elitism"] = args.elitism
        if args.flip_mutation_prob is not None and hasattr(cfg, "flip_mutation_prob"):
            overrides["flip_mutation_prob"] = args.flip_mutation_prob
        if overrides:
            cfg = replace(cfg, **overrides)
        fname = REPRESENTATIVE[domain]
        path = instance_path(domain, fname)
        generations = gens_for(domain, args)
        pop = args.population_size or cfg.population_size
        for sel in SELECTIONS:
            for run in range(args.runs):
                out = os.path.join(args.output, domain, fname, sel, f"run_{run}.json")
                if os.path.exists(out) and not args.force:
                    print(f"[skip cached] {domain}/{fname}/{sel}/run{run}")
                    continue
                ev, creator, operators, vocab, _ = spec["build"](
                    path, cfg, cfg.crossover_prob, cfg.mutation_prob)
                reward_fn = custom_reward if args.custom_reward else None
                selection = make_selection(sel, pop, vocab, dts_cfg=dts_cfg,
                                           device=args.device,
                                           custom_reward_function=reward_fn)
                print(f"[run] {domain}/{fname}/{sel}/run{run}  "
                      f"(gens={generations}, pop={pop})")
                run_one(f"{domain}/{fname}/{sel}/run{run}", creator, ev, operators,
                        selection, population_size=pop, generations=generations,
                        elitism=cfg.elitism, output_path=out, quiet=True,
                        diversity_fn=spec["diversity"])


def load_mean_curve(domain, fname, sel, output):
    """Average the per-generation best fitness (`max`) across all runs found."""
    base = os.path.join(output, domain, fname, sel)
    if not os.path.isdir(base):
        return None
    curves = []
    for f in sorted(os.listdir(base)):
        if not f.endswith(".json"):
            continue
        arr = np.asarray(json.load(open(os.path.join(base, f)))["max"], dtype=float)
        curves.append(arr)
    if not curves:
        return None
    # Pad ragged runs (e.g. one interrupted run) with NaN to the longest length,
    # then average ignoring NaNs — so a partial run contributes where it has data
    # instead of truncating the whole curve to its length.
    L = max(len(c) for c in curves)
    padded = np.full((len(curves), L), np.nan)
    for i, c in enumerate(curves):
        padded[i, :len(c)] = c
    return np.nanmean(padded, axis=0)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    panels = selected_panels(args)
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    axes = axes[0]
    for ax, (domain, title) in zip(axes, panels):
        fname = REPRESENTATIVE[domain]
        has_data = False
        for sel in SELECTIONS:
            avg = load_mean_curve(domain, fname, sel, args.output)
            if avg is None:
                continue
            has_data = True
            y = np.log(np.clip(-avg, 1e-9, None))           # log(cost); lower=better
            smooth = pd.Series(y).ewm(alpha=0.02).mean()
            ax.plot(y, alpha=0.15, color=SEL_STYLE[sel]["color"])
            ax.plot(smooth, linewidth=2.2, **SEL_STYLE[sel])
        ax.set_title(f"{title} — {gens_for(domain, args)} gens", fontsize=13)
        ax.set_xlabel("generation", fontsize=12)
        ax.set_ylabel("log(fitness)", fontsize=12)
        ax.grid(True, alpha=0.3)
        if has_data:
            ax.legend(fontsize=11)
        else:
            ax.text(0.5, 0.5, "no runs yet", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=13)
    fig.suptitle("Best-individual fitness per generation "
                 f"(avg over {args.runs} runs, mode={args.mode}) — DTS vs Tournament",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(args.figdir, exist_ok=True)
    tag = "all_domains" if len(panels) == len(PANELS) else "_".join(d for d, _ in panels)
    png = os.path.join(args.figdir, f"reproduction_fitness_{tag}.png")
    pdf = os.path.join(args.figdir, f"reproduction_fitness_{tag}.pdf")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"\nFigure saved to:\n  {png}\n  {pdf}")


def main():
    p = argparse.ArgumentParser(description="Reproduce the paper's fitness figure locally.")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--mode", choices=list(MODES), default="demo",
                   help="per-domain generations preset: paper (~20h, exact), "
                        "demo (~4-5h, shows the shape), quick (~30min sanity)")
    p.add_argument("--generations", type=int, default=None,
                   help="override generations for ALL domains (ignores --mode)")
    p.add_argument("--population-size", type=int, default=None)
    p.add_argument("--domains", nargs="+", choices=[d for d, _ in PANELS],
                   default=[d for d, _ in PANELS],
                   help="which experiment(s) to run/plot (default: all three). "
                        "e.g. --domains graph_coloring")
    p.add_argument("--crossover-prob", type=float, default=None,
                   help="override crossover probability (default: per-domain config)")
    p.add_argument("--flip-mutation-prob", type=float, default=None,
                   help="override per-gene flip mutation probability (GC/SC only)")
    p.add_argument("--elitism", type=int, default=None,
                   help="override the number of elites")
    p.add_argument("--epsilon", type=float, default=None,
                   help="override DTS teacher-forcing epsilon (constant; sets start == floor)")
    p.add_argument("--custom-reward", action="store_true",
                   help="use the paper's custom DTS reward (Eliad's Graph Coloring config)")
    p.add_argument("--force", action="store_true", help="recompute even if a run JSON exists")
    p.add_argument("--plot-only", action="store_true", help="skip running, just plot cached runs")
    p.add_argument("--output", default=os.path.join("runs", "paper_figures"))
    p.add_argument("--figdir", default="figures")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if not args.plot_only:
        run_all(args)
    plot(args)


if __name__ == "__main__":
    main()
