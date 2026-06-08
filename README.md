# Deep Tournament Selection (DTS) for Genetic Algorithms

Reference implementation of **"Deep Tournament Selection for Genetic Algorithms"**
(Eliad Shem-Tov, Ron Edri, Achiya Elyasaf — Ben-Gurion University of the Negev).

DTS is a **learned, domain-independent selection operator** for genetic algorithms. It reframes
tournament selection as a Markov Decision Process optimized online with policy-gradient
reinforcement learning, and is implemented as a drop-in [EC-KitY](https://github.com/EC-KitY/EC-KitY)
`SelectionMethod`.

<p align="center">
  <img src="images/dns_arch.png" alt="Deep Tournament Selection architecture" width="780">
</p>

## Abstract

In Genetic Algorithms (GAs), the selection operator plays a critical role in balancing exploration
and exploitation. However, classical and adaptive selection mechanisms largely rely on static rules
or handcrafted heuristics that fail to adapt to the real-time dynamics of the evolving population. In
this work, we introduce **Deep Tournament Selection (DTS)**, a novel domain-independent selection
operator that reformulates tournament selection as a Markov Decision Process optimized via
reinforcement learning. DTS evaluates candidate solutions in a tournament using a Transformer encoder
augmented with global and local rank-based positional encodings, along with a self-attention pointer
mechanism. The policy is trained fully online using policy-gradient reinforcement learning **without
requiring additional fitness evaluations**, enabling the operator to dynamically adjust its selection
pressure. We evaluate DTS on three canonical combinatorial optimization domains — **Graph Coloring,
Set Cover, and the Traveling Salesman Problem** — and show faster convergence, improved solution
quality, and robust performance compared to classical and dynamic selection baselines, while
introducing negligible computational overhead and preserving population diversity.

## Method

DTS operates through three components (`deep_tournament_selection/selection/`):

1. **Fitness-Augmented Embedding** — a Transformer encoder maps each individual (an integer vector)
   to a latent vector, augmented with **global** (population-level) and **local** (within-tournament)
   rank-based positional encodings that inject relative fitness information.
2. **Contextual Tournament Evaluation** — a self-attention pointer mechanism jointly scores the
   candidates in each tournament and produces a stochastic selection policy.
3. **Selection as an MDP** — parent sampling is framed as an RL problem and the policy is trained
   online with policy gradients, using the improvement in the top-*m* individuals between consecutive
   generations as reward (no extra fitness evaluations).

The whole stack is wrapped in a thin EC-KitY adapter,
`selection/eckity_adapter.py::DeepTournamentSelection(SelectionMethod)`, so it slots into any GA as a
single line: `selection_methods=[(dts, 1)]`.

## Repository layout

```
deep_tournament_selection/
  selection/            # DTS operator + EC-KitY adapter + the neural networks
    eckity_adapter.py   #   DeepTournamentSelection(SelectionMethod)  <-- the glue
    deep_neural_selection.py, population_to_vec_transformer.py, self_attention_pointer.py, ...
  caching_evaluator.py  # persistent fitness cache (ports the original GA's fitness_dict)
  problems/             # EC-KitY evaluators + loaders + custom operators + bundled instances
    tsp.py, graph_coloring.py, set_cover.py, operators.py, data/
  experiments/          # per-problem runners + shared helpers
    tsp.py, graph_coloring.py, set_cover.py, common.py, runner_utils.py
  config.py             # paper hyperparameters
  logging_utils.py      # per-generation JSON statistics
run_experiments.py      # single entry point to sweep all problems x instances x runs
figures/                # fitness / diversity plots from the paper
images/                 # architecture diagrams
```

## Benchmark domains (from the paper)

| Domain | Encoding | Operators | Instances |
|---|---|---|---|
| **Graph Coloring** | integer vector (color per vertex) | uniform crossover + uniform mutation | DIMACS, 96–450 vertices |
| **Set Cover** | bit vector (subset selected) | uniform crossover + bit-flip mutation | OR-Library, 1000/2000 subsets |
| **TSP** | permutation (tour order) | SCX crossover + RSM mutation | TSPLIB, 48–1291 cities |

All are framed as maximization (fitness = negated cost), matching DTS. Instance files are bundled
under `deep_tournament_selection/problems/data/`.

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
```
Requires Python 3.9+, PyTorch, numpy, eckity, overrides.

## Usage

Each problem has its own runner; swap DTS for the tournament baseline with `--selection tournament`.
Defaults follow the paper (pop 100, 6000 gens for GC/SC, 1000 for TSP); use flags for a quick run:

```bash
# quick single-instance runs
python -m deep_tournament_selection.experiments.graph_coloring --instance queen8_12.col.txt --generations 200
python -m deep_tournament_selection.experiments.set_cover      --instance scp41.txt        --generations 200
python -m deep_tournament_selection.experiments.tsp            --instance att48.tsp         --generations 200

# baseline comparison on the same setup
python -m deep_tournament_selection.experiments.tsp --instance att48.tsp --generations 200 --selection tournament

# sweep everything (paper protocol uses --runs 15)
python run_experiments.py --problems all --selection both --runs 3 --generations 500
```

Common flags: `--instance <file|all>`, `--selection dts|tournament`, `--population-size`,
`--generations`, `--runs`, `--crossover-prob`, `--mutation-prob`, `--output`, `--device`, `--quiet`.
Per-generation best/avg fitness is written as JSON under `--output` (default `runs/`).

### Using DTS in your own EC-KitY experiment

```python
from deep_tournament_selection.experiments.common import build_dts_operator

dts = build_dts_operator(population_size=100, vocab_size=2)  # vocab_size = max gene value + 1
# ... inside your Subpopulation:
selection_methods=[(dts, 1)]
```

## Fitness cache

EC-KitY re-evaluates every individual every generation. To preserve the original GA's
cross-generation fitness cache (skip recomputing already-scored genotypes — elites, unchanged
clones), wrap your evaluator in `CachingEvaluator` (the runners do this by default):

```python
from deep_tournament_selection import CachingEvaluator
evaluator = CachingEvaluator(MyEvaluator())
print(evaluator.cache_stats())   # {'hits': ..., 'misses': ..., 'hit_rate': ...}
```

## Results figures

The paper's fitness and population-diversity plots are in [`figures/`](figures/)
(`fitness_graph_dns_all_domains.pdf`, per-domain plots, and `hamming_distance_all_domains.pdf`).

## Citation

> **Note:** the paper has not been published yet. A citation (BibTeX) will be added here once it is
> available.

Paper: *Deep Tournament Selection for Genetic Algorithms* — Eliad Shem-Tov, Ron Edri, Achiya Elyasaf
(Ben-Gurion University of the Negev).

## License

Released under the **BSD 3-Clause License** — see [LICENSE](LICENSE).
