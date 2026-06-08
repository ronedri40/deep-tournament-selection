# Deep Tournament Selection (DTS) for EC-KitY

A learned, RL-trained **selection** operator for genetic algorithms, packaged as an
[EC-KitY](https://github.com/EC-KitY/EC-KitY) `SelectionMethod`. This repo follows the same
structure as [BERT-Mutation-GA](https://github.com/EC-KitY/BERT-Mutation-GA): one custom genetic
operator wrapped in a thin EC-KitY adapter, plus runnable experiments.

Instead of a fixed rule (tournament, roulette, …), DTS *learns* which individuals to select. A
Transformer encoder embeds the whole population, a self-attention pointer network picks tournament
winners, and the policy is trained online with REINFORCE using the population's fitness improvement
as reward.

## Layout

```
deep_tournament_selection/
  selection/
    eckity_adapter.py        # DeepTournamentSelection(SelectionMethod)  <-- the EC-KitY glue
    deep_neural_selection.py # core DTS: REINFORCE training loop
    population_to_vec_transformer.py / self_attention_pointer.py / ...  # the neural networks
    tournament_utils.py      # standalone tournament helper (teacher-forcing baseline)
  experiments/
    common.py                # build_dts_operator(...) -> ready-to-use SelectionMethod
    one_max.py               # OneMax demo (built-in EC-KitY problem) using DTS
```

This is a standalone repository that depends on EC-KitY (it does not modify the EC-KitY source
tree). Install it next to EC-KitY and it works as a drop-in selection operator.

## How it plugs in

`DeepTournamentSelection` extends EC-KitY's `SelectionMethod`. Its `select(source_inds, dest_inds)`
converts the `Individual` objects into the numpy `(population, fitness_dict)` form the learned
operator expects, runs the policy, then clones the chosen individuals back into `dest_inds`.

```python
from deep_tournament_selection.experiments.common import build_dts_operator

dts = build_dts_operator(population_size=300, vocab_size=2)  # vocab_size = max gene value + 1
# ... then in your Subpopulation:
selection_methods=[(dts, 1)]
```

## Benchmark problems (from the paper)

The repo bundles the three problems reported in the DTS paper, each as an EC-KitY
evaluator with a per-problem runner (mirroring BERT-Mutation-GA's experiments layout).
Instance files are bundled under `deep_tournament_selection/problems/data/`.

| Problem | Representation | Operators | Runner |
|---|---|---|---|
| **TSP** | permutation (IntVector) | SCX crossover + RSM mutation | `experiments.tsp` |
| **Graph Coloring** | IntVector, colors `[0, n-10]` | uniform XO + per-gene int mutation | `experiments.graph_coloring` |
| **Set Cover** | bit vector | uniform XO + per-bit flip mutation | `experiments.set_cover` |

All three are framed as maximization (fitness = negated cost), matching DTS. Each runner
swaps in DTS by default, or the standard tournament baseline with `--selection tournament`:

```bash
# defaults (paper instances, pop=100, 6000 gens) — heavy; use flags for a quick run:
python -m deep_tournament_selection.experiments.tsp --instance att48.tsp --generations 100
python -m deep_tournament_selection.experiments.graph_coloring --instance queen8_12.col.txt --generations 100
python -m deep_tournament_selection.experiments.set_cover --instance scp41.txt --generations 100

# compare against the baseline selection on the same setup:
python -m deep_tournament_selection.experiments.tsp --instance att48.tsp --generations 100 --selection tournament

# run every bundled instance:
python -m deep_tournament_selection.experiments.set_cover --instance all --runs 5
```

Common flags: `--instance <file|all>`, `--selection dts|tournament`, `--population-size`,
`--generations`, `--runs`, `--crossover-prob`, `--mutation-prob`, `--output`, `--device`, `--quiet`.
Per-generation best/avg fitness is saved as JSON under `--output` (default `runs/`).
Default hyperparameters per problem live in `config.py`.

## Fitness cache

EC-KitY re-evaluates every individual every generation. To preserve the original GA's
cross-generation fitness cache (skip recomputing the fitness of genotypes already scored — elites,
unchanged clones), wrap your evaluator in `CachingEvaluator`:

```python
from deep_tournament_selection import CachingEvaluator

evaluator = CachingEvaluator(MyEvaluator())   # tuple(vector) -> fitness, like the old fitness_dict
Subpopulation(..., evaluator=evaluator, ...)
print(evaluator.cache_stats())                # {'hits': ..., 'misses': ..., 'hit_rate': ...}
```

The `one_max` demo enables it by default (use `--no-cache` to turn it off). Caching matters most when
the fitness function is expensive; it works with single-process evaluation (`max_workers=1`).

## Install & run

```bash
pip install -e .            # or: pip install -r requirements.txt
python -m deep_tournament_selection.experiments.one_max          # full run (pop=300, len=100)
# quick smoke test:
python -m deep_tournament_selection.experiments.one_max --population-size 60 --length 40 --generations 30
```

You should see best fitness climb toward `length`, with periodic `loss: ..., reward: ...` lines
printed by the DTS training loop — confirming the operator is learning inside EC-KitY's evolution.
