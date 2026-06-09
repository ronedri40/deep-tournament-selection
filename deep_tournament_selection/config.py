"""Per-problem hyperparameters matching the paper's experimental setting.

From "Deep Tournament Selection for Genetic Algorithms" (Shem-Tov, Edri, Elyasaf):
identical GA parameters across domains — population 100, elitism 2, crossover
probability 0.5, uniform mutation probability 0.01, 15 repeats; 6000 generations
for Graph Coloring and Set Cover, 1000 for TSP. DTS uses tournament size k=5, a
2-layer / 4-head Transformer encoder (feedforward 256, latent 32), Adam with
lr 2e-3 linearly decayed to 1e-3, reward over the top-m=5 individuals, and model
updates every 10 generations.

The runners expose CLI overrides so you can run a quick smoke test without editing
this file.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DTSConfig:
    """Hyperparameters for the Deep Tournament Selection operator (paper setting)."""
    tournament_size: int = 5
    latent_dim: int = 32
    dim_feedforward: int = 256
    n_layers: int = 2
    n_heads: int = 4
    learning_rate: float = 2e-3
    final_lr: float = 1e-3            # linearly decayed to this
    train_every_n_gens: int = 10
    # Teacher forcing: start greedy (follow the tournament winner) and decay
    # exponentially down to a 0.2 floor — a gradual transition from greedy
    # behaviour to fully learned selection (paper).
    epsilon_greedy: float = 1.0
    epsilon_greedy_decay: float = 0.999
    min_epsilon: float = 0.2


# Shared GA parameters (identical across domains, per the paper)
POPULATION_SIZE = 100
ELITISM = 2
CROSSOVER_PROB = 0.5
MUTATION_PROB = 0.01      # uniform mutation probability (per gene for GC/SC)
RUNS = 15


@dataclass(frozen=True)
class TSPConfig:
    instances: List[str] = field(default_factory=lambda: [
        "att48.tsp", "berlin52.tsp", "st70.tsp", "lin105.tsp", "pr107.tsp",
        "pcb442.tsp", "d1291.tsp",
    ])
    population_size: int = POPULATION_SIZE
    generations: int = 1000
    crossover_prob: float = CROSSOVER_PROB
    mutation_prob: float = MUTATION_PROB
    elitism: int = ELITISM
    runs: int = RUNS


@dataclass(frozen=True)
class GraphColoringConfig:
    instances: List[str] = field(default_factory=lambda: [
        "instances_games120.col.txt", "instances_myciel7.col.txt",
        "miles1000.col.txt", "miles1500.col.txt", "mulsol.i.2.col",
        "queen8_12.col.txt", "zeroin.i.1.col", "zeroin.i.2.col",
    ])
    population_size: int = POPULATION_SIZE
    generations: int = 6000
    crossover_prob: float = CROSSOVER_PROB
    mutation_prob: float = MUTATION_PROB
    penalty: float = 100.0
    colors_margin: int = 10  # max colors = n_nodes - colors_margin
    elitism: int = ELITISM
    runs: int = RUNS


@dataclass(frozen=True)
class SetCoverConfig:
    instances: List[str] = field(default_factory=lambda: [
        "scp41.txt", "scp51.txt", "scp52.txt", "scp53.txt", "scp54.txt",
        "scp56.txt", "scp57.txt", "scp64.txt", "scp65.txt",
    ])
    population_size: int = POPULATION_SIZE
    generations: int = 6000
    crossover_prob: float = CROSSOVER_PROB
    mutation_prob: float = MUTATION_PROB
    penalty: float = 100.0
    weighted: bool = False
    elitism: int = ELITISM
    runs: int = RUNS
