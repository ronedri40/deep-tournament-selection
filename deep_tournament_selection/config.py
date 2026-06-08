"""Per-problem default hyperparameters, mirroring the paper's experiment setup.

Defaults match the original repo's *_dns.py scripts (population 100, 6000
generations, etc.). The experiment runners expose CLI overrides so you can run a
quick smoke test without editing this file.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DTSConfig:
    """Hyperparameters for the Deep Tournament Selection operator."""
    learning_rate: float = 2e-3
    train_every_n_gens: int = 10
    tournament_size: int = 5
    latent_dim: int = 32
    epsilon_greedy: float = 1.0
    epsilon_greedy_decay: float = 0.999
    min_epsilon: float = 0.2


@dataclass(frozen=True)
class TSPConfig:
    instances: List[str] = field(default_factory=lambda: [
        "att48.tsp", "berlin52.tsp", "st70.tsp", "lin105.tsp", "pr107.tsp",
        "pcb442.tsp", "d1291.tsp",
    ])
    population_size: int = 100
    generations: int = 6000
    crossover_prob: float = 0.6
    mutation_prob: float = 0.1
    elitism: int = 2
    runs: int = 20


@dataclass(frozen=True)
class GraphColoringConfig:
    instances: List[str] = field(default_factory=lambda: [
        "instances_games120.col.txt", "instances_myciel7.col.txt",
        "miles1000.col.txt", "miles1500.col.txt", "mulsol.i.2.col",
        "queen8_12.col.txt", "zeroin.i.1.col", "zeroin.i.2.col",
    ])
    population_size: int = 100
    generations: int = 6000
    crossover_prob: float = 0.8
    mutation_prob: float = 0.5
    flip_mutation_prob: float = 0.005
    penalty: float = 100.0
    colors_margin: int = 10  # max colors = n_nodes - colors_margin
    elitism: int = 3
    runs: int = 30


@dataclass(frozen=True)
class SetCoverConfig:
    instances: List[str] = field(default_factory=lambda: [
        "scp41.txt", "scp51.txt", "scp52.txt", "scp53.txt", "scp54.txt",
        "scp56.txt", "scp57.txt", "scp64.txt", "scp65.txt",
    ])
    population_size: int = 100
    generations: int = 6000
    crossover_prob: float = 0.8
    mutation_prob: float = 0.5
    flip_mutation_prob: float = 0.1
    penalty: float = 100.0
    weighted: bool = False
    elitism: int = 3
    runs: int = 30
