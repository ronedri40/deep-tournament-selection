"""Step-by-step runner with a CUSTOM DTS reward.

Same shape as ``runners/graph_coloring.py``, but it shows how to train DTS with
your own reward instead of the default ``get_reward_from_fitness_scores``. Define
(or import) a reward function with the signature

    reward_fn(cur_gen_fitness, prev_gen_fitness, population) -> float

and pass it to ``make_selection(..., custom_reward_function=reward_fn)``. The
policy calls it once per generation. Run with
``python -m deep_tournament_selection.runners.custom_runner``.
"""

import os

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.genetic_operators.mutations.vector_random_mutation import (
    IntVectorOnePointMutation,
)
from eckity.subpopulation import Subpopulation

from ..caching_evaluator import CachingEvaluator
from ..config import DTSConfig, GraphColoringConfig
from ..elitist_breeder import ElitistBreeder
from ..experiments.runner_utils import make_selection
from ..logging_utils import FileLogger
from ..problems import DATA_DIR, GraphColoringEvaluator, VectorUniformCrossover
from ..problems.diversity import graph_coloring_diversity

# The paper's Graph Coloring reward; swap REWARD_FN below for your own function.
from ..selection.custom_reward import custom_reward

# --------------------------------------------------------------------------- #
# 1. PARAMETERS  — edit these
# --------------------------------------------------------------------------- #
cfg = GraphColoringConfig()
dts_cfg = DTSConfig()

INSTANCE = "instances_myciel7.col.txt"  # file under problems/data/graph_coloring/
GENERATIONS = 2000  # cfg.generations is 6000 in the paper
POPULATION_SIZE = cfg.population_size  # 100
CROSSOVER_PROB = cfg.crossover_prob  # 0.5
MUTATION_PROB = cfg.mutation_prob  # 0.5 (operator-level)
FLIP_MUTATION_PROB = cfg.flip_mutation_prob  # 0.1 (per-gene)
ELITISM = cfg.elitism  # 2
DEVICE = "cpu"  # "cpu" or "cuda"
OUTPUT_PATH = os.path.join("runs", "custom_reward", INSTANCE, "run_0.json")

# The reward DTS optimizes. Replace with any
# f(cur_gen_fitness, prev_gen_fitness, population) -> float.
REWARD_FN = custom_reward

# --------------------------------------------------------------------------- #
# 2. EVALUATOR
# --------------------------------------------------------------------------- #
instance_path = os.path.join(DATA_DIR, "graph_coloring", INSTANCE)
evaluator = GraphColoringEvaluator(instance_path, penalty=cfg.penalty)
n_nodes = evaluator.n_nodes
max_colors = n_nodes - cfg.colors_margin
evaluator = CachingEvaluator(evaluator)

# --------------------------------------------------------------------------- #
# 3. ENCODING + OPERATORS
# --------------------------------------------------------------------------- #
creator = GAIntVectorCreator(length=n_nodes, bounds=(0, max_colors))
operators = [
    VectorUniformCrossover(probability=CROSSOVER_PROB),
    IntVectorOnePointMutation(
        probability=MUTATION_PROB, probability_for_each=FLIP_MUTATION_PROB
    ),
]

# --------------------------------------------------------------------------- #
# 4. SELECTION  — DTS with the CUSTOM reward plugged in
# --------------------------------------------------------------------------- #
selection = make_selection(
    "dts",
    POPULATION_SIZE,
    vocab_size=max_colors + 1,
    dts_cfg=dts_cfg,
    device=DEVICE,
    custom_reward_function=REWARD_FN,
)

# --------------------------------------------------------------------------- #
# 5. ASSEMBLE THE GA
# --------------------------------------------------------------------------- #
logger = FileLogger(OUTPUT_PATH, diversity_fn=graph_coloring_diversity)
algo = SimpleEvolution(
    Subpopulation(
        creators=creator,
        population_size=POPULATION_SIZE,
        evaluator=evaluator,
        higher_is_better=True,
        elitism_rate=ELITISM / POPULATION_SIZE,
        operators_sequence=operators,
        selection_methods=[(selection, 1)],
    ),
    breeder=ElitistBreeder(),
    max_generation=GENERATIONS,
    statistics=[logger],
)

# --------------------------------------------------------------------------- #
# 6. RUN
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    algo.evolve()
    best = algo.best_of_run_.get_pure_fitness()
    print(f"best fitness = {best:.0f}  (custom reward = {REWARD_FN.__name__})")
    saved = logger.save(extra={"label": f"custom_reward/{INSTANCE}"})
    print(f"metrics saved to: {saved}")
