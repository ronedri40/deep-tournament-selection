"""OneMax demo: the learned DTS operator plugged into EC-KitY's SimpleEvolution.

This is the selection-side analogue of BERT-Mutation-GA's experiments: we take a
problem that already exists in EC-KitY (OneMax) and swap the standard
``TournamentSelection`` for our ``DeepTournamentSelection`` (DTS).

Run:
    python -m experiments.one_max
    python -m experiments.one_max --population-size 100 --length 50 --generations 50
"""
import argparse

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import GABitStringVectorCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import BitStringVectorNFlipMutation
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import (
    ThresholdFromTargetTerminationChecker,
)

from .common import build_dts_operator
from ..caching_evaluator import CachingEvaluator


class OneMaxEvaluator(SimpleIndividualEvaluator):
    """Standard OneMax fitness: maximize the number of 1s (EC-KitY example problem)."""

    def evaluate_individual(self, individual):
        return sum(individual.vector)


def main():
    parser = argparse.ArgumentParser(description="OneMax with Deep Tournament Selection (DTS)")
    parser.add_argument("--population-size", type=int, default=300)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-cache", dest="cache", action="store_false",
                        help="disable the persistent fitness cache")
    args = parser.parse_args()

    dts = build_dts_operator(
        population_size=args.population_size,
        vocab_size=2,  # bit vector: genes are 0/1
        device=args.device,
    )

    # Persistent fitness cache (ports the original GA's fitness_dict): skips
    # recomputation for genotypes already scored (elites, unchanged clones).
    evaluator = OneMaxEvaluator()
    if args.cache:
        evaluator = CachingEvaluator(evaluator)

    algo = SimpleEvolution(
        Subpopulation(
            creators=GABitStringVectorCreator(length=args.length),
            population_size=args.population_size,
            evaluator=evaluator,
            higher_is_better=True,
            elitism_rate=1 / args.population_size,
            operators_sequence=[
                VectorKPointsCrossover(probability=0.5, k=1),
                BitStringVectorNFlipMutation(
                    probability=0.2, probability_for_each=0.05, n=args.length
                ),
            ],
            # the learned DTS operator replaces the usual TournamentSelection here
            selection_methods=[(dts, 1)],
        ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=args.generations,
        termination_checker=ThresholdFromTargetTerminationChecker(
            optimal=args.length, threshold=0.0
        ),
        statistics=BestAverageWorstStatistics(),
    )

    algo.evolve()
    print(algo.execute())
    if args.cache:
        print("fitness cache:", evaluator.cache_stats())


if __name__ == "__main__":
    main()
