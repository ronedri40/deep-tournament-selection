"""Strong-elitism breeder.

EC-KitY's default ``SimpleBreeder`` inserts the elites into the next generation
and then applies the crossover/mutation operators to the WHOLE next generation —
including the elites — so the best individual can degrade from one generation to
the next.

The original GA this project is based on used *strong* elitism: the best
``n_elite`` individuals are carried over **unchanged** (never crossed or mutated).
``ElitistBreeder`` restores that behaviour: operators are applied only to the
non-elite offspring, and the elites are appended untouched.
"""
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.genetic_operators.selections.elitism_selection import ElitismSelection


class ElitistBreeder(SimpleBreeder):
    def apply_breed(self, population):
        for subpopulation in population.sub_populations:
            nextgen_population = []

            num_elites = subpopulation.n_elite
            if num_elites > 0:
                ElitismSelection(
                    num_elites=num_elites,
                    higher_is_better=subpopulation.higher_is_better,
                ).apply_operator((subpopulation.individuals, nextgen_population))

            selection = subpopulation.get_selection_methods()[0][0]
            nextgen_population = selection.select(
                subpopulation.individuals, nextgen_population
            )
            self.selected_individuals = nextgen_population

            elites = nextgen_population[:num_elites]
            offspring = nextgen_population[num_elites:]

            # Crossover (arity 2) processes offspring in fixed-size chunks, so the
            # batch must be a multiple of the largest operator arity. With an odd
            # number of elites the offspring count can be odd (e.g. 100 - 3 = 97),
            # which would hand the crossover a lone individual. Pad with throwaway
            # clones up to a whole multiple, then trim them back off.
            operators = subpopulation.get_operators_sequence()
            max_arity = max((op.get_operator_arity() for op in operators), default=1)
            pad = (-len(offspring)) % max_arity
            if pad:
                offspring = offspring + [offspring[-1].clone() for _ in range(pad)]

            offspring = self._apply_operators(operators, offspring)
            if pad:
                offspring = offspring[:-pad]

            subpopulation.individuals = elites + offspring
