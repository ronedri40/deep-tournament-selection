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

            # Protect the elites: apply operators ONLY to the offspring.
            elites = nextgen_population[:num_elites]
            offspring = self._apply_operators(
                subpopulation.get_operators_sequence(),
                nextgen_population[num_elites:],
            )
            subpopulation.individuals = elites + offspring
