"""EC-KitY adapter for the Deep Tournament Selection (DTS) operator.

This is the thin glue layer that lets the learned, RL-trained DTS engine
(``DTSPolicy``) plug into EC-KitY's evolutionary loop by wrapping it as
an EC-KitY ``SelectionMethod``.

Interface mismatch this bridges
-------------------------------
DTS (numpy world)::

    select(population: np.ndarray, n_to_select: int,
           fitness_dict: dict, generation_index: int) -> np.ndarray

EC-KitY (Individual world)::

    select(source_inds: list[Individual], dest_inds: list[Individual]) -> dest_inds
"""
import numpy as np
from overrides import override

from eckity.genetic_operators.selections.selection_method import SelectionMethod

from .dts_policy import DTSPolicy


class DeepTournamentSelection(SelectionMethod):
    """EC-KitY ``SelectionMethod`` wrapping the learned DTS operator.

    Assumes maximization (``higher_is_better=True``) — this matches
    ``DTSPolicy``, which always treats fitness as a value to maximize.

    Parameters
    ----------
    policy : DTSPolicy
        The raw, already-constructed DTS engine (encoder + pointer + RL training).
    higher_is_better : bool, optional
        Fitness direction, by default True. Keep True unless the wrapped DTS is
        adapted for minimization.
    events : List[str], optional
        Selection events, by default None.
    """

    def __init__(self, policy: DTSPolicy, higher_is_better: bool = True, events=None):
        super().__init__(events=events, higher_is_better=higher_is_better)
        self.policy = policy
        # DTS needs a monotonically increasing per-generation counter for its
        # trajectory/reward bookkeeping (gen i vs gen i-1). SimpleBreeder calls
        # this select() exactly once per generation for a single subpopulation,
        # so an internal counter is correct.
        self.generation_index = 0

    @override
    def select(self, source_inds, dest_inds):
        # Number EC-KitY expects us to produce. Any elites have already been
        # placed in dest_inds by the breeder, so we only fill the remainder.
        n_to_select = len(source_inds) - len(dest_inds)

        # Individual objects -> the numpy world DTS understands.
        population = np.array([ind.vector for ind in source_inds])
        fitness_dict = {tuple(ind.vector): ind.get_pure_fitness() for ind in source_inds}
        # Map each gene-vector back to its source Individual so we can clone winners.
        lookup = {tuple(ind.vector): ind for ind in source_inds}

        selected_vectors = self.policy.select(
            population, n_to_select, fitness_dict, self.generation_index
        )
        self.generation_index += 1

        # numpy winners -> cloned Individual objects. DTS only ever picks
        # individuals that exist in the source population, so lookup never misses.
        for vec in selected_vectors:
            clone = lookup[tuple(vec)].clone()
            clone.selected_by.append(type(self).__name__)
            dest_inds.append(clone)

        self.selected_individuals = dest_inds
        return dest_inds
