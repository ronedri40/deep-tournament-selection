"""Persistent fitness cache for EC-KitY — a faithful port of the tuple-keyed
fitness cache used in the original custom GA (``ga_deapless.py``'s ``fitness_dict``).

EC-KitY's ``SimplePopulationEvaluator`` re-evaluates every individual every
generation with no caching. Wrapping the problem evaluator in ``CachingEvaluator``
restores the original behaviour: an individual whose genotype was already scored
(an elite, or an unchanged clone produced by selection) is served from cache
instead of being recomputed — which matters when the fitness function is expensive.

Usage:
    evaluator = CachingEvaluator(MyEvaluator())
    Subpopulation(..., evaluator=evaluator, ...)

Notes
-----
* Caches by ``tuple(individual.vector)`` — the same key the original GA used.
* Works with single-process evaluation (``max_workers=1``). Like the original
  cache, it is not shared across processes, so a process-pool evaluation would
  not see each other's cache entries.
* ``max_size`` optionally bounds the cache (oldest entries evicted), mirroring
  the original ``clear_fitness_dict_after_sampling`` memory-bounding intent.
"""
from collections import OrderedDict

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


class CachingEvaluator(SimpleIndividualEvaluator):
    def __init__(self, inner: SimpleIndividualEvaluator, max_size=None, events=None):
        super().__init__(events=events)
        self.inner = inner
        self.max_size = max_size
        self.fitness_cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def evaluate_individual(self, individual):
        key = tuple(individual.vector)
        if key in self.fitness_cache:
            self.hits += 1
            self.fitness_cache.move_to_end(key)  # LRU touch
            return self.fitness_cache[key]

        self.misses += 1
        value = self.inner.evaluate_individual(individual)
        self.fitness_cache[key] = value
        if self.max_size is not None and len(self.fitness_cache) > self.max_size:
            self.fitness_cache.popitem(last=False)  # evict oldest
        return value

    def cache_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.fitness_cache),
            "hit_rate": hit_rate,
        }
