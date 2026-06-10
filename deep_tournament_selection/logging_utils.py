"""Per-generation statistics logger.

Records best / worst / average / std fitness and per-generation wall-clock time,
and writes them as JSON to the run's output path — periodically during the run
and once more at the end. Modeled on BERT-Mutation-GA's FileLogger.
"""
import json
import os
import time

import numpy as np
from eckity.statistics.statistics import Statistics


class FileLogger(Statistics):
    def __init__(self, output_path=None, save_every_n_generations=50, format_string=None):
        super().__init__(format_string or "")
        self.output_path = output_path
        self.save_every_n_generations = save_every_n_generations
        self.best_per_generation = []
        self.worst_per_generation = []
        self.avg_per_generation = []
        self.std_per_generation = []
        self.time_per_generation = []
        self._last_time = time.time()

    def write_statistics(self, sender, data_dict):
        sub_pop = data_dict["population"].sub_populations[0]
        fitnesses = np.array(
            [ind.get_pure_fitness() for ind in sub_pop.individuals], dtype=float
        )
        self.best_per_generation.append(sub_pop.get_best_individual().get_pure_fitness())
        self.worst_per_generation.append(sub_pop.get_worst_individual().get_pure_fitness())
        self.avg_per_generation.append(float(np.mean(fitnesses)))
        self.std_per_generation.append(float(np.std(fitnesses)))

        now = time.time()
        self.time_per_generation.append(now - self._last_time)
        self._last_time = now

        if (
            self.output_path
            and self.save_every_n_generations
            and data_dict["generation_num"] % self.save_every_n_generations == 0
        ):
            self.save()

    def save(self, extra=None):
        if self.output_path is None:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        payload = {
            "best_per_generation": self.best_per_generation,
            "worst_per_generation": self.worst_per_generation,
            "avg_per_generation": self.avg_per_generation,
            "std_per_generation": self.std_per_generation,
            "time_per_generation": self.time_per_generation,
            "generations": len(self.best_per_generation),
            "best_fitness": (
                max(self.best_per_generation) if self.best_per_generation else None
            ),
        }
        if extra:
            payload.update(extra)
        with open(self.output_path, "w") as f:
            json.dump(payload, f, indent=2)
        return self.output_path
