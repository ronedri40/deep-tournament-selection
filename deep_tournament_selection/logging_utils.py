"""Per-generation statistics logger.

Records per-generation fitness statistics (mean / std / median / max / min) and
wall-clock time, and writes them as JSON to the run's output path — periodically
during the run and once more at the end. Column names match the original
deep_roulette_selection repo's `generation_metrics` / results.json.
"""
import json
import os
import time

import numpy as np
from eckity.statistics.statistics import Statistics


class FileLogger(Statistics):
    METRIC_KEYS = ("mean", "std", "median", "max", "min", "time")

    def __init__(self, output_path=None, save_every_n_generations=50, format_string=None):
        super().__init__(format_string or "")
        self.output_path = output_path
        self.save_every_n_generations = save_every_n_generations
        self.generation_metrics = {key: [] for key in self.METRIC_KEYS}
        self._last_time = time.time()

    def write_statistics(self, sender, data_dict):
        sub_pop = data_dict["population"].sub_populations[0]
        fitness_values = np.array(
            [ind.get_pure_fitness() for ind in sub_pop.individuals], dtype=float
        )
        self.generation_metrics["mean"].append(float(np.mean(fitness_values)))
        self.generation_metrics["std"].append(float(np.std(fitness_values)))
        self.generation_metrics["median"].append(float(np.median(fitness_values)))
        self.generation_metrics["max"].append(float(np.max(fitness_values)))
        self.generation_metrics["min"].append(float(np.min(fitness_values)))

        now = time.time()
        self.generation_metrics["time"].append(now - self._last_time)
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
        payload = {key: self.generation_metrics[key] for key in self.METRIC_KEYS}
        if extra:
            payload.update(extra)
        with open(self.output_path, "w") as f:
            json.dump(payload, f, indent=2)
        return self.output_path
