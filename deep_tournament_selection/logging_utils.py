"""Lightweight JSON statistics for the experiment runners.

EC-KitY logs best/avg/worst to the Python logger; this Statistics subclass also
records the best (and average) fitness of every generation and writes them to a
JSON file at the end of a run, so results can be analysed/plotted later.
"""
import json
import os

from eckity.statistics.statistics import Statistics


class JsonStatistics(Statistics):
    """Records best/average fitness per generation and dumps to JSON on save()."""

    def __init__(self, output_path=None, format_string=None):
        super().__init__(format_string or "")
        self.output_path = output_path
        self.best_per_gen = []
        self.avg_per_gen = []

    def write_statistics(self, sender, data_dict):
        sub_pop = data_dict["population"].sub_populations[0]
        self.best_per_gen.append(sub_pop.get_best_individual().get_pure_fitness())
        self.avg_per_gen.append(sub_pop.get_average_fitness())

    def save(self, extra=None):
        if self.output_path is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        payload = {
            "best_per_generation": self.best_per_gen,
            "avg_per_generation": self.avg_per_gen,
            "best_fitness": max(self.best_per_gen) if self.best_per_gen else None,
            "generations": len(self.best_per_gen),
        }
        if extra:
            payload.update(extra)
        with open(self.output_path, "w") as f:
            json.dump(payload, f, indent=2)
        return self.output_path
