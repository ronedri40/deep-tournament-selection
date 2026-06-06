import numpy as np
from collections import namedtuple

ReplaySamples = namedtuple("Sample", [
    "population",
    "fitness",
    "tournament",
    "predicted_tournament_indices",
    "next_generation_fitness",
])


class ReplayBuffer:
    def __init__(self, sample_size=16, min_size=128, max_buffer_size=1024, replay_weight=0.1):
        assert min_size >= sample_size, "min_size must be greater than or equal to sample_size"
        self.population_buffer: list[np.ndarray] = []
        self.fitness_buffer: list[np.ndarray] = []
        self.tournament_buffer: list[np.ndarray] = []
        self.predicted_tournament_buffer_indices: list[np.ndarray] = []
        self.sample_size = sample_size
        self.min_size = min_size
        self.max_buffer_size = max_buffer_size
        self.replay_weight = replay_weight

    def get_replay_weight(self):
        return self.replay_weight

    def log_to_buffer(self, population: np.ndarray, fitness: np.ndarray,
                      tournament: np.ndarray, predicted_tournament_indices: np.ndarray):
        self.population_buffer.append(population)
        self.fitness_buffer.append(fitness)
        self.tournament_buffer.append(tournament)
        self.predicted_tournament_buffer_indices.append(predicted_tournament_indices)

        if len(self.population_buffer) > self.max_buffer_size:
            self.population_buffer.pop(0)
            self.fitness_buffer.pop(0)
            self.tournament_buffer.pop(0)
            self.predicted_tournament_buffer_indices.pop(0)

    def is_sample_ready(self):
        return len(self.population_buffer) >= self.min_size

    def sample_from_buffer(self) -> ReplaySamples:
        if len(self.population_buffer) < self.min_size:
            raise ValueError("Not enough data in buffer to sample.")

        indices = np.random.choice(len(self.population_buffer) - 1, self.sample_size, replace=False)
        sampled_population = [self.population_buffer[i] for i in indices]
        sampled_fitness = [self.fitness_buffer[i] for i in indices]
        sampled_tournament = [self.tournament_buffer[i] for i in indices]
        sampled_predicted_tournament_indices = [self.predicted_tournament_buffer_indices[i] for i in indices]
        sampled_next_generation_fitness = [self.fitness_buffer[i + 1] for i in indices]

        return ReplaySamples(
            sampled_population,
            sampled_fitness,
            sampled_tournament,
            sampled_predicted_tournament_indices,
            sampled_next_generation_fitness,
        )
