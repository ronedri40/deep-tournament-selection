import numpy as np
import torch


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class FitnessFeaturizer:
    """
    A class for featurizing fitness scores into a feature representation.
    Assumes higher fitness scores are better.
    """
    N_FEATURES = 5  # Number of features generated per individual

    def __init__(self, numerical_stability: float = 1e-8):
        self.best_fitness = -np.inf
        self.numerical_stability = numerical_stability

    def featurize(self, fitness_scores: np.ndarray) -> np.ndarray:
        features = []

        # Z-Score Normalization
        z_scores = self.get_z_score(fitness_scores)
        features.append(z_scores)

        # Improvement Indicator
        is_improved = self.get_is_improved(fitness_scores)
        features.append(is_improved)

        # Normalized Range
        norm_range = self.get_norm_range(fitness_scores)
        features.append(norm_range)

        # SNES Weights
        snes_weights = self.get_snes_weights(fitness_scores)
        features.append(snes_weights)

        # DES Weights
        des_weights = self.get_des_weights(fitness_scores)
        features.append(des_weights)

        # Concatenate all features
        feature_matrix = np.concatenate(features, axis=1)

        # Update best fitness
        current_best = np.max(fitness_scores)
        if current_best > self.best_fitness:
            self.best_fitness = current_best

        return feature_matrix

    def get_z_score(self, fitness_scores: np.ndarray) -> np.ndarray:
        mean = np.mean(fitness_scores)
        std = np.std(fitness_scores) + self.numerical_stability
        z_scores = (fitness_scores - mean) / std
        return z_scores.reshape(-1, 1)

    def get_is_improved(self, fitness_scores: np.ndarray) -> np.ndarray:
        is_improved = (fitness_scores > self.best_fitness).astype(float)
        return is_improved.reshape(-1, 1)

    def get_norm_range(self, fitness_scores: np.ndarray) -> np.ndarray:
        min_val = np.min(fitness_scores)
        max_val = np.max(fitness_scores)
        norm_range = (fitness_scores - min_val) / (max_val - min_val + self.numerical_stability)
        return norm_range.reshape(-1, 1)

    def get_snes_weights(self, fitness_scores: np.ndarray) -> np.ndarray:
        population_size = len(fitness_scores)
        ranks = np.argsort(np.argsort(-fitness_scores))
        weights = np.clip(
            np.log(population_size / 2 + 1) - np.log(ranks + 1),
            a_min=0.0,
            a_max=None,
        )
        weights /= np.sum(weights) + self.numerical_stability
        return (weights - 1 / population_size).reshape(-1, 1)

    def get_des_weights(self, fitness_scores: np.ndarray, temperature=12.5) -> np.ndarray:
        population_size = len(fitness_scores)
        ranks = np.argsort(np.argsort(-fitness_scores))
        centered_ranks = ranks / (population_size - 1) - 0.5
        sigmoid_vals = sigmoid(temperature * centered_ranks)
        weights = np.exp(-20 * sigmoid_vals)
        weights /= np.sum(weights) + self.numerical_stability
        return weights.reshape(-1, 1)
