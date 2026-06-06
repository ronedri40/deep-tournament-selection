import torch


class TrajectoryTracker:
    def __init__(self, device='cpu'):
        self.device = device
        self.pending_trajectories_log_probs = []
        self.trajectory_probabilities = []
        self.trajectory_rewards = []
        self.generation_tracked_fitness_scores = []

    def log_pending_logprob(self, log_prob):
        self.pending_trajectories_log_probs.append(log_prob)

    def match_rewards_to_pending_trajectories(self, rewards):
        self.log_trajectory_to_memory(self.pending_trajectories_log_probs.pop(0), rewards=rewards)

    def detatch_pending_log_probs(self):
        # Detach pending trajectories log probs to prevent inplace modification error
        # (stale gradients from previous optimization steps)
        for i in range(len(self.pending_trajectories_log_probs)):
            self.pending_trajectories_log_probs[i] = self.pending_trajectories_log_probs[i].detach()

    def log_trajectory_to_memory(self, trajectory_proba, rewards):
        self.trajectory_probabilities.append(trajectory_proba)
        self.trajectory_rewards.append(torch.tensor(rewards, device=self.device))
