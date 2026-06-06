class RewardBaseline:
    def __init__(self, beta=0.95):
        self.beta = beta
        self.value = None

    def update(self, rewards):
        mean_r = rewards.mean().item()
        if self.value is None:
            self.value = mean_r
        else:
            self.value = self.beta * self.value + (1 - self.beta) * mean_r

    def advantage(self, rewards):
        return rewards - self.value
