import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """
    Critic Network (Value Function Estimator) for Actor-Critic algorithm.
    
    Takes embedded population state and outputs a scalar value V(s) representing
    the expected future reward from that state.
    
    This is used in Actor-Critic to compute advantages via TD error:
        A(s,a) = r + γ*V(s') - V(s)
    """
    
    def __init__(self, d_model=32, hidden_dim=128, pooling='mean', dropout=0.1):
        """
        Args:
            d_model: Dimension of population embeddings (should match actor's latent_dim)
            hidden_dim: Hidden layer dimension for MLP
            pooling: Pooling strategy for population ('mean', 'max', or 'attention')
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.d_model = d_model
        self.pooling = pooling
        
        # If using attention pooling, create attention layer
        if pooling == 'attention':
            self.attention_query = nn.Linear(d_model, d_model)
            self.attention_key = nn.Linear(d_model, d_model)
            self.attention_scale = d_model ** 0.5
        
        # MLP to estimate value from pooled representation
        self.value_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Output single scalar value
        )
        
    def forward(self, embedded_population):
        """
        Args:
            embedded_population: [batch, n_individuals, d_model]
                Population embeddings from PopulationToVecTransformer
                
        Returns:
            state_value: [batch, 1] - Estimated value V(s) for each state in batch
        """
        batch_size, n_individuals, d_model = embedded_population.shape
        
        # Pool across population dimension to get state representation
        if self.pooling == 'mean':
            # Simple mean pooling
            state_repr = embedded_population.mean(dim=1)  # [batch, d_model]
            
        elif self.pooling == 'max':
            # Max pooling
            state_repr = embedded_population.max(dim=1)[0]  # [batch, d_model]
            
        elif self.pooling == 'attention':
            # Attention-based pooling (learn to weight individuals)
            # Query is mean of population
            query = self.attention_query(embedded_population.mean(dim=1, keepdim=True))  # [batch, 1, d_model]
            keys = self.attention_key(embedded_population)  # [batch, n_individuals, d_model]
            
            # Compute attention scores
            scores = torch.bmm(query, keys.transpose(1, 2)) / self.attention_scale  # [batch, 1, n_individuals]
            weights = torch.softmax(scores, dim=-1)  # [batch, 1, n_individuals]
            
            # Weighted sum of embeddings
            state_repr = torch.bmm(weights, embedded_population).squeeze(1)  # [batch, d_model]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Estimate state value through MLP
        state_value = self.value_head(state_repr)  # [batch, 1]
        
        return state_value
