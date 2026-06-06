import torch
import torch.nn as nn
import torch.nn.functional as F


class RecombinationAttention(nn.Module):
    """
    Implements: w_t = softmax( softmax(QK^T / sqrt(Dk)) * (Ft W_V) )
    where Ft is the input of shape [B, N, D].
    """

    def __init__(self, embed_dim, key_dim):
        super().__init__()
        self.W_Q = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, key_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, 1, bias=False)  # produces scalar per token
        self.scale = key_dim ** 0.5

    def forward(self, Ft):
        """
        Ft: Tensor of shape [B, N, D]
        returns: w_t of shape [B, N] — a probability distribution
        """

        # 1) Compute Q, K
        Q = self.W_Q(Ft)  # [B, N, Dk]
        K = self.W_K(Ft)  # [B, N, Dk]

        # 2) Attention logits: QK^T / sqrt(Dk)
        attn_logits = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # [B, N, N]
        A = F.softmax(attn_logits, dim=-1)  # self-attention over tokens

        # 3) Compute V = Ft W_V -> scalar per token
        V = self.W_V(Ft).squeeze(-1)  # [B, N]

        # 4) Multiply A @ V
        w_raw = torch.matmul(A, V.unsqueeze(-1)).squeeze(-1)  # [B, N]

        # 5) Final probability distribution
        w_t = F.softmax(w_raw, dim=-1)  # [B, N]

        return w_t
