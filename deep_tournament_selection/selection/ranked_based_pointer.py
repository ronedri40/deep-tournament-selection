from torch import nn
import torch


class RankedBasedPointer(nn.Module):
    def __init__(self, pointer_len, d_model, log_stability=1e-12, max_selection_sequence_length=32):
        super().__init__()
        self.pointer_len = pointer_len
        self.d_model = d_model
        self.log_stability = log_stability
        self.pos_emb = nn.Embedding(max_selection_sequence_length, d_model)

    def forward(self, sent_embeds, pointer_len=None, teacher_forcing=None, population_order=None):
        """
        sent_embeds: [B, N, d_model]
        teacher_forcing:[B, pointer_len] or None
        population_order:[B, N]
        Returns:
        logits: [B, pointer_len, N]
        sampled_indexes:[B, pointer_len]
        """
        assert population_order is not None, "Population order is required for ranked-based pointer"

        if pointer_len is None:
            pointer_len = self.pointer_len

        # ---- Encode sentences ----
        H = sent_embeds
        ranked_based_embeddings = self.pos_emb(population_order)

        logits = []
        sampled_idxs = []
        w_t = (H * ranked_based_embeddings).sum(dim=-1)  # dot product of the last
        # dimension
        log_probs = torch.log_softmax(w_t, dim=-1)

        for t in range(pointer_len):

            # ---- Self-attention pointer distribution ----
            # RecombinationAttention returns w_t = softmax(...)

            logits.append(log_probs)

            # ---- Sample or teacher-force ----
            if teacher_forcing is None:
                idx = torch.distributions.Categorical(logits=w_t).sample()
            else:
                idx = teacher_forcing[:, t]

            sampled_idxs.append(idx)

        logits = torch.stack(logits, dim=1)  # [B, pointer_len, N]
        sampled_idxs = torch.stack(sampled_idxs, dim=1)  # [B, pointer_len]

        return logits, sampled_idxs
