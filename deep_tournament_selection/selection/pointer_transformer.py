import torch
import torch.nn as nn


class PointerTransformer(nn.Module):
    """
    Transformer Pointer Network:
    Input  : [batch, n_sent, d]
    Output : pointer sequence [batch, j]
    """

    def __init__(self, d_model=256, n_heads=4, n_layers=3,
                 pointer_len=5, dropout=0.1, max_pointers=128):
        super().__init__()
        self.pointer_len = pointer_len
        self.d_model = d_model

        # Decoder: autoregressive pointer generator
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Embedding for pointer indices fed as tokens to decoder
        self.pos_emb = nn.Embedding(max_pointers, d_model)  # sentence indices

        # Start token for decoder
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, sent_embeds, pointer_len=None, teacher_forcing=None, population_order=None):
        """
        sent_embeds: [batch, n_sent, d_model]
        teacher_forcing: optional ground-truth pointer sequences [batch, j]
        Returns:
            logits: [batch, j, n_sent] (pointer distribution)
        """
        if pointer_len is None:
            pointer_len = self.pointer_len

        batch, n_sent, d = sent_embeds.shape

        # ---- Encode input sentences ----
        H = sent_embeds  # [B, n_sent, d_model]
        if population_order is not None:
            ranked_based_embeddings = self.pos_emb(population_order)
            H += ranked_based_embeddings

        # ---- Prepare decoder inputs ----
        dec_in = self.start_token.expand(batch, 1, self.d_model)
        logits = []
        sampled_indexes = []
        batch_indices = torch.arange(batch)

        for t in range(pointer_len):
            # decode step
            dec_out = self.decoder(dec_in, H)[:, -1:, :]  # last step hidden

            # pointer logits = attention between dec_out and encoder outputs
            # dec_out: [B,1,d], H: [B,n_sent,d]
            # dot product attention
            attn_logits = torch.bmm(dec_out, H.transpose(1, 2)).squeeze(1)
            # shape: [B, n_sent]

            logits.append(attn_logits)

            if teacher_forcing is None:
                # categorical sampling
                next_idx = torch.distributions.Categorical(logits=attn_logits).sample()
            else:
                next_idx = teacher_forcing[:, t]

            sampled_indexes.append(next_idx)
            # Convert selected sentence embedding into decoder token
            # next_emb = self.idx_embed(next_idx).unsqueeze(1)
            next_emb = sent_embeds[batch_indices, next_idx, :]
            next_emb = next_emb.unsqueeze(1)

            dec_in = torch.cat([dec_in, next_emb], dim=1)

        logits = torch.stack(logits, dim=1)
        sampled_indexes = torch.stack(sampled_indexes, dim=1)
        return logits, sampled_indexes
