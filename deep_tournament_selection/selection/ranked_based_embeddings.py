from torch import nn


class RankedBasedEmbedding(nn.Module):
    def __init__(self, latent_dim, max_selection_sequence_length=256):
        super().__init__()
        self.pos_emb = nn.Embedding(max_selection_sequence_length, latent_dim)

    def forward(self, x, population_order):
        """
        Input:
            x : LongTensor [batch, n_sentences, sentence_length]
        Output:
            embeddings : [batch, n_sentences, latent_dim]
        """

        return self.pos_emb(population_order)
