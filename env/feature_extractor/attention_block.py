import torch
import torch.nn.functional as F
from torch import nn
import math


# implements skip-connection module / short-cut module
class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """
    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def get_PE(self, seq_len):
        return self.encoding[:seq_len, :]

# implements Normalization module
class Normalization(nn.Module):

    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.normalization(x)


# implements the original Multi-head Self-Attention module
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        # todo randn?rand
        self.W_query = nn.Parameter(torch.randn(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.randn(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.randn(n_heads, input_dim, val_dim))

        # self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.randn(n_heads, val_dim, embed_dim))
            # self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

    def forward(self, q, h=None):
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, batch_size, n_query, key_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)

        # Calculate keys and values (n_heads, batch_size, graph_size, key_size or val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        attn = F.softmax(compatibility, dim=-1)  # (n_heads, batch_size, n_query, graph_size)

        heads = torch.matmul(attn, V)  # (n_heads, batch_size, n_query, val_size)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            # (batch_size * n_query, n_heads * val_size)
            self.W_out.view(-1, self.embed_dim)  # (n_heads * val_size, embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


# implements the encoder
class MultiHeadEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization
    ):
        super(MultiHeadEncoder, self).__init__()
        self.MHA_sublayer = MultiHeadAttentionsubLayer(
            n_heads,
            embed_dim,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFandNormsubLayer(
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input):
        out = self.MHA_sublayer(input)
        return self.FFandNorm_sublayer(out)


# implements the encoder (DAC-Att sublayer)
class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            normalization,
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()

        self.MHA = MultiHeadAttention(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.Norm = Normalization(embed_dim)

    def forward(self, x):
        # Attention
        out = self.MHA(x)

        # Residual connection and Normalization
        return self.Norm(out + x)


# implements the encoder (FFN sublayer)
class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            embed_dim,
            feed_forward_hidden,
            normalization,
    ):
        super(FFandNormsubLayer, self).__init__()

        self.FF = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_hidden, embed_dim)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)

        self.Norm = Normalization(embed_dim)

    def forward(self, x):
        # FF
        out = self.FF(x)

        # Residual connection and Normalization
        return self.Norm(out + x)


class EmbeddingNet(nn.Module):

    def __init__(self,
                 node_dim,
                 embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias=False)

    def forward(self, x):
        h_em = self.embedder(x)
        return h_em