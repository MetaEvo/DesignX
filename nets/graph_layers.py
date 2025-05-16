import os
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
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x, pe_id=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if pe_id is None:
            x = x + self.pe
        else:
            for bid in range(x.shape[0]):
                x[bid][:len(pe_id[bid])] = x[bid][:len(pe_id[bid])] + self.pe[0][pe_id[bid]]
        # return self.dropout(x)
        return x
    
class PositionalEncodingSin(nn.Module):

    def __init__(self, d_model, seq_len):
        super().__init__()
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, pe_id=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if pe_id is None:
            x = x + self.pe[0,:x.shape[1]]
        else:
            for bid in range(x.shape[0]):
                x[bid][:len(pe_id[bid])] = x[bid][:len(pe_id[bid])] + self.pe[0][pe_id[bid]]
        return x
    
class MLP(torch.nn.Module):
    def __init__(self,
                input_dim ,
                mid_dim1,
                mid_dim2,
                output_dim,
                p=0
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, mid_dim1)
        self.fc2 = torch.nn.Linear(mid_dim1,mid_dim2)
        self.fc3=torch.nn.Linear(mid_dim2,output_dim)
        self.activation1 = nn.LeakyReLU(0.3)

    def forward(self, in_):
        result = self.activation1(self.fc1(in_))
        result = self.activation1(self.fc2(result))
        result = self.fc3(result).squeeze(-1)
        return result

class MLP3(torch.nn.Module):
    def __init__(self,
                input_dim ,
                # mid_dim,
                output_dim,
    ):
        super(MLP3, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_dim)
        # self.fc2=torch.nn.Linear(mid_dim,output_dim)
        self.activation1 = nn.LeakyReLU(0.3)

    def forward(self, in_):
        result = self.activation1(self.fc1(in_))
        # result = self.fc2(result)
        return result

# implements MLP module for actor
class MLP_for_actor(torch.nn.Module):
    def __init__(self,
                 input_dim=64,
                 embedding_dim=16,
                 output_dim=1,
    ):
        super(MLP_for_actor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, embedding_dim)
        self.fc2 = torch.nn.Linear(embedding_dim, output_dim)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, in_):
        result = self.ReLU(self.fc1(in_))
        result = self.fc2(result).squeeze(-1)
        return result


# implements Normalization module
class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalization = normalization

        if not self.normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.normalization == 'layer':
            return (input - input.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(input.var((1, 2)).view(-1, 1, 1) + 1e-05)

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


# implements the encoder for Critic net
class MultiHeadAttentionLayerforCritic(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionLayerforCritic, self).__init__(
            SkipConnection(  # Attn & Residual
                    MultiHeadAttention(
                        n_heads,
                        input_dim=embed_dim,
                        embed_dim=embed_dim
                    )                
            ),
            Normalization(embed_dim, normalization),  # Norm
            SkipConnection(  # FFN & Residual
                    nn.Sequential(
                        nn.Linear(embed_dim, feed_forward_hidden),
                        nn.ReLU(inplace = True),
                        nn.Linear(feed_forward_hidden, embed_dim)
                    ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)   # Norm
        ) 



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

        # self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, q_lengths=None):

        batch_size, graph_size, input_dim = q.size()
        
        mask = torch.ones_like(q).to(q.device)
        if q_lengths is not None:  # batch_size
            i = torch.arange(graph_size).repeat(batch_size).view(batch_size, graph_size)
            lengths = q_lengths.view(-1, 1).repeat(1, graph_size)
            mask[i >= lengths] = 0

        q = q * mask

        h = q  # compute self-attention

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

        # Calculate compatibility (n_heads, batch_size, n_comp, n_comp)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        
        # Mask Softmax in colunms
        maskinf = torch.zeros_like(compatibility).to(compatibility.device)
        if q_lengths is not None:
            i = torch.arange(graph_size).view(1, 1, 1, graph_size).repeat(self.n_heads, batch_size, graph_size, 1).to(compatibility.device)  # 0 ~ graphsize each row
            lengths = q_lengths.view(1, -1, 1, 1).repeat(self.n_heads, 1, graph_size, graph_size).to(compatibility.device)
            maskinf[i >= lengths] = -torch.inf
        
        # compat_arange = graph_size - torch.matmul(torch.triu(torch.ones(graph_size,graph_size)), torch.tril(torch.ones(graph_size,graph_size)))
        # compat_arange = compat_arange.view(1, 1, graph_size, graph_size).repeat(self.n_heads, batch_size, 1, 1)
        # compat_length = q_lengths.view(1, -1, 1, 1).repeat(self.n_heads, 1, graph_size, graph_size)
        # compatibility[compat_arange >= compat_length] = -torch.inf
            
        compatibility += maskinf

        attn = F.softmax(compatibility, dim=-1)   # (n_heads, batch_size, n_query, graph_size)

        # Mask attn in rows
        mask01 = torch.ones_like(attn).to(attn.device)
        if q_lengths is not None:
            i = torch.arange(graph_size).view(1, 1, graph_size, 1).repeat(self.n_heads, batch_size, 1, graph_size).to(attn.device)  # 0 ~ graphsize each colunm
            lengths = q_lengths.view(1, -1, 1, 1).repeat(self.n_heads, 1, graph_size, graph_size).to(attn.device)
            mask01[i >= lengths] = 0
        
        # attn[compat_arange >= compat_length] = 0
        attn_ = attn * mask01
       
        heads = torch.matmul(attn_, V)  # (n_heads, batch_size, n_query, val_size)
        # if torch.sum(torch.isnan(heads)) > 0 or torch.sum(torch.isinf(heads)) > 0:
        #     print('heads')
        #     print(heads)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),  # (batch_size * n_query, n_heads * val_size)
            self.W_out.view(-1, self.embed_dim)  # (n_heads * val_size, embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out

  
# implements the multi-head compatibility layer
class MultiHeadCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadCompat, self).__init__()
    
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(1 * key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        # self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h = None, mask=None):
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)  
        K = torch.matmul(hflat, self.W_key).view(shp)   

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = torch.matmul(Q, K.transpose(2, 3))
        
        return self.norm_factor * compatibility


# implements the encoder
class MultiHeadEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer'
    ):
        super(MultiHeadEncoder, self).__init__()
        self.MHA_sublayer = MultiHeadAttentionsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                )
        
        self.FFandNorm_sublayer = FFandNormsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                )
        
    def forward(self, input, q_length):
        out = self.MHA_sublayer(input, q_length)
        return self.FFandNorm_sublayer(out)

# implements the encoder (DAC-Att sublayer)   
class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()
        
        self.MHA = MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
        
        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input, q_length):
        # Attention
        out = self.MHA(input, q_length)
        
        # Residual connection and Normalization
        return self.Norm(out + input)


# implements the encoder (FFN sublayer)   
class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayer, self).__init__()
        
        self.FF = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        
        self.Norm = Normalization(embed_dim, normalization)
    
    def forward(self, input):
    
        # FF
        out = self.FF(input)
        
        # Residual connection and Normalization
        return self.Norm(out + input)


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
    
