import torch, time
from .attention_block import MultiHeadEncoder, EmbeddingNet, PositionalEncoding
from torch import nn
import numpy as  np


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


# neural feature extractor
class Feature_Extractor(nn.Module):
    def __init__(self,
                 node_dim=3,    # todo: node_dim should be win * 2
                 hidden_dim=16,
                 n_heads=1,
                 ffh=16,
                 n_layers=1,
                 use_positional_encoding = True,
                 is_mlp = False,
                 device='cpu',
                 ):
        super(Feature_Extractor, self).__init__()
        # bs * dim * pop_size * 2  ->  bs * dim * pop_size * hidden_dim
        self.device = device
        self.embedder = EmbeddingNet(node_dim=node_dim,embedding_dim=hidden_dim).to(device)
        # positional_encoding, we only add PE at before the dimension encoder
        # since each dimension should be regarded as different parts, their orders matter.
        self.is_mlp = is_mlp
        if not self.is_mlp:
            self.use_PE = use_positional_encoding
            if self.use_PE:
                self.position_encoder = PositionalEncoding(hidden_dim,512).to(device)

            # bs * dim * pop_size * hidden_dim -> bs * dim * pop_size * hidden_dim
            self.dimension_encoder = mySequential(*(MultiHeadEncoder(n_heads=n_heads,
                                                    embed_dim=hidden_dim,
                                                    feed_forward_hidden=ffh,
                                                    normalization='n2')
                                                    for _ in range(n_layers))).to(device)
            # the gradients are predicted by attn each dimension of a single individual.
            # bs * pop_size * dim * 128 -> bs * pop_size * dim * 128
            self.individual_encoder = mySequential(*(MultiHeadEncoder(n_heads=n_heads,
                                                    embed_dim=hidden_dim,
                                                    feed_forward_hidden=ffh,
                                                    normalization='n1')
                                                    for _ in range(n_layers))).to(device)
        else:
            self.mlp = nn.Linear(hidden_dim, hidden_dim).to(device)
            self.acti = nn.ReLU()
        # print('------------------------------------------------------------------')
        # print('The feature extractor has been successfully initialized...')
        # print(self.get_parameter_number())
        # print('------------------------------------------------------------------')
        self.is_train = False
        self.to(device)


    def set_on_train(self):
        self.is_train = True
    
    def set_off_train(self):
        self.is_train = False

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # return 'Total {} parameters, of which {} are trainable.'.format(total_num, trainable_num)
        return total_num

    # todo: add a dimension representing the window
    # xs(bs * pop_size * dim) are the candidates,
    # ys(bs * pop_size) are the corresponding obj values, they are np.array
    def forward(self, xs, ys,):
        # if self.is_train:
        return self._run(xs, ys,)
        # else:
        #     with torch.no_grad():
        #         return self._run(xs, ys,)
    
    def _run(self, xs, ys, ):
        # return torch.rand(32)
        _,d = xs.shape
        # xs = (xs - p.lb) / (p.ub - p.lb)
        xs = xs[None,:,:] # 1 * n * d
        try:
            ys = ys[None,:]
        except IndexError:
            print(ys.shape, ys)
            raise IndexError
        y_ = (ys - ys.min(-1)[:, None]) / (ys.max(-1)[:, None] - ys.min(-1)[:, None] + 1e-12)
        e = np.log10(ys) / 32
        y_ = ys / np.power(10, np.ceil(e*32))
        ys = y_[:, :, None]
        es = e[:, :, None]

        # pre-processing data as the form of per_dimension_feature bs * d * n * 2
        a_x = xs[:, :, :, None]
        a_y = np.repeat(ys, d, -1)[:, :, :, None]
        a_e = np.repeat(es, d, -1)[:, :, :, None]
        raw_feature = np.concatenate([a_x, a_y, a_e], axis=-1).transpose((0, 2, 1, 3)) # bs * d * n * 2
        raw_feature = torch.tensor(raw_feature,dtype=torch.float32).to(self.device)
        h_ij = self.embedder(raw_feature) # bs * dim * pop_size * 2  ->  bs * dim * pop_size * hd
        bs, dim, pop_size, node_dim = h_ij.shape
        # resize h_ij as (bs*dim) * pop_size * hd
        h_ij = h_ij.view(-1,pop_size,node_dim)
        if not self.is_mlp:
            o_ij = self.dimension_encoder(h_ij).view(bs,dim,pop_size,node_dim) # bs * dim * pop_size * 128 -> bs * dim * pop_size * hd
            # resize o_ij, to make each dimension of the single individual into as a group
            o_i = o_ij.permute(0,2,1,3).contiguous().view(-1,dim,node_dim)
            if self.use_PE:
                o_i = o_i + self.position_encoder.get_PE(dim) * 0.5
            out = self.individual_encoder(o_i).view(bs,pop_size,dim,node_dim) # (bs * pop_size) * dim * 128 -> bs * pop_size * dim * hidden_dim
            # bs * pop_size * hidden_dim
            out = torch.mean(out,-2)
            return torch.mean(out, 1).squeeze()
        else:
            out = self.mlp(self.acti(h_ij)).view(bs,dim,pop_size,node_dim)
            out = torch.mean(out, -3)
            return out

