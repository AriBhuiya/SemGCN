from torch import nn as nn
import torch
from torch.nn import functional as f
from torch import Tensor
from models.sem_graph_conv import SemGraphConv3, SemGraphConv5
from models.mlp_mixer_pytorch.mlp_mixer_pytorch import MLPMixer

class ResidualConverter(nn.Module):
    def __init__(self, in_dims, out_dims, extras=True):
        super(ResidualConverter, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=16, stride = 16)

    def forward(self, x):
        return self.pool(x)

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, adj: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.adj = adj.to('cuda')

    def forward(self, x):
        # 16X16 ......... 64, 16, 2 ...
        x = torch.matmul(self.adj, x)
        # print('3-------------', x.shape)
        x = self.linear(x)
        # print('4-------------', x.shape)
        return x


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
        # print('1------------------', dim_in, dim_q, dim_k) # 16, 4 , 4

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # print('0------------------', query.shape, key.shape, value.shape) # 64, 16, 8 all three
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        return scaled_dot_product_attention(q, k, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class Residual2(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        # return x + self.dropout(sublayer(self.norm(x)))
        # return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))
        return tensors[0] + self.dropout(self.sublayer(self.norm(*tensors)))

class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            num_layers: int = 6,
            dim_model: int = 8,
            num_heads: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor, residuals:list = None) -> Tensor:
        # seq_len, dimension = src.size(1), src.size(2)
        for i, layer in enumerate(self.layers):
            # r = residuals[len(residuals) - i - 1]
            # src = layer(src + r)
            src = layer(src)
        return src



class JustAttentionLayer(nn.Module):
    def __init__(
            self,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return src


class ModifiedTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            adj,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )

        self.gcn = Residual(
            SemGraphConv5(dim_model, dim_model, adj),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.gcn(src)



class TransformerEncoderLayerWithMixer(nn.Module):
    def __init__(
            self,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)

        self.dim_model = dim_model
        

        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = MLPMixer(
                image_size = 16,
                channels = 16,
                patch_size = 2,
                dim = dim_feedforward,
                depth = 2,
                num_classes = 1000
            )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        src = src.reshape(-1, 16, 16, 16)
        return self.feed_forward(src)


class GraphConvolution2(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, dropout = 0.1, is_last=False):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        # self.adj = adj.to('cuda')
        self.dropout = dropout
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.einsum('bik,kj->bij', x, self.weight)  # Modified here
        output = torch.einsum('ii,bij->bij', adj, support)  # Modified here

        if self.bias is not None:
            return output + self.bias.unsqueeze(0)  # Modified here to add an extra dimension
        else:
            return output
        
        
    


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphFormerLikeAttention(nn.Module):
    def __init__(self, num_heads, dim_model:int, dropout = 0.1):
        super(GraphFormerLikeAttention, self).__init__()
        self.norm = nn.LayerNorm(dim_model)
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attn = JustAttentionLayer(dim_model = dim_model, num_heads = num_heads, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        out = self.norm(x)
        out = self.attn(out)
        out = self.dropout(out)
        return x + out