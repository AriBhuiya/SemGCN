from torch import nn as nn
import torch
from torch.nn import functional as f
from torch import Tensor
from models.sem_graph_conv import SemGraphConv

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
        # self.feed_forward = Residual(
        #     feed_forward(dim_model, dim_feedforward),
        #     dimension=dim_model,
        #     dropout=dropout,
        # )

        self.gcn = Residual(
            SemGraphConv(dim_model, dim_model, adj),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.gcn(src)