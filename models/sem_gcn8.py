from __future__ import absolute_import
from functools import reduce
import torch.nn as nn
from models.sem_graph_conv import SemGraphConv
from models.graph_non_local import GraphNonLocal
from models.myarc import GraphConvolution, TransformerEncoder, ResidualConverter, JustAttentionLayer, ModifiedTransformerEncoderLayer, GraphFormerLikeAttention


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        # self.gconv = GraphConvolution(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()
        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        return residual + out


class _GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(_GraphNonLocal, self).__init__()
        self.nonlocal1 = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.nonlocal1(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class SemGCN8(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN8, self).__init__()
        num_layers = 2
        num_heads = 4
        dim_model=128

        adj_n, adj_s = adj

        self.inputblock = _GraphConv(adj_s, coords_dim[0], dim_model, p_dropout)
        self.semgcn_res_blocks = _ResGraphConv(adj_s, dim_model, dim_model, dim_model, p_dropout)
        self.attn_block = GraphFormerLikeAttention(num_heads, 128, dropout = 0.1)
        self.output_block = SemGraphConv(dim_model, coords_dim[1], adj_s, bias=True )

        middle_layers = []

        for layer in range(num_layers):
            middle_layers.append(self.attn_block)
            middle_layers.append(self.semgcn_res_blocks)

        self.middle_block = nn.Sequential(*middle_layers)
       


    def forward(self, x):
        # print(x.shape)
        # shape here is 64, 16, 2
        out = self.inputblock(x)
        out = self.middle_block(out)
        out = self.attn_block(out)
        out = self.output_block(out)

        return out
