from __future__ import absolute_import
from functools import reduce
import torch.nn as nn
from models.sem_graph_conv import SemGraphConv, SemGraphConv2
from models.graph_non_local import GraphNonLocal
from models.new_arc import AttentiveGCNLayer, MultiHeadAttention, AttentiveAdjacency
from models.myarc import GraphConvolution2

class _GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv2(input_dim, output_dim)
        # self.gconv = GraphConvolution2(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, adj):
        x = self.gconv(x, adj).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()
        self.gconv1 = SemGraphConv2(input_dim, hid_dim, p_dropout)
        self.gconv2 = SemGraphConv2(hid_dim, output_dim, p_dropout)

    def forward(self, x, adj):
        residual = x
        out = self.gconv1(x, adj)
        out = self.gconv2(out, adj)
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



class SemGCN9(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN9, self).__init__()
        num_layers = 4
        num_heads = 4
        num_joints = 16

        self.adj_n, self.adj_s = adj
        self.adj_n, self.adj_s = self.adj_n.to('cuda'), self.adj_s.to('cuda')
        self.adj_s = self.adj_n

        self.inputblock = _GraphConv(coords_dim[0], hid_dim, p_dropout)
        
        self.attention_input = AttentiveAdjacency(num_heads, num_joints)

        self.attention_mid = AttentiveAdjacency(num_heads, num_joints)
        # Using attention weight sharing

        self.hidden_layers_gcn = nn.ModuleList([
            _ResGraphConv(hid_dim, hid_dim, hid_dim, p_dropout) for layer in range(num_layers)
        ])

        self.attention_layers = nn.ModuleList([
            AttentiveAdjacency(num_heads, num_joints) for layer in range(3)
        ])


        self.output_block = SemGraphConv2(hid_dim, coords_dim[1])

    def forward(self, x):
        # shape here is 64, 16
        # Input block
        adjusted_adj_s = self.attention_input(self.adj_s)
        # adjusted_adj_s = self.adj_n
        for layer_attention in self.attention_layers:
            adjusted_adj_s = layer_attention(adjusted_adj_s)


        out = self.inputblock(x, adjusted_adj_s)

        for layer_gcn in self.hidden_layers_gcn:
            out = layer_gcn(out, adjusted_adj_s)

        # adjusted_adj_s = self.attention_mid(out, self.adj_s)
        out = self.output_block(out, adjusted_adj_s)

        return out
