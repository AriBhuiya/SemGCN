from __future__ import absolute_import
from functools import reduce
import torch.nn as nn
from models.sem_graph_conv import SemGraphConv
from models.graph_non_local import GraphNonLocal
from models.myarc import GraphConvolution, GraphConvolution2, TransformerEncoder, ResidualConverter, JustAttentionLayer, ModifiedTransformerEncoderLayer


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        # self.gconv = GraphConvolution2(adj, input_dim, output_dim)
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
        # x = self.attn(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()
        adj_n, adj_s = adj
        self.gconv1 = _GraphConv(adj_n, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj_n, hid_dim, output_dim, p_dropout)

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


class SemGCN7(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN7, self).__init__()
        num_layers = 4
        # Remove nonlocal layer
        nodes_group = None
        adj_n, adj_s = adj


        _gconv_input = [_GraphConv(adj_n, coords_dim[0], hid_dim, p_dropout=p_dropout)]
        _gconv_layers = []

        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
            _gconv_layers.append(_GraphConv(adj_s, hid_dim, hid_dim, p_dropout=p_dropout))
            # _gconv_layers.append(TransformerEncoder(num_layers=1, dim_model=128, num_heads=4,
            #                                       dim_feedforward=256,
            #                                       dropout=0.1, ))
            # _gconv_layers.append(JustAttentionLayer(dim_model=128, num_heads=4,
            #                                       dim_feedforward=256,
            #                                       dropout=0.1, ))ModifiedTransformerEncoderLayer
            # _gconv_layers.append(ModifiedTransformerEncoderLayer(adj, dim_model=128, num_heads=4,
            #                                       dim_feedforward=256,
            #                                       dropout=0.1, ))
            # _gconv_input.append(_GraphNonLocal(hid_dim, grouped_order, restored_order, group_size))
                

        self.gconv_input = nn.Sequential(*_gconv_input)

        # uncomment below if you want no residual
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        # self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.gconv_output = SemGraphConv(hid_dim, 3, adj_n )
        # self.gconv_output = GraphConvolution(hid_dim, 8, adj)

    def forward(self, x):
        # print(x.shape)
        # shape here is 64, 16, 2
        out = self.gconv_input(x)
        out = self.gconv_layers(out)
        # residual_outputs = []
        # for layer in self.gconv_layers:
        #     residual_outputs.append(self.res_graph_to_transformer_linear(out))
        #     out = layer(out)

        out = self.gconv_output(out)
        

        return out
