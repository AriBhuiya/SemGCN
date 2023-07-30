from __future__ import absolute_import
from functools import reduce
import torch.nn as nn
from models.sem_graph_conv import SemGraphConv
from models.graph_non_local import GraphNonLocal
from models.myarc import GraphConvolution, TransformerEncoder, ResidualConverter, JustAttentionLayer, ModifiedTransformerEncoderLayer, TransformerEncoderLayerWithMixer
from models.mlp_mixer_pytorch.mlp_mixer_pytorch import MLPMixer

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


class SemGCN4(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(SemGCN4, self).__init__()
        num_layers = 4
        dim_model = 256
        self.dim_model = dim_model
        dim_feedforward = 4096
        self.dim_feedforward = dim_feedforward

        self.input_layer = nn.Sequential(
            nn.Linear(coords_dim[0], dim_model),
            nn.LayerNorm(dim_model),
            nn.ReLU())

        self.mixer = MLPMixer(
            image_size= int(dim_model ** 0.5),
            channels=16,
            patch_size=16,
            dim=dim_feedforward,
            depth=5,
            num_classes=1000
        )

        self.output = nn.Linear(dim_feedforward // 16, 3)


    def forward(self, x):
        # print(x.shape)
        # shape here is 64, 16, 2
        out_layer1 = self.input_layer(x) # 64X16X4
        # print(out.shape)
        out = out_layer1.reshape(-1, 16, int(self.dim_model**0.5), int(self.dim_model**0.5))
        # print(out.shape)
        out = self.mixer(out) # 64 X 1024
        # print(out.shape)
        out = out.reshape(-1, 16, self.dim_feedforward // 16)
        out +=out_layer1
        out = self.output(out)
        # rearrange to 64X 16 X 3
        # print(out.shape)

        return out
