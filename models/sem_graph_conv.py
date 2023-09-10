from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # passing 2 adj
        # self.adj, self.adj_s = adj
        # self.adj = self.adj_s
        self.adj = adj[0]
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        # print(torch.matmul(adj * M, h0))
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SemGraphConv2(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SemGraphConv2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        max_edges = 256
        self.e = nn.Parameter(torch.zeros(1, max_edges, dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, adj):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        m = (adj > 0)
        e = self.e[:, :len(m.nonzero())]

        adj = -9e15 * torch.ones_like(adj).to(input.device)
        # print('===================', adj.shape, m.shape, e.shape)
        adj[m] = e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SemGraphConv3(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(6, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # passing 2 adj
        # self.adj, self.adj_s = adj
        # self.adj = self.adj_s
        self.adj, self.adj_s, self.adj_m = adj


        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)


        self.m2 = (self.adj_s > 0)
        self.e2 = nn.Parameter(torch.zeros(1, len(self.m2.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e2.data, 1)

        self.m3 = (self.adj_m > 0)
        self.e3 = nn.Parameter(torch.zeros(1, len(self.m3.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e3.data, 1)

        self.weightage = nn.Parameter(torch.tensor(1.0))
        self.weightage2 = nn.Parameter(torch.tensor(1.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)


        h2 = torch.matmul(input, self.W[2])
        h3 = torch.matmul(input, self.W[3])
        adj = -9e15 * torch.ones_like(self.adj_s).to(input.device)
        adj[self.m2] = self.e2
        adj = F.softmax(adj, dim=1)
        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output2 = torch.matmul(adj * M, h2) + torch.matmul(adj * (1 - M), h3)

        h4 = torch.matmul(input, self.W[4])
        h5 = torch.matmul(input, self.W[5])
        adj = -9e15 * torch.ones_like(self.adj_m).to(input.device)
        # print('==================', self.m2.shape, self.e2.shape, self.m3.shape, self.e3.shape)
        adj[self.m3] = self.e3
        adj = F.softmax(adj, dim=1)
        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output3 = torch.matmul(adj * M, h4) + torch.matmul(adj * (1 - M), h5)

        output = output + output2 * self.weightage  + output3 * self.weightage2


        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    


class SemGraphConv4(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(6, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # passing 2 adj
        # self.adj, self.adj_s = adj
        # self.adj = self.adj_s
        self.adj, self.adj_s, self.adj_m = adj


        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)


        self.m2 = (self.adj_s > 0)
        self.e2 = nn.Parameter(torch.zeros(1, len(self.m2.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e2.data, 1)

        self.m3 = (self.adj_m > 0)
        self.e3 = nn.Parameter(torch.zeros(1, len(self.m3.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e3.data, 1)

        self.weightage = nn.Parameter(torch.tensor(1.0))
        self.weightage2 = nn.Parameter(torch.tensor(1.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        
        self.weightage_nn = nn.Sequential(
                        nn.Flatten(start_dim=1),  # Flattens the tensor starting from dimension 1
                        nn.Linear(16*in_features, 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Linear(64, 2)
                    )



    def forward(self, input):
        weightage = self.weightage_nn(input)
        weightage_1 = weightage[:, 0].unsqueeze(1)
        weightage_2 = weightage[:, 1].unsqueeze(1)


        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)


        h2 = torch.matmul(input, self.W[2])
        h3 = torch.matmul(input, self.W[3])
        adj = -9e15 * torch.ones_like(self.adj_s).to(input.device)
        adj[self.m2] = self.e2
        adj = F.softmax(adj, dim=1)
        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output2 = torch.matmul(adj * M, h2) + torch.matmul(adj * (1 - M), h3)

        h4 = torch.matmul(input, self.W[4])
        h5 = torch.matmul(input, self.W[5])
        adj = -9e15 * torch.ones_like(self.adj_m).to(input.device)
        # print('==================', self.m2.shape, self.e2.shape, self.m3.shape, self.e3.shape)
        adj[self.m3] = self.e3
        adj = F.softmax(adj, dim=1)
        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output3 = torch.matmul(adj * M, h4) + torch.matmul(adj * (1 - M), h5)

        output = output + output2 * weightage_1.unsqueeze(-1)  + output3 * weightage_2.unsqueeze(-1)


        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'






# changes for experimaeyt

class SemGraphConv5(nn.Module):
    """
    Semantic graph convolution layer
    """
    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv5, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # modify below for 3 hops
        self.W = nn.Parameter(torch.zeros(size=(6, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.adj, self.adj_s, self.adj_m, self.adj_l = adj


        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)


        self.m2 = (self.adj_s > 0)
        self.e2 = nn.Parameter(torch.zeros(1, len(self.m2.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e2.data, 1)

        # uncomment below for 3 hops
        self.m3 = (self.adj_m > 0)
        self.e3 = nn.Parameter(torch.zeros(1, len(self.m3.nonzero()), dtype=torch.float))
        nn.init.constant_(self.e3.data, 1)

        # uncomment below for 4 hops
        # self.m4 = (self.adj_l > 0)
        # self.e4 = nn.Parameter(torch.zeros(1, len(self.m4.nonzero()), dtype=torch.float))
        # nn.init.constant_(self.e4.data, 1)

        self.weightage = nn.Parameter(torch.tensor(1.0))
        self.weightage2 = nn.Parameter(torch.tensor(1.0))
        # self.weightage3 = nn.Parameter(torch.tensor(1.0))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        
        # self.weightage_nn = nn.Sequential(
        #                 nn.Flatten(start_dim=1),  # Flattens the tensor starting from dimension 1
        #                 nn.Linear(16*in_features, 64),
        #                 nn.LayerNorm(64),
        #                 nn.ReLU(),
        #                 nn.Linear(64, 2)
        #             )



    def forward(self, input):
        # weightage = self.weightage_nn(input)
        # weightage_1 = weightage[:, 0].unsqueeze(1)
        # weightage_2 = weightage[:, 1].unsqueeze(1)


        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)


        h2 = torch.matmul(input, self.W[2])
        h3 = torch.matmul(input, self.W[3])
        adj = -9e15 * torch.ones_like(self.adj_s).to(input.device)
        adj[self.m2] = self.e2
        adj = F.softmax(adj, dim=1)
        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output2 = torch.matmul(adj * M, h2) + torch.matmul(adj * (1 - M), h3)
        
        
        # uncomment below for 3 hops
        h4 = torch.matmul(input, self.W[4])
        h5 = torch.matmul(input, self.W[5])
        adj = -9e15 * torch.ones_like(self.adj_m).to(input.device)
        # print('==================', self.m2.shape, self.e2.shape, self.m3.shape, self.e3.shape)
        adj[self.m3] = self.e3
        adj = F.softmax(adj, dim=1)
        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output3 = torch.matmul(adj * M, h4) + torch.matmul(adj * (1 - M), h5)


        # uncomment below for 4 hops
        # h6 = torch.matmul(input, self.W[6])
        # h7 = torch.matmul(input, self.W[7])
        # adj = -9e15 * torch.ones_like(self.adj_l).to(input.device)
        # # print('==================', self.m2.shape, self.e2.shape, self.m3.shape, self.e3.shape)
        # adj[self.m4] = self.e4
        # adj = F.softmax(adj, dim=1)
        # M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        # output4 = torch.matmul(adj * M, h6) + torch.matmul(adj * (1 - M), h7)

        # uncomment below for 3 hops
        # output = output  + output2  * self.weightage  + output3 * self.weightage2 #+ output4 * self.weightage3
        # output = output * self.weightage + output2*self.weightage2 + output3*self.weightage3
        output = output + output2 * self.weightage + output3 * self.weightage2
        # output = output/2 + output2/2
        # output = output  + output2  * self.weightage
        # output = output + output2 * weightage_1.unsqueeze(-1)  + output3 * weightage_2.unsqueeze(-1)



        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'