from torch import nn
import torch
import torch.nn.functional as F

adj = [[0, 1, 0],
       [1, 0, 1],
       [0, 1, 0]]

adj = torch.tensor(adj)
print('Adjacency Matrix', adj)


m = (adj > 0)
e = nn.Parameter(torch.zeros(1, len(m.nonzero()), dtype=torch.float))
nn.init.constant_(e.data, 0.51)
print('e', e)
adj = -9e15 * torch.ones_like(adj)
adj[m] = e
print('Modified Adj', adj)
# print(adj)
adj = F.softmax(adj, dim=1)
print(adj)

M = torch.eye(adj.size(0), dtype=torch.float)
print(M)

inp = torch.rand(3,2)
W = torch.ones((2, 2,3))
h0 = torch.matmul(inp, W[0])
h1 = torch.matmul(inp, W[1])

print('h0 ==> ', h0.shape, h0)
print('h1 ==> ', h1.shape, h1)

# print(W.shape)
print('adj X M', adj * M)
h0 = torch.ones(3,3)
output = torch.matmul(adj * M, h0) #+ torch.matmul(adj * (1 - M), h1)
print('output',output, output.shape)