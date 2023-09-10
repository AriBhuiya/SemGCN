from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp
from models.floyd_warshall import floyd_warshall_sparse

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# def adj_mx_from_edges(num_pts, edges, sparse=True):
#     edges = np.array(edges, dtype=np.int32)
#     data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
#     adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

#     # build symmetric adjacency matrix
#     adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
#     # print('----------------------------------', type(adj_mx))
#     # print('sparse adj_mx', adj_mx)
#     adj_mx_s = floyd_warshall_sparse(adj_mx)
#     # print('floyd adj_mx', adj_mx)
#     # ''' The below line is the normalization operation given in every literature'''
#     adj_mx_s = normalize(adj_mx_s + sp.eye(adj_mx_s.shape[0]))
#     if sparse:
#         adj_mx_s = sparse_mx_to_torch_sparse_tensor(adj_mx_s)
#     else:
#         adj_mx_s = torch.tensor(adj_mx_s.todense(), dtype=torch.float)


#     adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
#     if sparse:
#         adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
#     else:
#         adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
#     return adj_mx, adj_mx_s


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    # print('----------------------------------', type(adj_mx))
    # print('sparse adj_mx', adj_mx)
    # print('just matrix', torch.tensor(adj_mx.todense()))
    adj_mx_paths = floyd_warshall_sparse(adj_mx)
    # print('floyd adj_mx', adj_mx)
    # ''' The below line is the normalization operation given in every literature'''
    # matrix of hop= 2
    adj_mx_s = (adj_mx_paths.toarray() == 2).astype(float)
    # Convert the dense numpy matrix back to lil_matrix
    adj_mx_s = sp.lil_matrix(adj_mx_s)
    adj_mx_s = normalize(adj_mx_s + sp.eye(adj_mx_s.shape[0]))
    if sparse:
        adj_mx_s = sparse_mx_to_torch_sparse_tensor(adj_mx_s)
    else:
        adj_mx_s = torch.tensor(adj_mx_s.todense(), dtype=torch.float)

    adj_mx_m = (adj_mx_paths.toarray() == 3).astype(float)
    # Convert the dense numpy matrix back to lil_matrix
    adj_mx_m = sp.lil_matrix(adj_mx_m)

    adj_mx_m = normalize(adj_mx_m + sp.eye(adj_mx_m.shape[0]))
    if sparse:
        adj_mx_m = sparse_mx_to_torch_sparse_tensor(adj_mx_m)
    else:
        adj_mx_m = torch.tensor(adj_mx_m.todense(), dtype=torch.float)


    # For 4 hops
    adj_mx_l = (adj_mx_paths.toarray() == 4).astype(float)
    # Convert the dense numpy matrix back to lil_matrix
    adj_mx_l = sp.lil_matrix(adj_mx_l)

    adj_mx_l = normalize(adj_mx_l + sp.eye(adj_mx_l.shape[0]))
    if sparse:
        adj_mx_l = sparse_mx_to_torch_sparse_tensor(adj_mx_l)
    else:
        adj_mx_l = torch.tensor(adj_mx_l.todense(), dtype=torch.float)

    


    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    # print('after norm', adj_mx)
    return adj_mx, adj_mx_s, adj_mx_m, adj_mx_l

def adj_mx_from_skeleton(skeleton):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
    return adj_mx_from_edges(num_joints, edges, sparse=False)
