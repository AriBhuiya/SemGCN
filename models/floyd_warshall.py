import numpy as np
import torch


def floyd_warshall(adj_matrix):
    # Number of vertices in the graph
    V = adj_matrix.shape[0]
    print(adj_matrix[1, 1])
    # Create a matrix to store shortest distances
    dist_matrix = torch.full((V, V), float('inf'))
    
    # Initialize with the given adjacency matrix
    for i in range(V):
        for j in range(V):
            if i == j:
                dist_matrix[i, j] = 0
            elif adj_matrix[i, j] != 0:
                dist_matrix[i, j] = 1
    
    # Update distances
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist_matrix[i, j] = min(dist_matrix[i, j], dist_matrix[i, k] + dist_matrix[k, j])
    
    return dist_matrix




import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

def floyd_warshall_sparse(adj_matrix):
    # Convert the csr_matrix to lil format for easier element-wise operations
    adj_matrix = adj_matrix.tolil()
    
    # Number of vertices in the graph
    V = adj_matrix.shape[0]
    
    # Create a matrix to store shortest distances
    dist_matrix = lil_matrix((V, V))
    dist_matrix[:] = float('inf')
    
    # Initialize with the given adjacency matrix
    for i in range(V):
        dist_matrix[i, i] = 0
        for j in adj_matrix.rows[i]:
            dist_matrix[i, j] = 1
    
    # Update distances
    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist_matrix[i, j] = min(dist_matrix[i, j], dist_matrix[i, k] + dist_matrix[k, j])
    
    return dist_matrix
