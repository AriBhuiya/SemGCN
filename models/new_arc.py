import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_per_head = self.dim_model // self.num_heads

        

        self.linear_q = nn.Linear(dim_model, dim_model)
        self.linear_k = nn.Linear(dim_model, dim_model)
        self.linear_v = nn.Linear(dim_model, dim_model)
        
        self.linear_out = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.layer_norm = nn.LayerNorm(16)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_model * 4, dim_model)
        )

    def forward(self, q, k, v):
        num_nodes, _ = q.size()
        residual = q  # Store the input for residual connection
        
        q = self.linear_q(q).view(num_nodes, self.num_heads, self.dim_per_head).transpose(1, 2)
        k = self.linear_k(k).view(num_nodes, self.num_heads, self.dim_per_head).transpose(1, 2)
        v = self.linear_v(v).view(num_nodes, self.num_heads, self.dim_per_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / ((self.dim_per_head) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        

        # weighted_v = self.linear_out(weighted_v)
        
        

        # Position-wise feed-forward network and residual connection
        # ff_out = self.feed_forward(out)
        # out = self.layer_norm(out + ff_out)
        weighted_v = torch.matmul(attn, v).transpose(1,2).reshape(num_nodes, self.num_heads * self.dim_per_head)
        # Layer normalization and residual connection
        # print('====', residual.shape, weighted_v.shape)
        out = self.linear_out(weighted_v)
        out = self.layer_norm(residual + out)
        # weighted_v = weighted_v.mean(dim=1)
        # weighted_v = weighted_v.mean(dim=0)
        # Position-wise feed-forward network and residual connection
        ff_out = self.feed_forward(out)
        out = self.layer_norm(out + ff_out)
        
        return out


class AttentiveAdjacency(nn.Module):
    def __init__(self, num_heads, node_feature_dim):
        super(AttentiveAdjacency, self).__init__()
        self.attention = MultiHeadAttention(num_heads, node_feature_dim)

    def forward(self, adj_shortest_path, adj_matrix=None):
        # Using node_features as Q, K, V
        adjusted_adj= self.attention(adj_shortest_path, adj_shortest_path, adj_shortest_path)

        # If you want to blend the adjusted matrix with the original adjacency matrix, 
        # you can add that logic here. For example:
        # if adj_matrix is not None:
        #     adjusted_features = 0.5 * (adjusted_features + adj_matrix)
        return adjusted_adj


class AttentiveGCNLayer(nn.Module):
    def __init__(self, num_heads, gcn_input_dim, gcn_output_dim):
        super(AttentiveGCNLayer, self).__init__()
        
        self.attention = MultiHeadAttention(num_heads, gcn_input_dim, gcn_input_dim, gcn_input_dim)
        self.gcn = GCN(gcn_input_dim, gcn_output_dim)

    def forward(self, x, adj_shortest_path):
        batch_size, num_nodes, _ = x.shape
        reshaped_adj = adj_shortest_path.view(batch_size * num_nodes, num_nodes)
        reshaped_x = x.view(batch_size * num_nodes, -1)

        attentive_adj = self.attention(reshaped_x, reshaped_x, reshaped_adj)
        attentive_adj = attentive_adj.view(batch_size, num_nodes, num_nodes)

        out = self.gcn(x, attentive_adj)
        return out

class StackedAttentiveGCN(nn.Module):
    def __init__(self, num_layers, num_heads, gcn_input_dim, gcn_hidden_dim, gcn_output_dim):
        super(StackedAttentiveGCN, self).__init__()

        # Creating initial layer
        self.layers = nn.ModuleList([AttentiveGCNLayer(num_heads, gcn_input_dim, gcn_hidden_dim)])
        
        # Creating intermediate layers
        for _ in range(num_layers - 2): # -2 because we're adding an initial and a final layer
            self.layers.append(AttentiveGCNLayer(num_heads, gcn_hidden_dim, gcn_hidden_dim))

        # Creating final layer
        self.layers.append(AttentiveGCNLayer(num_heads, gcn_hidden_dim, gcn_output_dim))

    def forward(self, x, adj_shortest_path):
        out = x
        for layer in self.layers:
            out = layer(out, adj_shortest_path)
        return out
