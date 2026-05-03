import torch
import torch.nn as nn
import numpy as np


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        adj = adj.to(x.device)
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        return torch.relu(torch.matmul(adj_norm, self.linear(x)))


class GCNEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layer1 = GraphConvLayer(in_features, hidden_features)
        self.layer2 = GraphConvLayer(hidden_features, out_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, adj):
        x = self.dropout(torch.relu(self.layer1(x, adj)))
        return self.layer2(x, adj)


def build_adjacency_from_similarity(sim_matrix, k_neighbors=10):
    n = sim_matrix.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        similarities = sim_matrix[i].copy()
        similarities[i] = -np.inf
        adj[i, np.argsort(similarities)[-k_neighbors:]] = 1
    adj = ((adj + adj.T) > 0).astype(float)
    np.fill_diagonal(adj, 1)
    return adj


def normalize_adjacency(adj):
    adj = torch.tensor(adj, dtype=torch.float32)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
