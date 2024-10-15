# reference from DeepDock nature machine intellience paper
# import numpy as np
import torch

def compute_euclidean_distances_matrix(X, Y):
    # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # (X-Y)^2 = X^2 + Y^2 -2XY
    X = X.double()
    Y = Y.double()
    dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
    return dists**0.5
def compute_euclidean_distances_matrix_TopN( X, Y,B, N_l,topN = 1):
    X = X.double()
    Y = Y.double()
    dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2,    axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
    dists = torch.nan_to_num((dists**0.5).view(B, N_l,-1,24),10000).sort(axis=-1)[0][:,:,:,:topN]
    dist_topN = []
    for i in range(topN):
        dist_topN.append(dists[:,:,:,i])
    return dist_topN
