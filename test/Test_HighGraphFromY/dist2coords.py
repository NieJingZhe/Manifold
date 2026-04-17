from sklearn.manifold import MDS
import numpy as np
import torch

def coords2dict_mds(dist_matrix, dim=3):
    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=0)
    coords = mds.fit_transform(dist_matrix)
    return coords


def coords2dict_tch(dist_matrix, lr=1e-2, steps=1000):
    N = dist_matrix.shape[0]
    D = torch.tensor(dist_matrix, dtype=torch.float32)
    X = torch.randn(N, 3, requires_grad=True)

    optimizer = torch.optim.Adam([X], lr=lr)
    for step in range(steps):
        dist_pred = torch.cdist(X, X)
        loss = ((dist_pred - D) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return X.detach().numpy()
