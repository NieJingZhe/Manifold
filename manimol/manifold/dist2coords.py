from sklearn.manifold import MDS
import numpy as np
import torch
import torch.nn.functional as F
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


# utils/refine_utils.py


def compute_Q_from_coords(coords: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    coords: (N, d) torch tensor
    returns Q_pred: (N, N) normalized affinity (torch tensor)
    """
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)      # (N, N, d)
    dist_sq = (diff * diff).sum(dim=2)                    # (N, N)
    q = 1.0 / (1.0 + dist_sq)                             # student-t kernel
    q.fill_diagonal_(0.0)
    q_sum = q.sum()
    if q_sum.item() == 0:
        q = q + eps
        q_sum = q.sum()
    Q_pred = q / (q_sum + eps)
    return Q_pred

def refine_loss_from_P_coords(P_target: torch.Tensor, coords: torch.Tensor, eps: float = 1e-12) -> (torch.Tensor, torch.Tensor):
    """
    P_target: (N,N) torch tensor (can be not on same device, ensure to move outside)
    coords: (N, d) torch tensor (requires_grad True or False depending on usage)
    returns: (loss_tensor, Q_pred)
    """
    # ensure same device & dtype handled by caller
    # clamp P and normalize for stability (assume P_target already prepared in model pipeline)
    P = torch.clamp(P_target, min=0.0)
    P_sum = P.sum()
    if P_sum.item() == 0:
        raise ValueError("P_target is all zeros.")
    P = P / (P_sum + eps)

    Q_pred = compute_Q_from_coords(coords, eps=eps)
    # CE loss: - sum P * log Q
    loss = - (P * torch.log(Q_pred + eps)).sum()
    loss = loss / (P.sum() + eps)   # optional normalization (if P is already normalized this does nothing)
    return loss, Q_pred
