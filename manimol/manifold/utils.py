# utils.py
"""
Common small utilities for P-graph construction.
This file contains pure tensor/np helpers (no RDKit).
"""

import torch
import numpy as np
from typing import Optional, Tuple


def umap_low_kernel(D: np.ndarray, a: float, b: float, eps: float = 1e-12) -> np.ndarray:
    """UMAP-style low-dimensional affinity kernel.

    Q = 1 / (1 + a * D^(2b)), diagonal set to 0 and clipped to (eps,1-eps).

    Inputs
    - D: (N,N) pairwise Euclidean distances (numpy array)
    - a,b: kernel hyperparameters
    - eps: small value for numerical stability

    Output
    - Q: (N,N) affinity/probability matrix (numpy array)
    """
    Q = 1.0 / (1.0 + a * np.clip(D, eps, None) ** (2 * b))
    np.fill_diagonal(Q, 0.0)
    return np.clip(Q, eps, 1.0 - eps)


def pairwise_euclid(Y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix from coordinates Y.

    Inputs
    - Y: (N,3) numpy array of coordinates (or (N,d))
    - eps: small value added inside sqrt for stability

    Output
    - D: (N,N) numpy distance matrix
    """
    diff = Y[:, None, :] - Y[None, :, :]
    return np.sqrt((diff ** 2).sum(-1) + eps)


def to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy/torch/list to torch Tensor with given dtype.

    Inputs
    - x: numpy array, torch tensor, list, or scalar
    - dtype: torch dtype for floats

    Output
    - torch.Tensor
    """
    if torch.is_tensor(x):
        # allow changing dtype for numeric tensors (but preserve long when requested)
        try:
            return x.to(dtype=dtype)
        except Exception:
            return x
    return torch.tensor(x, dtype=dtype)


def logit(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable logit: log(p) - log(1-p).

    Inputs
    - p: tensor of probabilities
    - eps: clamp value

    Output
    - tensor of logits same shape as p
    """
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log(1.0 - p)


def ensure_bidirectional(edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make edges bidirectional by appending reversed edges and duplicating attributes.

    Inputs
    - edge_index: LongTensor [2, E]
    - edge_attr: Tensor [E, F]

    Outputs
    - edge_index2: LongTensor [2, 2E]
    - edge_attr2: Tensor [2E, F]
    """
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    src, dst = edge_index
    rev = torch.stack([dst, src], dim=0)
    edge_index2 = torch.cat([edge_index, rev], dim=1)
    edge_attr2 = torch.cat([edge_attr, edge_attr], dim=0)
    return edge_index2, edge_attr2


def select_topk_from_P(P: torch.Tensor, topk: int, exclude_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Selects top-k neighbors for each node from a probability matrix P.

    Inputs
    - P: FloatTensor [N,N]
    - topk: int (<= N-1)
    - exclude_mask: optional BoolTensor [N,N] where True entries are forbidden (set to 0)

    Outputs
    - src: LongTensor [E_p] (source indices)
    - dst: LongTensor [E_p] (destination indices)
    - vals: FloatTensor [E_p] (P values for each selected edge)
    """
    N = P.size(0)
    src_list = []
    dst_list = []
    val_list = []
    for i in range(N):
        scores = P[i].clone()
        scores[i] = 0.0
        if exclude_mask is not None:
            scores[exclude_mask[i]] = 0.0
        k = min(max(0, topk), N - 1)
        if k == 0:
            continue
        idx = torch.topk(scores, k=k, largest=True).indices
        for j in idx.tolist():
            src_list.append(i)
            dst_list.append(j)
            # keep as 1D tensor for concatenation later
            val_list.append(scores[j].unsqueeze(0))
    if len(src_list) == 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    return torch.tensor(src_list, dtype=torch.long), torch.tensor(dst_list, dtype=torch.long), torch.cat(val_list, dim=0)
