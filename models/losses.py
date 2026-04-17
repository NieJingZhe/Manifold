import numpy as np
import torch
from .kernels import pairwise_dist, KERNEL  # 保留原来的 import，兼容回退

_EPS = 1e-12

def _to_torch(x, dtype=torch.float32, device=None):
    if torch.is_tensor(x):
        return x.to(dtype=dtype, device=device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    if isinstance(x, (list, tuple)):
        # 如果 list 里都是 ndarray，就先堆成一个大 ndarray
        try:
            x = np.array(x)
        except Exception:
            # 如果里面是 ragged list，不规则，就保持原状
            pass
    return torch.as_tensor(x, dtype=dtype, device=device)

def _xlogy(x, y, eps=_EPS):
    """数值稳定版 x*log(y)，当 x==0 时返回 0。支持 torch tensor。"""
    y = torch.clamp(y, eps, 1.0 - eps)
    mask = (x != 0)
    return torch.where(mask, x * torch.log(y), torch.zeros_like(y))

def _xlog1my(x, y, eps=_EPS):
    """数值稳定版 (1-x)*log(1-y)，当 x==1 时返回 0。支持 torch tensor。"""
    y = torch.clamp(y, eps, 1.0 - eps)
    mask = (x != 1)
    return torch.where(mask, (1.0 - x) * torch.log(1.0 - y), torch.zeros_like(y))

def _pairwise_dist_torch(Y):
    """纯 torch 版 pairwise distance (兼容 autograd)。"""
    diff = Y.unsqueeze(1) - Y.unsqueeze(0)   # (N, N, dim)
    D = torch.norm(diff, dim=-1)            # (N, N)
    return D

def _call_pairwise_dist(Y):
    """尝试用外部 pairwise_dist；失败时使用 torch 本地实现；若外部返回 numpy，转回 tensor。"""
    # 优先尝试外部 pairwise_dist（可能是 numpy 版本）
    try:
        out = pairwise_dist(Y)
    except Exception:
        # 回退到 torch 实现
        return _pairwise_dist_torch(Y)
    else:
        if isinstance(out, np.ndarray):
            return torch.from_numpy(out).to(dtype=Y.dtype, device=Y.device)
        if torch.is_tensor(out):
            return out.to(dtype=Y.dtype, device=Y.device)
        # 其他可转为 tensor 的情况
        return _to_torch(out, dtype=Y.dtype, device=Y.device)

def _kernel_forward_tensor(D):
    """调用 KERNEL.forward(D)。自动在 numpy/tensor 之间转换以兼容旧实现。"""
    try:
        Q = KERNEL.forward(D)
    except Exception:
        # 可能 KERNEL 需要 numpy
        D_np = D.detach().cpu().numpy()
        Q_np = KERNEL.forward(D_np)
        Q = torch.from_numpy(np.asarray(Q_np)).to(dtype=D.dtype, device=D.device)
    else:
        if isinstance(Q, np.ndarray):
            Q = torch.from_numpy(Q).to(dtype=D.dtype, device=D.device)
        elif torch.is_tensor(Q):
            Q = Q.to(dtype=D.dtype, device=D.device)
    return Q

def _kernel_dQdd_tensor(D):
    """调用 KERNEL.dQdd(D)，并把输出转成 tensor（兼容 numpy 实现）。"""
    try:
        d = KERNEL.dQdd(D)
    except Exception:
        D_np = D.detach().cpu().numpy()
        d_np = KERNEL.dQdd(D_np)
        d = torch.from_numpy(np.asarray(d_np)).to(dtype=D.dtype, device=D.device)
    else:
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(d).to(dtype=D.dtype, device=D.device)
        elif torch.is_tensor(d):
            d = d.to(dtype=D.dtype, device=D.device)
    return d
#修改了将P改为一个list
def CE(P_list, Y):
    """
    适配 P 为 list 的版本（每个元素对应一个样本的方阵）
    返回每个样本的 CE 矩阵列表（torch tensor 形式）
    CE_ij = - [ P_ij log Q_ij + (1-P_ij) log(1-Q_ij) ]，忽略对角项
    """
    device = None
    if torch.is_tensor(Y):
        device = Y.device
    Y = _to_torch(Y, device=device)

    CE_matrices = []
    idx = 0  # 用于切分 Y 中的节点
    for Pi in P_list:
        Pi = _to_torch(Pi, dtype=Y.dtype, device=Y.device)
        n = Pi.shape[0]

        Yi = Y[idx: idx + n]   # 每个样本的节点向量
        idx += n

        D = _call_pairwise_dist(Yi)           # (n, n)
        Q = _kernel_forward_tensor(D)         # (n, n)

        eye = torch.eye(n, dtype=torch.bool, device=Y.device)
        CE_mat = - _xlogy(Pi, Q) - _xlog1my(Pi, Q)
        CE_mat = CE_mat.masked_fill(eye, 0.0)
        CE_mat[~torch.isfinite(CE_mat)] = 0.0

        CE_matrices.append(CE_mat)

    return CE_matrices

def CE_gradient(P, Y):
    """
    使用 torch 计算 dL/dY 的全量梯度（返回 shape (N, dim)）。
    输入 P, Y 可以是 numpy 或 torch（内部会转换）。
    """
    device = None
    if torch.is_tensor(Y):
        device = Y.device
    Y = _to_torch(Y, device=device)
    P = _to_torch(P, dtype=Y.dtype, device=Y.device)

    diff = Y.unsqueeze(1) - Y.unsqueeze(0)      # (N, N, dim)
    D = _call_pairwise_dist(Y)                  # (N, N)
    Q = _kernel_forward_tensor(D)               # (N, N)
    N = Y.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=Y.device)

    # dL/dQ
    Qc = torch.clamp(Q, _EPS, 1.0 - _EPS)
    dLdQ = - (P / Qc) + ((1.0 - P) / (1.0 - Qc))
    dLdQ = dLdQ.masked_fill(eye, 0.0)

    # dQ/dd
    dQdd = _kernel_dQdd_tensor(D)
    dQdd = dQdd.masked_fill(eye, 0.0)

    dLdd = dLdQ * dQdd
    dLdd[~torch.isfinite(dLdd)] = 0.0

    # ∂d/∂y_i : vec = (y_i - y_j) / d_ij
    denom = D.unsqueeze(-1)                    # (N,N,1)
    vec = diff / denom
    vec[~torch.isfinite(vec)] = 0.0

    grad = torch.sum(dLdd.unsqueeze(-1) * vec, dim=1)  # (N, dim)
    grad[~torch.isfinite(grad)] = 0.0
    return grad
