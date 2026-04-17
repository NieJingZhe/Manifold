# ce.py  (替换你当前的 CE 模块内容)
import numpy as np
import torch
from .kernels import pairwise_dist, KERNEL  # 保留原来的 import，期望 KERNEL 支持 torch.Tensor 输入

# ======= 改进点说明（简要） =======
# - 使用 dtype-aware eps：_eps_for_tensor
# - 不再对 D 做 .detach()/.cpu().numpy()（这样会断开 autograd）
# - 如果 KERNEL.forward / KERNEL.dQdd 无法处理 torch.Tensor，会抛出明确错误提示
# - 兼容 P_list 为 list/numpy/torch 等

# ----------------- helper / 配置 -----------------
# 原来用 1e-12 导致在 float32 (GPU) 上精度丢失 —— 改为 dtype-aware 更安全
def _eps_for_tensor(tensor_or_dtype):
    """
    给定 torch.Tensor 或 torch.dtype，返回一个适合该 dtype 的 eps。
    采用 max(1e-7, 10 * machine_eps) 的策略，保证 float32 上 eps >= 1e-7，避免 log(1 - Q) 被截断。
    """
    if torch.is_tensor(tensor_or_dtype):
        dtype = tensor_or_dtype.dtype
    elif isinstance(tensor_or_dtype, torch.dtype):
        dtype = tensor_or_dtype
    else:
        # fallback
        return 1e-7
    meps = float(torch.finfo(dtype).eps)
    return max(1e-7, meps * 10.0)

# 保留对 numpy/list -> torch 的兼容转换
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

# 数值稳定的 x * log(y) 与 (1-x)*log(1-y)
import torch

def _to_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype)
    return t

def _broadcast_tensors_safe(a, b):
    """
    Try to broadcast a and b to the same shape.
    Return tensors (a_b, b_b).
    If broadcasting not possible, raise informative error.
    """
    # ensure torch.Tensor
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b)

    # move to same device/dtype for broadcasting convenience
    # prefer device of a if available, else b
    device = a.device if isinstance(a, torch.Tensor) else b.device
    dtype = a.dtype if isinstance(a, torch.Tensor) else b.dtype
    a = a.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)
    try:
        a_b, b_b = torch.broadcast_tensors(a, b)
    except Exception as e:
        raise RuntimeError(f"Cannot broadcast tensors: shapes {getattr(a,'shape',None)} and {getattr(b,'shape',None)}") from e
    return a_b, b_b

def _xlogy(x, y, eps=1e-12):
    """
    computes elementwise x * log(y) safely, but returns 0 where x==0.
    Input tensors will be broadcasted if possible.
    """
    # broadcast and align
    x_b, y_b = _broadcast_tensors_safe(x, y)

    # if either is empty -> return zeros with broadcasted shape
    if x_b.numel() == 0 or y_b.numel() == 0:
        return torch.zeros_like(x_b)

    y_clamped = torch.clamp(y_b, eps, 1.0 - eps)
    mask = x_b > 0
    # zeros_like(y_clamped) has same shape/device/dtype
    return torch.where(mask, x_b * torch.log(y_clamped), torch.zeros_like(y_clamped))

def _xlog1my(x, y, eps=1e-12):
    """
    computes elementwise (1-x) * log(1-y) safely, but returns 0 where (1-x)==0.
    Input tensors will be broadcasted if possible.
    """
    x_b, y_b = _broadcast_tensors_safe(x, y)

    if x_b.numel() == 0 or y_b.numel() == 0:
        return torch.zeros_like(x_b)

    # clamp y to avoid log(0)
    y_clamped = torch.clamp(y_b, eps, 1.0 - eps)
    mask = (1.0 - x_b) > 0
    # torch.log1p(-y_clamped) is numerically stable for log(1-y)
    return torch.where(mask, (1.0 - x_b) * torch.log1p(-y_clamped), torch.zeros_like(y_clamped))

def CE(Pi, Q, eps=1e-12, reduction='mean', off_diagonal=True):
    """
    Safe cross-entropy between two probability matrices Pi and Q.
    - reduction: 'sum' | 'mean' | 'none'
    - off_diagonal: if True, exclude diagonal elements when computing CE (common for P matrices)
    Returns scalar (if reduction!='none') or full matrix (if reduction=='none').
    """
    # convert and broadcast (reuse your _broadcast_tensors_safe)
    Pi_t, Q_t = _broadcast_tensors_safe(Pi, Q)

    if Pi_t.numel() == 0:
        if reduction == 'none':
            return torch.zeros_like(Pi_t)
        return torch.tensor(0.0, device=Pi_t.device, dtype=Pi_t.dtype)

    term1 = _xlogy(Pi_t, Q_t, eps=eps)
    term2 = _xlog1my(Pi_t, Q_t, eps=eps)
    CE_mat = - (term1 + term2)  # shape [N,N] or broadcast

    if off_diagonal:
        if CE_mat.dim() >= 2 and CE_mat.shape[-1] == CE_mat.shape[-2]:
            eye = torch.eye(CE_mat.shape[-1], dtype=torch.bool, device=CE_mat.device)
            mask = ~eye
            vals = CE_mat[mask]
            if reduction == 'none':
                out = torch.zeros_like(CE_mat)
                out[mask] = vals
                return out
            if vals.numel() == 0:
                return torch.tensor(0.0, device=CE_mat.device, dtype=CE_mat.dtype)
            if reduction == 'sum':
                return vals.sum()
            return vals.mean()
        else:
            # cannot form square mask -> fallback to normal reductions
            if reduction == 'sum':
                return CE_mat.sum()
            elif reduction == 'mean':
                return CE_mat.mean()
            else:
                return CE_mat

    # not off_diagonal: use full matrix
    if reduction == 'sum':
        return CE_mat.sum()
    elif reduction == 'mean':
        return CE_mat.mean()
    elif reduction == 'none':
        return CE_mat
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
# 纯 torch 版 pairwise dist 的回退（兼容 autograd）
def _pairwise_dist_torch(Y):
    """纯 torch 版 pairwise distance (兼容 autograd)。"""
    diff = Y.unsqueeze(1) - Y.unsqueeze(0)   # (N, N, dim)
    D = torch.norm(diff, dim=-1)            # (N, N)
    return D

def _call_pairwise_dist(Y):
    """
    优先尝试外部 pairwise_dist（如果它能接受 torch Tensor 并返回 torch Tensor）。
    如果失败则回退为内部 torch 版本。
    如果外部返回 numpy ndarray，会转为 torch.Tensor（但前提仍是外部能接受 torch）。
    """
    # 优先尝试外部 pairwise_dist（可能是你改过的 torch-native 版本）
    try:
        out = pairwise_dist(Y)
    except Exception:
        # 回退到 torch 实现（安全）
        return _pairwise_dist_torch(Y)
    else:
        # 如果外部返回 numpy -> 转成 torch（注意：如果外部内部使用 numpy 并要求 numpy 输入而你传 torch，会触发上面的 except）
        if isinstance(out, np.ndarray):
            return torch.from_numpy(out).to(dtype=Y.dtype, device=Y.device)
        if torch.is_tensor(out):
            return out.to(dtype=Y.dtype, device=Y.device)
        # 其他可转为 tensor 的情况
        return _to_torch(out, dtype=Y.dtype, device=Y.device)

# 重要：直接调用 KERNEL.forward / dQdd，**不再**做 .detach() 操作
def _kernel_forward_tensor(D):
    """
    调用 KERNEL.forward(D) 并保证返回 torch.Tensor（若 KERNEL 返回 numpy，会转换为 torch）。
    如果 KERNEL 无法处理 torch.Tensor（抛异常），这里会抛出明确错误，提示需要为 KERNEL 提供 torch 版本。
    """
    try:
        Q = KERNEL.forward(D)   # 期望 KERNEL 能直接接受 torch.Tensor
    except Exception as e:
        # 如果这里失败，往往是因为 KERNEL.forward 仅支持 numpy 输入 —— 我们不做 detach 转 numpy（会断开 autograd）
        raise RuntimeError(
            "KERNEL.forward 不能直接处理 torch.Tensor（在调用时抛异常）。"
            " 请确保你使用的 KERNEL 实现能够接受并返回 torch.Tensor，"
            " 或者在此处提供一个 torch-native 的 fallback 实现。"
            f" 原始错误: {e}"
        )
    else:
        if isinstance(Q, np.ndarray):
            # 如果 KERNEL 返回 numpy（不推荐），直接转回 torch（注意：如果 KERNEL 内部用了 numpy 但接受 torch 输入，这里仍保留计算图？"
            # 实际上如果 KERNEL 内部用了 numpy，该前向已断开 autograd —— 因此这里仍会导致无梯度，"
            # 所以强烈建议让 KERNEL 返回 torch.Tensor）
            Q = torch.from_numpy(np.asarray(Q)).to(dtype=D.dtype, device=D.device)
        elif torch.is_tensor(Q):
            Q = Q.to(dtype=D.dtype, device=D.device)
        else:
            Q = _to_torch(Q, dtype=D.dtype, device=D.device)
    return Q

def _kernel_dQdd_tensor(D):
    """
    调用 KERNEL.dQdd(D)，并把输出转成 torch.Tensor。
    如果 KERNEL.dQdd 无法接受 torch.Tensor，会抛错（不做 detach 转 numpy，以避免静默断开 autograd）。
    """
    try:
        d = KERNEL.dQdd(D)
    except Exception as e:
        raise RuntimeError(
            "KERNEL.dQdd 不能直接处理 torch.Tensor（在调用时抛异常）。"
            " 请确保 KERNEL.dQdd 能接受 torch.Tensor 并返回 torch.Tensor，"
            " 或为 KERNEL 提供 torch-native 的 dQ/dd 实现。"
            f" 原始错误: {e}"
        )
    else:
        if isinstance(d, np.ndarray):
            d = torch.from_numpy(np.asarray(d)).to(dtype=D.dtype, device=D.device)
        elif torch.is_tensor(d):
            d = d.to(dtype=D.dtype, device=D.device)
        else:
            d = _to_torch(d, dtype=D.dtype, device=D.device)
    return d

# ---------- CE 主逻辑 ----------

def CE_gradient(P, Y):
    """
    使用 torch 计算 dL/dY 的全量梯度（返回 shape (N, dim)）。
    输入 P, Y 可以是 numpy 或 torch（内部会转换）。
    注意：如果你的 KERNEL 实现无法在 torch 中计算 dQ/dd，请先为其提供 torch 版本。
    """
    device = None
    if torch.is_tensor(Y):
        device = Y.device
    Y = _to_torch(Y, device=device)
    P = _to_torch(P, dtype=Y.dtype, device=Y.device)

    # eps for dtype
    eps = _eps_for_tensor(Y)

    diff = Y.unsqueeze(1) - Y.unsqueeze(0)      # (N, N, dim)
    D = _call_pairwise_dist(Y)                  # (N, N)
    Q = _kernel_forward_tensor(D)               # (N, N)
    N = Y.shape[0]
    eye = torch.eye(N, dtype=torch.bool, device=Y.device)

    # dL/dQ
    Qc = torch.clamp(Q, eps, 1.0 - eps)
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
