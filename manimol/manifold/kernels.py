# kernels.py
"""
Torch-native kernels for CE / manifold loss.

设计目标：
- 当输入为 torch.Tensor 时，全部用 torch 运算（不会 detach / 转 numpy），保证 autograd 连通。
- 若输入为 numpy.ndarray，会内部转换为 torch 并返回 torch.Tensor（如需 numpy 输出请在上层显式转换）。
- find_ab_params 保留 numpy + scipy 的拟合（离线工具）。
- SmoothKRowExpKernel.fit_from_dist 使用 numpy 实现拟合（便于复用原有算法），但会在 fit 后把参数转换为 torch 用于 forward/dQdd。
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# small eps
EPS = 1e-12

# ----------------- 工具：输入类型兼容 -----------------
def _ensure_torch(x, device=None, dtype=torch.float32):
    """
    If x is numpy array -> convert to torch on device.
    If x is torch -> ensure dtype/device.
    Returns torch tensor.
    """
    if torch.is_tensor(x):
        if device is None:
            return x.to(dtype=dtype)
        else:
            return x.to(device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        if device is None:
            return torch.from_numpy(x).to(dtype=dtype)
        else:
            return torch.from_numpy(x).to(device=device, dtype=dtype)
    # for python lists etc.
    return torch.as_tensor(x, dtype=dtype, device=device)

def _is_torch(x):
    return torch.is_tensor(x)

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def clamp01_torch(Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Clamp to [eps, 1-eps] and set diagonal to 0. Returns torch tensor."""
    Q = torch.clamp(Q, min=eps, max=1.0 - eps)
    n = Q.shape[0]
    if n > 0:
        Q = Q.clone()
        Q.view(-1)[::n+1] = 0.0  # set diagonal to zero
    return Q

# ----------------- pairwise dist (torch) -----------------
def pairwise_dist(Y):
    """
    Accepts torch.Tensor or numpy.ndarray; returns torch.Tensor of shape (N,N).
    Uses sqrt(sum((yi-yj)^2) + EPS).
    """
    EPS=1e-12
    if not torch.is_tensor(Y):
        Y = _ensure_torch(Y)
    # Y: (N, dim)
    diff = Y.unsqueeze(1) - Y.unsqueeze(0)   # (N,N,dim)
    D = torch.sqrt(torch.sum(diff * diff, dim=-1) + EPS)
    return D

# --------------- UMAP a,b 拟合（保留 numpy 版本） ---------------
def _umap_target_curve(x, min_dist=0.5, spread=1.0):
    y = np.ones_like(x)
    m = x > min_dist
    y[m] = np.exp(-(x[m] - min_dist) / spread)
    return y

def find_ab_params(min_dist=0.5, spread=1.0):
    """仍用 scipy curve_fit 在 numpy 上拟合，离线工具。"""
    from scipy.optimize import curve_fit
    x = np.linspace(0, 3.0 * spread, 300)
    y = _umap_target_curve(x, min_dist=min_dist, spread=spread)
    f = lambda x, a, b: 1.0 / (1.0 + a * (x ** (2.0 * b)))
    (a, b), _ = curve_fit(f, x, y, p0=(1.6, 0.8), maxfev=10000)
    return float(a), float(b)

# ----------------- 基类接口 -----------------
class BaseKernel:
    def forward(self, D):
        raise NotImplementedError
    def dQdd(self, D):
        raise NotImplementedError
    def inv(self, Q):
        raise NotImplementedError

# ----------------- UMAP 低维核（torch 实现） -----------------
@dataclass
class UMAPLowKernel_1(BaseKernel):
    a: float = 1.6
    b: float = 0.8

    def forward(self, D):
        D_t = _ensure_torch(D)
        # Q = 1 / (1 + a * (max(D,EPS) ** (2b)))
        Dp = torch.clamp(D_t, min=EPS)
        Q = 1.0 / (1.0 + self.a * torch.pow(Dp, 2.0 * self.b))
        return clamp01_torch(Q)

    def dQdd(self, D):
        D_t = _ensure_torch(D)
        Dp = torch.clamp(D_t, min=EPS)
        num = (2.0 * self.a * self.b) * torch.pow(Dp, 2.0 * self.b - 1.0)
        den = torch.pow(1.0 + self.a * torch.pow(Dp, 2.0 * self.b), 2.0)
        d = - num / (den + EPS)
        # diagonal safe
        if d.dim() == 2:
            n = d.shape[0]
            d = d.clone()
            d.view(-1)[::n+1] = 0.0
        return d

    def inv(self, Q):
        Q_t = _ensure_torch(Q)
        x = ((1.0 / torch.clamp(Q_t, min=1e-12, max=1.0 - 1e-12) - 1.0) / self.a).clamp(min=0.0)
        return torch.pow(x, 1.0 / (2.0 * self.b))
    
class UMAPLowKernel(nn.Module):
    def __init__(self, a, b):
        super().__init__(); self.a=a; self.b=b
    def forward(self, D):                 # D: [n,n]
        Q = 1.0 / (1.0 + self.a * torch.clamp(D, min=EPS).pow(2*self.b))
        Q = torch.clamp(Q, 1e-12, 1-1e-12)
        Q.fill_diagonal_(0.0)
        return Q

# ----------------- Gaussian 核（torch） -----------------
@dataclass
class GaussianKernel(BaseKernel):
    sigma: float = 1.0

    def forward(self, D):
        D_t = _ensure_torch(D)
        s2 = float(self.sigma ** 2)
        Q = torch.exp(-(D_t * D_t) / (2.0 * s2))
        return clamp01_torch(Q)

    def dQdd(self, D):
        D_t = _ensure_torch(D)
        s2 = float(self.sigma ** 2)
        Q = torch.exp(-(D_t * D_t) / (2.0 * s2))
        d = - (D_t / s2) * Q
        if d.dim() == 2:
            n = d.shape[0]; d = d.clone(); d.view(-1)[::n+1] = 0.0
        return d

    def inv(self, Q):
        Q_t = _ensure_torch(Q)
        s2 = float(self.sigma ** 2)
        return torch.sqrt(torch.clamp(-2.0 * s2 * torch.log(torch.clamp(Q_t, min=1e-12, max=1.0 - 1e-12)), min=0.0))

# ----------------- Student-t 核（torch） -----------------
@dataclass
class StudentTKernel(BaseKernel):
    nu: float = 1.0

    def forward(self, D):
        D_t = _ensure_torch(D)
        Q = 1.0 / (1.0 + (D_t * D_t) / float(self.nu))
        return clamp01_torch(Q)

    def dQdd(self, D):
        D_t = _ensure_torch(D)
        nu = float(self.nu)
        Dp = torch.clamp(D_t, min=EPS)
        d = - (2.0 * Dp / nu) / (torch.pow(1.0 + (Dp * Dp) / nu, 2.0) + EPS)
        if d.dim() == 2:
            n = d.shape[0]; d = d.clone(); d.view(-1)[::n+1] = 0.0
        return d

    def inv(self, Q):
        Q_t = _ensure_torch(Q)
        x = (1.0 / torch.clamp(Q_t, min=1e-12, max=1.0 - 1e-12) - 1.0) * float(self.nu)
        return torch.sqrt(torch.clamp(x, min=0.0))

# ----------------- UMAP 行核（exp 版本）-----------------
class UMAPRowExpKernel(BaseKernel):
    def __init__(self, rho: np.ndarray, sigma: np.ndarray):
        # store as numpy but also prepare torch versions lazily
        self.rho_np = np.asarray(rho, dtype=float)
        self.sigma_np = np.asarray(sigma, dtype=float)
        self._rho_t: Optional[torch.Tensor] = None
        self._sigma_t: Optional[torch.Tensor] = None

    def _ensure_tensors(self, device=None, dtype=torch.float32):
        if self._rho_t is None:
            self._rho_t = _ensure_torch(self.rho_np, device=device, dtype=dtype)
            self._sigma_t = _ensure_torch(self.sigma_np, device=device, dtype=dtype)
        return self._rho_t, self._sigma_t

    def _qi(self, D):
        # D can be numpy or torch; ensure torch
        D_t = _ensure_torch(D)
        rho_t, sigma_t = self._ensure_tensors(device=D_t.device, dtype=D_t.dtype)
        Ri = rho_t[:, None]
        Si = sigma_t[:, None] + EPS
        X = D_t - Ri
        X = torch.clamp(X, min=0.0)
        qi = torch.exp(- X / Si)
        return qi

    def forward(self, D):
        qi = self._qi(D)
        qj = qi.T
        Q = qi + qj - qi * qj
        return clamp01_torch(Q)

    def dQdd(self, D):
        D_t = _ensure_torch(D)
        rho_t, sigma_t = self._ensure_tensors(device=D_t.device, dtype=D_t.dtype)
        Ri = rho_t[:, None]
        Rj = rho_t[None, :]
        Si = sigma_t[:, None] + EPS
        Sj = sigma_t[None, :] + EPS

        qi = self._qi(D_t)
        qj = qi.T

        mask_i = (D_t > Ri).to(dtype=D_t.dtype)
        mask_j = (D_t > Rj).to(dtype=D_t.dtype)
        dqi_dd = - (qi / Si) * mask_i
        dqj_dd = - (qj / Sj) * mask_j

        dQdd = dqi_dd * (1.0 - qj) + dqj_dd * (1.0 - qi)
        # diagonal zero
        n = dQdd.shape[0]
        dQdd = dQdd.clone(); dQdd.view(-1)[::n+1] = 0.0
        return dQdd

    def inv(self, Q):
        raise NotImplementedError("UMAPRowExpKernel: no closed-form inv for fuzzy-union row kernel.")

# ----------------- UMAP 行族核（torch） -----------------
class UMAPRowFamilyKernel(BaseKernel):
    def __init__(self, rho: np.ndarray, sigma: np.ndarray, a: float = 1.6, b: float = 0.8):
        self.rho_np = np.asarray(rho, dtype=float)
        self.sigma_np = np.asarray(sigma, dtype=float)
        self.a = float(a); self.b = float(b)
        self._rho_t: Optional[torch.Tensor] = None
        self._sigma_t: Optional[torch.Tensor] = None

    def _ensure_tensors(self, device=None, dtype=torch.float32):
        if self._rho_t is None:
            self._rho_t = _ensure_torch(self.rho_np, device=device, dtype=dtype)
            self._sigma_t = _ensure_torch(self.sigma_np, device=device, dtype=dtype)
        return self._rho_t, self._sigma_t

    def _Z(self, D):
        D_t = _ensure_torch(D)
        rho_t, sigma_t = self._ensure_tensors(device=D_t.device, dtype=D_t.dtype)
        Ri = rho_t[:, None]
        Si = sigma_t[:, None] + EPS
        Z = (D_t - Ri) / Si
        Z = torch.clamp(Z, min=0.0)
        return Z, Si

    def _qi(self, D):
        Z, _ = self._Z(D)
        qi = 1.0 / (1.0 + self.a * torch.pow(torch.clamp(Z, min=0.0), 2.0 * self.b))
        return qi

    def forward(self, D):
        qi = self._qi(D)
        qj = qi.T
        Q = qi + qj - qi * qj
        return clamp01_torch(Q)

    def dQdd(self, D):
        D_t = _ensure_torch(D)
        Z, Si = self._Z(D_t)
        qi = self._qi(D_t)
        qj = qi.T

        Zi = torch.clamp(Z, min=0.0)
        mask_i = (Zi > 0.0).to(dtype=D_t.dtype)

        num = (2.0 * self.a * self.b) * torch.pow(Zi, 2.0 * self.b - 1.0)
        den = torch.pow(1.0 + self.a * torch.pow(Zi, 2.0 * self.b), 2.0)
        dqi_dd = - (num / (den + EPS)) * (1.0 / Si) * mask_i

        dqj_dd = dqi_dd.T
        dQdd = dqi_dd * (1.0 - qj) + dqj_dd * (1.0 - qi)
        n = dQdd.shape[0]; dQdd = dQdd.clone(); dQdd.view(-1)[::n+1] = 0.0
        return dQdd

    def inv(self, Q):
        raise NotImplementedError("UMAPRowFamilyKernel: no closed-form inv for fuzzy-union row kernel.")

# ----------------- SmoothKRowExpKernel -----------------
# 由于 fit_from_dist 原本基于 numpy 的二分法与度数统计，保持其 numpy 实现，
# 但在 fit 结束后把参数转换为 torch 以便 forward & dQdd 在 torch 内运行。
class SmoothKRowExpKernel(BaseKernel):
    def __init__(self,
                 K_HOP_MAX: int = 2,
                 k_offset: int = 2,
                 k_min: int = 3,
                 k_max: int = 8,
                 sigma_lo: float = 1e-6,
                 sigma_hi: float = 1.0,
                 sigma_iters: int = 32):
        self.K_HOP_MAX = K_HOP_MAX
        self.k_offset  = k_offset
        self.k_min     = k_min
        self.k_max     = k_max
        self.sigma_lo  = sigma_lo
        self.sigma_hi  = sigma_hi
        self.sigma_iters = sigma_iters

        # numpy-stored during fit
        self.rho_np: Optional[np.ndarray] = None
        self.sigma_np: Optional[np.ndarray] = None
        self.N: Optional[int] = None
        self.cand_mask_np: Optional[np.ndarray] = None
        self.frozen_row_np: Optional[np.ndarray] = None
        self.frozen_qrow_np: Optional[np.ndarray] = None

        # torch-stored after fit (for forward/dQdd)
        self.rho_t: Optional[torch.Tensor] = None
        self.sigma_t: Optional[torch.Tensor] = None
        self.cand_mask_t: Optional[torch.Tensor] = None
        self.frozen_row_t: Optional[torch.Tensor] = None
        self.frozen_qrow_t: Optional[torch.Tensor] = None

    # 复用原有 numpy 实现（保持行为一致）
    @staticmethod
    def _sigma_binary_search_row(k_of_sigma, fixed_k, lo, hi, iters):
        while k_of_sigma(hi) < fixed_k and hi < 1e6:
            hi *= 2.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if k_of_sigma(mid) < fixed_k:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _prob_row_from_sigma(sig, d_row, rho_i):
        x = d_row - rho_i
        x[x < 0.0] = 0.0
        return np.exp(- x / (sig + EPS))

    @staticmethod
    def _k_from_prob(prob_row):
        return np.power(2.0, float(prob_row.sum())) - 1.0

    @staticmethod
    def _degree_based_k(mol, n, k_offset, k_min, k_max):
        if mol is not None:
            try:
                deg = np.array([a.GetDegree() for a in mol.GetAtoms()], dtype=int)
                return np.clip(deg + k_offset, k_min, k_max).astype(float)
            except Exception:
                pass
        return np.full(n, float((k_min + k_max) // 2), dtype=float)

    def fit_from_dist(self, dist: np.ndarray, hop: Optional[np.ndarray],
                      mol=None, k_i: Optional[np.ndarray] = None):
        """
        原生 numpy 实现（保留），完成后会把 rho/sigma/cand/frozen 转换为 torch。
        """
        n = dist.shape[0]
        self.N = n

        if hop is not None:
            cand = (hop <= self.K_HOP_MAX)
        else:
            cand = np.ones_like(dist, dtype=bool)
        eye = np.eye(n, dtype=bool)
        cand = cand & (~eye) & np.isfinite(dist)
        self.cand_mask_np = cand

        if k_i is None:
            k_i = self._degree_based_k(mol, n, self.k_offset, self.k_min, self.k_max)

        rho   = np.zeros(n, dtype=float)
        sigma = np.zeros(n, dtype=float)
        frozen_row  = np.zeros(n, dtype=bool)
        frozen_qrow = np.zeros((n, n), dtype=float)

        for i in range(n):
            cand_i = np.where(cand[i])[0]
            m = cand_i.size
            if m == 0:
                rho[i] = 0.0; sigma[i] = 0.0; frozen_row[i] = True
                continue

            d_i = dist[i, cand_i].astype(float)
            pos = d_i[d_i > 0.0]
            rho_i = float(pos.min()) if pos.size > 0 else 0.0
            rho[i] = rho_i

            if m <= 2:
                frozen_row[i] = True
                frozen_qrow[i, cand_i] = 1.0 / m
                sigma[i] = 0.0
                continue

            k_target = float(min(k_i[i], m))

            def k_of_sigma(sig):
                p = self._prob_row_from_sigma(sig, d_i, rho_i)
                return self._k_from_prob(p)

            sigma_i = self._sigma_binary_search_row(
                k_of_sigma, k_target, lo=self.sigma_lo, hi=self.sigma_hi, iters=self.sigma_iters
            )
            sigma[i] = sigma_i

        self.rho_np = rho
        self.sigma_np = sigma
        self.frozen_row_np = frozen_row
        self.frozen_qrow_np = frozen_qrow

        # convert to torch for fast forward/dQdd
        self._convert_params_to_torch()
        return self

    def _convert_params_to_torch(self, device=None, dtype=torch.float32):
        if self.rho_np is None:
            return
        self.rho_t = _ensure_torch(self.rho_np, device=device, dtype=dtype)
        self.sigma_t = _ensure_torch(self.sigma_np, device=device, dtype=dtype)
        self.cand_mask_t = _ensure_torch(self.cand_mask_np.astype(float), device=device, dtype=dtype)
        self.frozen_row_t = _ensure_torch(self.frozen_row_np.astype(float), device=device, dtype=dtype)
        self.frozen_qrow_t = _ensure_torch(self.frozen_qrow_np.astype(float), device=device, dtype=dtype)

    def _qi_matrix(self, D: torch.Tensor) -> torch.Tensor:
        # Ensure params are torch on same device/dtype
        if self.rho_t is None:
            # try converting with D's device/dtype
            self._convert_params_to_torch(device=D.device, dtype=D.dtype)
        n = D.shape[0]
        Ri = self.rho_t[:, None]
        Si = (self.sigma_t[:, None] + EPS)
        X = D - Ri
        X = torch.clamp(X, min=0.0)
        qi = torch.exp(- X / Si)

        # mask non-candidates
        if self.cand_mask_t is not None:
            qi = qi * self.cand_mask_t
        # frozen rows overlay
        if self.frozen_row_t is not None and torch.any(self.frozen_row_t > 0):
            fr = (self.frozen_row_t > 0).to(dtype=D.dtype)
            # frozen_qrow is (n,n) float
            qi = torch.where(fr[:, None] > 0, self.frozen_qrow_t, qi)
        return qi

    def forward(self, D):
        D_t = _ensure_torch(D)
        qi = self._qi_matrix(D_t)
        qj = qi.T
        Q = qi + qj - qi * qj
        return clamp01_torch(Q)

    def dQdd(self, D):
        D_t = _ensure_torch(D)
        # ensure params on same device
        if self.rho_t is None:
            self._convert_params_to_torch(device=D_t.device, dtype=D_t.dtype)
        n = D_t.shape[0]
        Ri = self.rho_t[:, None]
        Rj = self.rho_t[None, :]
        Si = (self.sigma_t[:, None] + EPS)
        Sj = (self.sigma_t[None, :] + EPS)

        qi = self._qi_matrix(D_t)
        qj = qi.T

        mask_i = (D_t > Ri).to(dtype=D_t.dtype)
        mask_j = (D_t > Rj).to(dtype=D_t.dtype)
        if self.frozen_row_t is not None and torch.any(self.frozen_row_t > 0):
            fr = (self.frozen_row_t > 0).to(dtype=D_t.dtype)
            mask_i = mask_i * (1.0 - fr[:, None])
            mask_j = mask_j * (1.0 - fr[None, :])

        dqi_dd = - (qi / Si) * mask_i
        dqj_dd = - (qj / Sj) * mask_j

        dQdd = dqi_dd * (1.0 - qj) + dqj_dd * (1.0 - qi)
        # candidate mask
        if self.cand_mask_t is not None:
            dQdd = dQdd * self.cand_mask_t
        dQdd = dQdd.clone(); dQdd.view(-1)[::n+1] = 0.0
        # sanitize
        dQdd[~torch.isfinite(dQdd)] = 0.0
        return dQdd

    def inv(self, Q):
        raise NotImplementedError("SmoothKRowExpKernel: fuzzy-union 行核无简单 inv.")

    def build_P_from_dist(self, dist: np.ndarray, hop: Optional[np.ndarray],
                          mol=None, k_i: Optional[np.ndarray] = None):
        """Fit using numpy then build P (returns torch.Tensor)."""
        self.fit_from_dist(dist, hop, mol=mol, k_i=k_i)
        # use torch forward now
        D_t = _ensure_torch(dist)
        qi = self._qi_matrix(D_t)
        P = qi + qi.T - qi * qi.T
        return clamp01_torch(P)

# ----------------- 全局默认 KERNEL（torch 版本） -----------------
# If you want to change kernel, assign KERNEL = UMAPLowKernel(...) or SmoothKRowExpKernel(...)
KERNEL: BaseKernel = UMAPLowKernel(a=1.6, b=0.8)


# ----------------- minimal test snippet (不在生产中自动运行) -----------------
if __name__ == "__main__":   # 便于本地单文件测试
    # small torch test: ensure forward/dQdd preserve grad path when used in CE pipeline
    Y = torch.randn(6, 3, requires_grad=True)
    D = pairwise_dist(Y)
    Q = KERNEL.forward(D)
    dQ = KERNEL.dQdd(D)
    print("Q.requires_grad:", Q.requires_grad)   # should be True for operations depending on Y
    # small scalar to backprop (simulate CE scalar)
    s = Q.sum()
    s.backward()
    print("Y.grad is None?", Y.grad is None)
    print("Y.grad norm:", None if Y.grad is None else Y.grad.norm().item())
