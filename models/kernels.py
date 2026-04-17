# kernels.py
import numpy as np
from dataclasses import dataclass

EPS = 1e-12

# ========== 工具 ==========
def pairwise_dist(Y: np.ndarray) -> np.ndarray:
    diff = Y[:, None, :] - Y[None, :, :]
    return np.sqrt((diff**2).sum(-1) + EPS)

def clamp01(Q: np.ndarray) -> np.ndarray:
    Q = np.clip(Q, 1e-12, 1.0 - 1e-12)
    np.fill_diagonal(Q, 0.0)
    return Q

# ========== UMAP 的 a,b 拟合 ==========
def _umap_target_curve(x, min_dist=0.5, spread=1.0):
    y = np.ones_like(x)
    m = x > min_dist
    y[m] = np.exp(-(x[m] - min_dist) / spread)
    return y

def find_ab_params(min_dist=0.5, spread=1.0):
    """拟合 1/(1+a x^{2b}) 逼近 UMAP 目标曲线（与官方思路一致）"""
    from scipy.optimize import curve_fit
    x = np.linspace(0, 3.0*spread, 300)
    y = _umap_target_curve(x, min_dist=min_dist, spread=spread)
    f = lambda x,a,b: 1.0 / (1.0 + a * (x ** (2.0*b)))
    (a,b), _ = curve_fit(f, x, y, p0=(1.6, 0.8), maxfev=10000)
    return float(a), float(b)

# ========== 基类接口 ==========
class BaseKernel:
    def forward(self, D: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def dQdd(self, D: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def inv(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# ========== 低维 UMAP 核 ==========
@dataclass
class UMAPLowKernel(BaseKernel):
    a: float = 1.6
    b: float = 0.8
    def forward(self, D):
        Q = 1.0 / (1.0 + self.a * (np.maximum(D, EPS) ** (2.0*self.b)))
        return clamp01(Q)
    def dQdd(self, D):
        Dp = np.maximum(D, EPS)
        num = (2.0 * self.a * self.b) * (Dp ** (2.0*self.b - 1.0))
        den = (1.0 + self.a * (Dp ** (2.0*self.b))) ** 2
        return - num / (den + EPS)
    def inv(self, Q):
        x = ((1.0/np.clip(Q,1e-12,1-1e-12) - 1.0) / self.a).clip(min=0.0)
        return x ** (1.0/(2.0*self.b))

# ========== Gaussian 核 ==========
@dataclass
class GaussianKernel(BaseKernel):
    sigma: float = 1.0
    def forward(self, D):
        s2 = self.sigma**2
        Q = np.exp(-(D**2) / (2.0*s2))
        return clamp01(Q)
    def dQdd(self, D):
        s2 = self.sigma**2
        Q = self.forward(D)
        return - (D / s2) * Q
    def inv(self, Q):
        s2 = self.sigma**2
        return np.sqrt(np.maximum(-2.0*s2*np.log(np.clip(Q,1e-12,1-1e-12)), 0.0))

# ========== Student-t 核 ==========
@dataclass
class StudentTKernel(BaseKernel):
    nu: float = 1.0
    def forward(self, D):
        Q = 1.0 / (1.0 + (D**2)/self.nu)
        return clamp01(Q)
    def dQdd(self, D):
        nu = self.nu
        Dp = np.maximum(D, EPS)
        return - (2.0 * Dp / nu) / ((1.0 + (Dp**2)/nu) ** 2 + EPS)
    def inv(self, Q):
        x = (1.0/np.clip(Q,1e-12,1-1e-12) - 1.0) * self.nu
        return np.sqrt(np.maximum(x, 0.0))

# ========== UMAP-P 行自适应核（高维构 P 所用：exp 版本） ==========
class UMAPRowExpKernel(BaseKernel):
    """
    低维也使用高维行核 + fuzzy union：
      q_i(d) = exp(- max(0, d - rho_i) / sigma_i )
      Q = q_i + q_j - q_i*q_j
    需要：rho, sigma 为s长度 N 的数组
    """
    def __init__(self, rho: np.ndarray, sigma: np.ndarray):
        self.rho = np.asarray(rho, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        assert self.rho.ndim == self.sigma.ndim == 1
        assert self.rho.shape == self.sigma.shape

    def _qi(self, D):
        Ri = self.rho[:, None]
        Si = (self.sigma[:, None] + EPS)
        X = D - Ri
        X[X < 0.0] = 0.0
        return np.exp(- X / Si)

    def forward(self, D):
        qi = self._qi(D)
        qj = qi.T
        Q = qi + qj - qi*qj
        return clamp01(Q)

    def dQdd(self, D):
        N = D.shape[0]
        Ri = self.rho[:, None]
        Rj = self.rho[None, :]
        Si = (self.sigma[:, None] + EPS)
        Sj = (self.sigma[None, :] + EPS)

        qi = self._qi(D)
        qj = qi.T

        mask_i = (D > Ri).astype(float)
        mask_j = (D > Rj).astype(float)
        dqi_dd = - (qi / Si) * mask_i
        dqj_dd = - (qj / Sj) * mask_j

        dQdd = dqi_dd*(1.0 - qj) + dqj_dd*(1.0 - qi)
        return dQdd

    def inv(self, Q):
        raise NotImplementedError("Fuzzy-union 的行核无简单解析逆。")

# ========== UMAP-P 同族自适应核（折中版：把 exp 换成 UMAP 族） ==========
class UMAPRowFamilyKernel(BaseKernel):
    """
    q_i(d) = 1 / (1 + a * Z^{2b}),  Z = max(0, (d - rho_i)/sigma_i)
    Q = q_i + q_j - q_i*q_j
    """
    def __init__(self, rho: np.ndarray, sigma: np.ndarray, a: float = 1.6, b: float = 0.8):
        self.rho = np.asarray(rho, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.a = float(a); self.b = float(b)
        assert self.rho.ndim == self.sigma.ndim == 1
        assert self.rho.shape == self.sigma.shape

    def _Z(self, D):
        Ri = self.rho[:, None]
        Si = (self.sigma[:, None] + EPS)
        Z = (D - Ri) / Si
        Z[Z < 0.0] = 0.0
        return Z, Si

    def _qi(self, D):
        Z, _ = self._Z(D)
        qi = 1.0 / (1.0 + self.a * (np.maximum(Z, 0.0) ** (2.0*self.b)))
        return qi

    def forward(self, D):
        qi = self._qi(D)
        qj = qi.T
        Q = qi + qj - qi*qj
        return clamp01(Q)

    def dQdd(self, D):
        Z, Si = self._Z(D)
        qi = self._qi(D)
        qj = qi.T

        Zi = np.maximum(Z, 0.0)
        mask_i = (Zi > 0.0).astype(float)

        # dqi/dd = -(2ab * Z^{2b-1}) / (1 + a Z^{2b})^2 * (1/Si) * mask
        num = (2.0*self.a*self.b) * (Zi ** (2.0*self.b - 1.0))
        den = (1.0 + self.a * (Zi ** (2.0*self.b))) ** 2
        dqi_dd = - (num / (den + EPS)) * (1.0/Si) * mask_i

        dqj_dd = dqi_dd.T  # 对称（Z 用列参数时与 i/j 互换）

        dQdd = dqi_dd*(1.0 - qj) + dqj_dd*(1.0 - qi)
        return dQdd

    def inv(self, Q):
        raise NotImplementedError("Fuzzy-union 的行核无简单解析逆。")


# ===== Smooth-k 自适应行核（UMAP-P 风格，继承 BaseKernel）=====
from typing import Optional
import numpy as np

# 复用你文件中已有的工具；若命名不同，把这两处改成你的函数名
# - pairwise_dist(Y) -> (N,N) 距离
# - clamp01(Q)       -> 将 Q 裁剪到 (1e-12, 1-1e-12)，并对角置 0
EPS = 1e-12

class SmoothKRowExpKernel(BaseKernel):
    """
    行自适应核 + smooth-k + fuzzy union：
      q_i(d) = exp( - max(0, d - rho_i) / sigma_i )
      Q = q_i + q_j - q_i*q_j   （fuzzy union）
    - 候选边：hop<=K 且非对角且 dist 有限
    - 每行目标“近邻数” smooth-k： sum_j q_i(d_ij) = log2(k_i + 1)
      用二分法解 σ_i
    - 候选≤2 的行做“冻结”（等权），避免数值不稳定

    使用流程：
      1) ker = SmoothKRowExpKernel(K_HOP_MAX=2).fit_from_dist(dist, hop, mol)
      2) P  = ker.build_P_from_dist(dist, hop, mol)   # 高维 P（对称）
      3) 低维也想用同族核时：把全局 KERNEL = ker，然后用你的 CE/梯度循环即可
    """

    def __init__(self,
                 K_HOP_MAX: int = 2,
                 k_offset: int = 2,
                 k_min: int = 3,
                 k_max: int = 8,
                 sigma_lo: float = 1e-6,
                 sigma_hi: float = 1.0,
                 sigma_iters: int = 32):
        # 超参
        self.K_HOP_MAX = K_HOP_MAX
        self.k_offset  = k_offset
        self.k_min     = k_min
        self.k_max     = k_max
        self.sigma_lo  = sigma_lo
        self.sigma_hi  = sigma_hi
        self.sigma_iters = sigma_iters
        # 拟合后得到的参数
        self.rho:   Optional[np.ndarray] = None   # (N,)
        self.sigma: Optional[np.ndarray] = None   # (N,)
        self.N:     Optional[int] = None
        # 辅助掩码与冻结行
        self.cand_mask:   Optional[np.ndarray] = None  # (N,N) bool
        self.frozen_row:  Optional[np.ndarray] = None  # (N,)   bool
        self.frozen_qrow: Optional[np.ndarray] = None  # (N,N)  float（仅冻结行有效）

    # --------- 工具：smooth-k σ 行二分 ----------
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
        # smooth-k: sum(p) = log2(k+1)  =>  k = 2^{sum p} - 1
        return np.power(2.0, float(prob_row.sum())) - 1.0

    @staticmethod
    def _degree_based_k(mol, n, k_offset, k_min, k_max):
        if mol is not None:
            try:
                deg = np.array([a.GetDegree() for a in mol.GetAtoms()], dtype=int)
                return np.clip(deg + k_offset, k_min, k_max).astype(float)
            except Exception:
                pass
        # 回退：常数
        return np.full(n, float((k_min + k_max) // 2), dtype=float)

    # --------- 拟合每行 (rho_i, sigma_i) ----------
    def fit_from_dist(self, dist: np.ndarray, hop: Optional[np.ndarray],
                      mol=None, k_i: Optional[np.ndarray] = None):
        """
        dist: (N,N) 距离矩阵（可含 inf）
        hop : (N,N) 最短 hop；None 则默认全连通
        mol : RDKit Mol，仅用于用度数确定 k_i；可为 None
        k_i : (N,) 每行目标 k（可自传）；默认用 deg+offset 并夹 [k_min,k_max]
        """
        n = dist.shape[0]
        self.N = n

        # 候选集合：hop<=K，非自身，dist 有限
        if hop is not None:
            cand = (hop <= self.K_HOP_MAX)
        else:
            cand = np.ones_like(dist, dtype=bool)
        eye = np.eye(n, dtype=bool)
        cand = cand & (~eye) & np.isfinite(dist)
        self.cand_mask = cand

        # 目标 k_i
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

        self.rho = rho
        self.sigma = sigma
        self.frozen_row  = frozen_row
        self.frozen_qrow = frozen_qrow
        return self

    # --------- 行核 qi 与前向 Q(D) ----------
    def _ensure_fit(self):
        if self.rho is None or self.sigma is None:
            raise RuntimeError("SmoothKRowExpKernel: call fit_from_dist(...) first.")

    def _qi_matrix(self, D: np.ndarray) -> np.ndarray:
        self._ensure_fit()
        n = D.shape[0]
        Ri = self.rho[:, None]
        Si = (self.sigma[:, None] + EPS)
        X = D - Ri
        X[X < 0.0] = 0.0
        qi = np.exp(- X / Si)

        # 非候选置 0
        if self.cand_mask is not None:
            qi = qi * self.cand_mask.astype(float)
        # 冻结行覆盖
        if self.frozen_row is not None and self.frozen_row.any():
            fr = self.frozen_row
            qi[fr] = self.frozen_qrow[fr]
        return qi

    def forward(self, D: np.ndarray) -> np.ndarray:
        """Q = q_i + q_j - q_i*q_j；随后 clamp 到 (1e-12, 1-1e-12)，对角置 0。"""
        qi = self._qi_matrix(D)
        qj = qi.T
        Q = qi + qj - qi * qj
        # 用你文件里的 clamp01；若名为 _clamp01 就相应替换
        return clamp01(Q)

    # --------- dQ/dd（用于 CE 梯度） ----------
    def dQdd(self, D: np.ndarray) -> np.ndarray:
        """
        dQ/dd = (dq_i/dd)*(1 - q_j) + (dq_j/dd)*(1 - q_i)
        其中 dq_i/dd = 0 (d <= rho_i)；否则 = - q_i / sigma_i
        冻结行对 dQ/dd 的贡献为 0；非候选边导数置 0。
        """
        self._ensure_fit()
        n = D.shape[0]
        Ri = self.rho[:, None]
        Rj = self.rho[None, :]
        Si = (self.sigma[:, None] + EPS)
        Sj = (self.sigma[None, :] + EPS)

        qi = self._qi_matrix(D)
        qj = qi.T

        mask_i = (D > Ri).astype(float)
        mask_j = (D > Rj).astype(float)
        # 冻结行：梯度置 0
        if self.frozen_row is not None and self.frozen_row.any():
            fr = self.frozen_row
            mask_i[fr] = 0.0        # 行 i 冻结
            mask_j[:, fr] = 0.0     # 列 j 冻结（对称）

        dqi_dd = - (qi / Si) * mask_i
        dqj_dd = - (qj / Sj) * mask_j

        dQdd = dqi_dd * (1.0 - qj) + dqj_dd * (1.0 - qi)

        # 非候选边导数清零
        if self.cand_mask is not None:
            cm = self.cand_mask.astype(float)
            dQdd = dQdd * cm
        np.fill_diagonal(dQdd, 0.0)
        dQdd[~np.isfinite(dQdd)] = 0.0
        return dQdd

    # --------- inv：行核 + fuzzy-union 无解析逆 ----------
    def inv(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError("SmoothKRowExpKernel: fuzzy-union 行核无简单解析逆。")

    # --------- 一键构建高维 P ----------
    def build_P_from_dist(self, dist: np.ndarray, hop: Optional[np.ndarray],
                          mol=None, k_i: Optional[np.ndarray] = None) -> np.ndarray:
        """
        用 dist/hop 拟合 (rho, sigma) 后，直接返回对称 P（fuzzy union）。
        """
        self.fit_from_dist(dist, hop, mol=mol, k_i=k_i)
        qi = self._qi_matrix(dist)
        P = qi + qi.T - qi * qi.T
        return clamp01(P)

# ========== —— 适配你的现有函数名 —— ==========
# 训练前设置全局 KERNEL = 某个实例（例如 UMAPLowKernel(...)）
KERNEL: BaseKernel = UMAPLowKernel(a=1.6, b=0.8)