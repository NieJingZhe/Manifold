import numpy as np
from typing import Optional, Dict, Any
from models import (
    compute_AE_tanimoto_distance_np,
    compute_augmented_graph_distance_np,
    compute_embed3d_distance_np,
    hop_matrix_from_mol,
)


def _extract_distance_matrix(dist_used, prefer_key: Optional[str] = "D_aug"):
    """从可能的返回类型（np.ndarray / dict / list-like）中提取方阵距离矩阵。

    优先查找 prefer_key，然后查找常见键名，再回退到第一个能识别为方阵的 array-like 值。
    如果无法提取，将抛出 ValueError。
    """
    if isinstance(dist_used, dict):
        # 优先取 prefer_key
        if prefer_key is not None and prefer_key in dist_used:
            return np.asarray(dist_used[prefer_key], dtype=float)

        # 常见候选键
        for key in ("D_aug", "D_topo", "D_edge", "D_feat", "dist", "matrix", "D"):
            if key in dist_used:
                return np.asarray(dist_used[key], dtype=float)

        # 回退：第一个能解析成方阵的 value
        for v in dist_used.values():
            try:
                arr = np.asarray(v, dtype=float)
            except Exception:
                continue
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                return arr

        raise ValueError(
            "Cannot extract a square distance matrix from dict. Available keys: %s"
            % (list(dist_used.keys()),)
        )
    else:
        arr = np.asarray(dist_used, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("dist must be a square (n x n) array, got shape: %s" % (arr.shape,))
        return arr


def build_high_dim_probabilities(
    mol=None,
    dist: Optional[np.ndarray] = None,
    dist_name: str = "D2",
    compute_D_params: Optional[Dict[str, Any]] = None,
    hop: Optional[np.ndarray] = None,
    K_HOP_MAX: int = 2,
    k_min: int = 3,
    k_max: int = 8,
    default_k_when_no_deg: float = 6.0,
    sigma_search_iters: int = 32,
    sigma_hi_init: float = 1.0,
    verbose: bool = False,
    data=None,
):
    """
    构造高维概率图（返回对称化的 P 矩阵以及中间产物）。

    返回 dict:
      - "P": 对称化的概率矩阵 (n x n)
      - "prob": 非对称未归一化的 prob 矩阵
      - "sigma": 每行二分搜索得到的 sigma
      - "dist": 使用的距离矩阵 (n x n)
      - "hop": 使用的 hop 矩阵或 None
    """

    params = compute_D_params or {}

    # ---------- 1. 准备距离矩阵 ----------
    if dist is None:
        if mol is None:
            raise ValueError("必须提供 dist 矩阵，或提供 mol 以便根据 dist_name 计算。")

        if verbose:
            print(f"[build_high_dim_probabilities] computing distance {dist_name} ...")

        if dist_name == "D1":
            raw = compute_augmented_graph_distance_np(mol=mol, data=data, **params)
            dist_mat = _extract_distance_matrix(raw, prefer_key="D_aug")
        elif dist_name == "D2":
            raw = compute_AE_tanimoto_distance_np(mol=mol, **params)
            # AE / tanimoto 函数通常返回矩阵，但也可能返回 dict
            dist_mat = _extract_distance_matrix(raw, prefer_key=None)
        elif dist_name == "D3":
            raw = compute_embed3d_distance_np(mol=mol, **params)
            dist_mat = _extract_distance_matrix(raw, prefer_key=None)
        else:
            raise ValueError(f"未知 dist_name: {dist_name}")
    else:
        dist_mat = np.asarray(dist, dtype=float)
        if dist_mat.ndim != 2 or dist_mat.shape[0] != dist_mat.shape[1]:
            raise ValueError("传入的 dist 必须是方阵 (n x n). got: %s" % (dist_mat.shape,))

    n = int(dist_mat.shape[0])

    if verbose:
        print(f"[build_high_dim_probabilities] distance matrix shape: {dist_mat.shape}")

    # ---------- 2. 准备 hop 矩阵 ----------
    hop_used = hop
    if hop_used is None and mol is not None:
        try:
            hop_used = hop_matrix_from_mol(mol)
        except Exception as e:
            if verbose:
                print(f"[build_high_dim_probabilities] hop_matrix_from_mol failed: {e}. Ignoring hop.")
            hop_used = None

    if hop_used is not None:
        hop_used = np.asarray(hop_used, dtype=float)
        if hop_used.shape != (n, n):
            if verbose:
                print("[build_high_dim_probabilities] Warning: hop 矩阵形状与 dist 不匹配，忽略 hop 筛选。")
            hop_used = None

    # ---------- 3. 计算每个节点目标 k_i ----------
    try:
        if mol is not None:
            deg = np.array([a.GetDegree() for a in mol.GetAtoms()], dtype=int)
            if deg.shape[0] != n:
                # 不匹配则忽略 deg
                deg = None
        else:
            deg = None

        if deg is not None:
            k_i = np.clip(deg + 2, k_min, k_max).astype(float)
        else:
            k_i = np.full(n, default_k_when_no_deg, dtype=float)
    except Exception:
        k_i = np.full(n, default_k_when_no_deg, dtype=float)

    # ---------- 4. 辅助函数 ----------
    def prob_row_from_sigma(sigma, d_row, rho_i):
        x = d_row - rho_i
        x = np.where(x < 0.0, 0.0, x)
        return np.exp(- x / (sigma + 1e-12))

    def k_from_prob(prob_row):
        s = float(prob_row.sum())
        return np.power(2.0, s) - 1.0

    def sigma_binary_search_row(k_of_sigma, fixed_k, lo=1e-6, hi=sigma_hi_init, iters=sigma_search_iters):
        # expand hi until >= fixed_k (robust策略)
        try:
            while k_of_sigma(hi) < fixed_k:
                hi *= 2.0
                if hi > 1e6:
                    break
        except Exception:
            hi = sigma_hi_init

        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            try:
                if k_of_sigma(mid) < fixed_k:
                    lo = mid
                else:
                    hi = mid
            except Exception:
                lo = mid
                hi *= 2.0
        return 0.5 * (lo + hi)

    # ---------- 5. 主循环 ----------
    prob = np.zeros((n, n), dtype=float)
    sigma_array = np.zeros(n, dtype=float)

    for i in range(n):
        # 全量可选集合：非自身且距离有限
        finite_mask = np.isfinite(dist_mat[i])
        cand = np.where((np.arange(n) != i) & finite_mask)[0]

        # 若有 hop 限制，则再筛一次
        if hop_used is not None:
            hop_mask = hop_used[i] <= K_HOP_MAX
            cand = cand[hop_mask[cand]]

        if cand.size == 0:
            sigma_array[i] = 0.0
            if verbose and ((i + 1) % 200 == 0):
                print(f"Row {i}: no candidates (all inf or excluded by hop)")
            continue

        if cand.size <= 2:
            prob[i, cand] = 1.0 / float(cand.size)
            sigma_array[i] = 0.0
            continue

        d_i = dist_mat[i, cand].astype(float)

        pos = d_i[d_i > 0.0]
        rho_i = float(pos.min()) if pos.size > 0 else 0.0

        k_target = float(min(k_i[i], cand.size))

        def k_of_sigma(sig):
            p = prob_row_from_sigma(sig, d_i, rho_i)
            return k_from_prob(p)

        sigma_i = sigma_binary_search_row(k_of_sigma, k_target, lo=1e-6, hi=sigma_hi_init, iters=sigma_search_iters)
        sigma_array[i] = sigma_i

        p_row = prob_row_from_sigma(sigma_i, d_i, rho_i)
        prob[i, cand] = p_row

        if verbose and ((i + 1) % 100 == 0 or i == n - 1):
            print(f"Sigma search row {i+1}/{n}")

    if verbose:
        print("Mean sigma =", float(sigma_array.mean()))

    # ---------- 6. 对称化（fuzzy union） ----------
    P = prob + prob.T - prob * prob.T

    return {
        "P": P,
        # "prob": prob,
        # "sigma": sigma_array,
        # "dist": dist_mat,
        # "hop": hop_used,
    }


# ---------------- 简要示例 ----------------
# result = build_high_dim_probabilities(mol=mol, dist_name="D2", compute_D_params={'radius':2,'nBits':2048})
# P = result['P']




