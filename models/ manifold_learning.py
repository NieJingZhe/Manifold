import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from vis import visualize_and_save_frame
from render import render_html_to_image
from constrains import center_and_rescale  
from losses import CE, CE_gradient
from kernels import (
    KERNEL, UMAPLowKernel, GaussianKernel, StudentTKernel, pairwise_dist,
    UMAPRowExpKernel, UMAPRowFamilyKernel, SmoothKRowExpKernel, find_ab_params,
)

class ManifoldRunner:
    def __init__(self, min_dist=0.9, spread=1.0, seed=42):
        """
        初始化时选择核函数（默认用 UMAP Low Kernel）。
        """
        self.seed = seed
        a, b = find_ab_params(min_dist=min_dist, spread=spread)
        self.kernel = UMAPLowKernel(a, b)
        self._eps = 1e-12

    def teacher_coords_from_smiles(self, smiles, optimize=True):
        """
        用 RDKit ETKDG 生成 3D 坐标（仅重原子，保持与你可视化的原子顺序一致）。
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # ETKDG 更稳
        params = AllChem.ETKDGv3()
        params.randomSeed = self.seed
        params.useSmallRingTorsions = True
        ok = AllChem.EmbedMolecule(mol, params)
        if ok != 0:
            raise RuntimeError("ETKDG embedding failed.")
        if optimize:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass
        conf = mol.GetConformer()
        mol_noH = Chem.RemoveHs(mol)
        idx_map = [conf.GetAtomPosition(a.GetIdx()) for a in mol_noH.GetAtoms()]
        Y_true = np.array([[p.x, p.y, p.z] for p in idx_map], dtype=float)
        return mol_noH, Y_true

    def q_from_Y(self, Y: np.ndarray):
        """
        给定坐标 Y，计算核矩阵 Q 和距离矩阵 D。
        """
        D = pairwise_dist(Y)
        Q = self.kernel.forward(D)
        return Q, D



mol, Y_true = teacher_coords_from_smiles(mol, seed=2025, optimize=True)
Y_true = center_and_rescale(Y_true, target_rms=1.0)
P, _ = q_from_Y(Y_true)   # 这里把“高维 Q*”记为 P（与你原代码接口一致）

# 初始化 Y（随机）并规范化
N = Y_true.shape[0]
rng = np.random.default_rng(123)
Y = rng.normal(size=(N, 3))
Y = center_and_rescale(Y, target_rms=1.0)

# 优化配置
lr = 0.01
epochs = 1000

output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

CE_array = []
for i in range(epochs):
    # 梯度 & 更新
    g = CE_gradient(P, Y)
    Y = Y - lr * g
    Y = center_and_rescale(Y, target_rms=1.0)

    # 记录 CE（用 off-diagonal 的均值）
    CE_mat = CE(P, Y)
    off = ~np.eye(N, dtype=bool)
    CE_current = CE_mat[off].mean()
    CE_array.append(CE_current)
