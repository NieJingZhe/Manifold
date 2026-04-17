import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def _rdkit_bond_targets_from_etkdg(mol):
    """用一次 ETKDG 得到每条键的“参考长度”，作为软目标；不加氢也可。"""
    mol2 = Chem.Mol(mol)
    AllChem.EmbedMolecule(mol2, AllChem.ETKDG())
    conf = mol2.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol2.GetNumAtoms())])
    targets = {}
    for b in mol2.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        d = np.linalg.norm(pos[i] - pos[j])
        targets[(i, j)] = targets[(j, i)] = float(d)
    return targets

def bond_spring_loss(Y, mol, targets=None):
    """∑(‖Yi-Yj‖ - Lij)^2，仅对成键原子对。"""
    if targets is None:
        targets = _rdkit_bond_targets_from_etkdg(mol)
    loss = 0.0
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        dij = np.linalg.norm(Y[i] - Y[j]) + 1e-9
        lij = targets.get((i, j), dij)   # 若没目标就不推动
        loss += (dij - lij) ** 2
    return loss / max(1, mol.GetNumBonds())

def _hop_dist_matrix(mol, kmax=3):
    """BFS 得到 hop 数矩阵；用于排斥时排除 1/2 邻。"""
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append(j); adj[j].append(i)
    INF = 10**9
    H = np.full((n, n), INF, dtype=int)
    for s in range(n):
        H[s, s] = 0
        q = [s]
        while q:
            u = q.pop(0)
            for v in adj[u]:
                if H[s, v] == INF:
                    H[s, v] = H[s, u] + 1
                    if H[s, v] < kmax:  # 小优化
                        q.append(v)
    return H

def repulsion_loss(Y, mol, cutoff=1.2, exclude_hop_le=2):
    """
    对非键对施加软排斥：max(0, cutoff - d)^2。
    默认排除 hop≤2 的对（邻居和次邻），只惩罚“真正不该撞在一起”的远端对。
    """
    n = mol.GetNumAtoms()
    H = _hop_dist_matrix(mol, kmax=exclude_hop_le+1)
    loss = 0.0; cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            if H[i, j] <= exclude_hop_le: 
                continue
            d = np.linalg.norm(Y[i] - Y[j]) + 1e-9
            pen = max(0.0, cutoff - d)
            if pen > 0:
                loss += pen * pen
                cnt += 1
    return loss / max(1, cnt)

def center_and_rescale(Y, target_rms=1.0):
    """每个 epoch：居中并把坐标 RMS 缩放到 target_rms，防止数值漂移或坍缩。"""
    Y = Y - Y.mean(axis=0, keepdims=True)
    rms = np.sqrt((Y**2).mean())
    if rms < 1e-8:
        return Y
    return Y * (target_rms / rms)
