import torch
import numpy as np
from collections import deque

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN

from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
import torch
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from rdkit import Chem




import numpy as np
from collections import deque
from rdkit import Chem
def hop_matrix_from_mol(mol: Chem.Mol, use_heavy_only=False, kmax=None):
    # 用别名，避免被同名变量遮蔽
    from collections import deque as _deque

    if use_heavy_only:
        idx_map = [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1]
        map_back = {h: i for i, h in enumerate(idx_map)}
        nH = len(idx_map)
        adj = [[] for _ in range(nH)]
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in map_back and j in map_back:
                u, v = map_back[i], map_back[j]
                adj[u].append(v); adj[v].append(u)
        INF = 10**9
        Hh = np.full((nH, nH), INF, dtype=int)
        for s in range(nH):
            Hh[s, s] = 0
            q = _deque([s])
            while q:
                u = q.popleft()
                if kmax is not None and Hh[s, u] >= kmax:
                    continue
                for v in adj[u]:
                    if Hh[s, v] == INF:
                        Hh[s, v] = Hh[s, u] + 1
                        q.append(v)
        # 映射回全原子
        n = mol.GetNumAtoms()
        H = np.full((n, n), INF, dtype=int)
        for a in range(n):
            if a in map_back:
                ua = map_back[a]
                for b in range(n):
                    if b in map_back:
                        vb = map_back[b]
                        H[a, b] = Hh[ua, vb]
                    elif b == a:
                        H[a, b] = 0
            else:
                nbrs = [bb.GetOtherAtomIdx(a) for bb in mol.GetAtomWithIdx(a).GetBonds()]
                r = next((x for x in nbrs if mol.GetAtomWithIdx(x).GetAtomicNum() > 1), None)
                H[a, a] = 0
                if r is not None and r in map_back:
                    ur = map_back[r]
                    for b in range(n):
                        if b in map_back:
                            vb = map_back[b]
                            H[a, b] = H[b, a] = min(H[a, b], Hh[ur, vb] + 1)
        return H
    else:
        # 全原子图 BFS
        n = mol.GetNumAtoms()
        adj = [[] for _ in range(n)]
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            adj[i].append(j); adj[j].append(i)
        INF = 10**9
        H = np.full((n, n), INF, dtype=int)
        for s in range(n):
            H[s, s] = 0
            q = _deque([s])
            while q:
                u = q.popleft()
                if kmax is not None and H[s, u] >= kmax:
                    continue
                for v in adj[u]:
                    if H[s, v] == INF:
                        H[s, v] = H[s, u] + 1
                        q.append(v)
        return H
    


def _all_pairs_shortest_path_from_edges(edge_index_np, weights_np, N, directed=False):
    """
    使用 scipy.sparse.csgraph.shortest_path 一次性计算所有对最短路。
    edge_index_np: shape (2, E)
    weights_np: shape (E,)
    返回 numpy (N,N) 矩阵，无法到达处为 np.inf
    """
    rows = edge_index_np[0]
    cols = edge_index_np[1]
    A = coo_matrix((weights_np, (rows, cols)), shape=(N, N))
    if not directed:
        A = A + A.T
    D = shortest_path(csgraph=A, directed=directed, method='D', unweighted=False)
    return D


def compute_AE_tanimoto_distance_np(mol, radius=2, nBits=2048):
    """返回 numpy (N,N) 的 AE-based Tanimoto 距离矩阵（1 - similarity）。"""
    mol_local = Chem.Mol(mol)
    N = mol_local.GetNumAtoms()
    atom_fps = []
    for atom_idx in range(N):
        env_ids = FindAtomEnvironmentOfRadiusN(mol_local, radius, atom_idx)
        env_mol = Chem.PathToSubmol(mol_local, env_ids)
        # --- 修复 RingInfo 未初始化问题 ---
        Chem.GetSymmSSSR(env_mol)  
        fp = AllChem.GetMorganFingerprintAsBitVect(env_mol, radius, nBits=nBits)
        atom_fps.append(fp)
    sim = np.zeros((N, N), dtype=float)
    for i in range(N):
        sim[i, :] = DataStructs.BulkTanimotoSimilarity(atom_fps[i], atom_fps)
    return (1.0 - sim).astype(np.float32)



def compute_embed3d_distance_np(mol, embed=True, optimize=True, maxIters=200):
    """基于 RDKit conformer 计算欧氏距离，返回 numpy (N,N)。"""
    if mol is None:
        raise ValueError("Invalid mol")
    mol_local = Chem.Mol(mol)
    if mol_local.GetNumConformers() == 0 and embed:
        AllChem.EmbedMolecule(mol_local, AllChem.ETKDG())
    if optimize and mol_local.GetNumConformers() > 0:
        AllChem.UFFOptimizeMolecule(mol_local, maxIters=maxIters)
    conf = mol_local.GetConformer()
    N = mol_local.GetNumAtoms()
    coords = np.zeros((N, 3), dtype=float)
    for i in range(N):
        p = conf.GetAtomPosition(i)
        coords[i, 0] = p.x
        coords[i, 1] = p.y
        coords[i, 2] = p.z
    D = cdist(coords, coords, metric='euclidean')
    return D.astype(np.float32)

def compute_augmented_graph_distance_np(mol, data, alpha=1.0, beta=1.0, large_value=1e6):
    """
    Robust version: accepts featurizer that returns either:
      - torch_geometric.data.Data (has .x/.edge_index/.edge_attr), or
      - dict-like with keys 'x','edge_index','edge_attr', or
      - tuple/list (x, edge_index, edge_attr, ...)

    Converts tensors -> numpy, computes D_feat (node feature euclid),
    D_topo (hop-count shortest path using scipy.sparse.csgraph.shortest_path with weight=1),
    D_edge (shortest path with weight = ||edge_attr||),
    normalizes and returns dict of numpy arrays (float32):
        {'D_aug','D_topo','D_edge','D_feat'}
    """
    # prepare a dict-like datapoint for featurizer that expects indexable mapping
    # (smile2graph4GEOM expects something like datapoint['rdmol'] etc.)
    meta = {'smiles': Chem.MolToSmiles(mol), 'rdmol': mol}

    

    # --- extract x, edge_index, edge_attr robustly ---
    # handle Data-like (has .x) or dict-like or tuple/list
    if hasattr(data, 'x'):
        x_t = data.x
        edge_index_t = data.edge_index
        edge_attr_t = data.edge_attr
    elif isinstance(data, dict):
        x_t = data.get('x', None)
        edge_index_t = data.get('edge_index', None)
        edge_attr_t = data.get('edge_attr', None)
    elif isinstance(data, (tuple, list)):
        # assume conventional ordering: (x, edge_index, edge_attr, ...)
        # allow shorter tuples (e.g., return only x, edge_index, edge_attr)
        if len(data) >= 3:
            x_t, edge_index_t, edge_attr_t = data[0], data[1], data[2]
        else:
            raise ValueError("featurizer returned tuple/list with <3 elements")
    else:
        raise ValueError("Unsupported featurizer return type: %s" % type(data))

    # convert node features to numpy
    if torch.is_tensor(x_t):
        node_features = x_t.detach().cpu().numpy()
    else:
        node_features = np.asarray(x_t)

    # normalize shapes: edge_index -> numpy shape (2, E)
    if torch.is_tensor(edge_index_t):
        edge_index_np = edge_index_t.detach().cpu().numpy()
    else:
        edge_index_np = np.asarray(edge_index_t)

    # If edge_index is shape (E,2), convert to (2,E)
    if edge_index_np.ndim == 2 and edge_index_np.shape[0] == 2:
        pass
    elif edge_index_np.ndim == 2 and edge_index_np.shape[1] == 2:
        edge_index_np = edge_index_np.T
    else:
        # try flattenable or raise
        edge_index_np = np.asarray(edge_index_np)
        if edge_index_np.ndim != 2 or min(edge_index_np.shape) != 2:
            raise ValueError("edge_index has unsupported shape: %s" % (edge_index_np.shape,))

    # edge_attr -> numpy (E, feat_dim) or (E,)
    if torch.is_tensor(edge_attr_t):
        edge_attr_np = edge_attr_t.detach().cpu().numpy()
    else:
        edge_attr_np = np.asarray(edge_attr_t)

    # Ensure consistent E:
    E = edge_index_np.shape[1]
    if edge_attr_np is None:
        # no edge attr: treat as zeros
        edge_attr_np = np.zeros((E,), dtype=float)

    # flatten edge_attr per-edge vector to compute norm
    # if edge_attr_np.ndim == 1 -> already per-edge scalar weight
    if edge_attr_np.ndim == 1:
        w_feat = edge_attr_np.astype(float)
    else:
        w_feat = np.linalg.norm(edge_attr_np.reshape(edge_attr_np.shape[0], -1), axis=1)

    N = node_features.shape[0]

    # D_feat: node features pairwise euclidean
    D_feat = cdist(node_features, node_features, metric='euclidean')

    # D_topo: shortest paths on graph with unit weight
    ones = np.ones(E, dtype=float)
    D_topo = _all_pairs_shortest_path_from_edges(edge_index_np, ones, N, directed=False)

    # D_edge: shortest paths with w_feat weights
    D_edge = _all_pairs_shortest_path_from_edges(edge_index_np, w_feat, N, directed=False)

    # safe normalization helper
    def _safe_norm_np(M):
        M = np.array(M, dtype=float)
        M = np.nan_to_num(M, nan=large_value, posinf=large_value)
        finite_mask = M < large_value
        if np.any(finite_mask):
            mmax = np.max(M[finite_mask])
            if mmax <= 0:
                return M
            return M / (mmax + 1e-12)
        else:
            return M

    D_topo_n = _safe_norm_np(D_topo)
    D_feat_n = _safe_norm_np(D_feat)
    D_edge_n = _safe_norm_np(D_edge)

    D_aug = D_topo_n + alpha * D_feat_n + beta * D_edge_n
    D_aug = np.nan_to_num(D_aug, nan=large_value, posinf=large_value)
    maxval = D_aug.max() if D_aug.size > 0 else 1.0
    D_aug = (D_aug / (maxval + 1e-12)).astype(np.float32)

    return {
        'D_aug': D_aug,
        'D_topo': D_topo.astype(np.float32),
        'D_edge': D_edge.astype(np.float32),
        'D_feat': D_feat.astype(np.float32)
    }