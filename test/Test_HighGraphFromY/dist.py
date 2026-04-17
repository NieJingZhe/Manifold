import torch
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from rdkit import Chem

from featurizer import Gen39AtomFeatures


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


def compute_augmented_graph_distance(mol, alpha=1.0, beta=1.0):
    """
    Augmented atom-wise distance:
        D_aug = D_topo + alpha * D_feat + beta * D_edge
    - D_topo: unweighted hop-count shortest-path (graph topology only)
    - D_feat: Euclidean distance between atom feature vectors
    - D_edge: shortest-path length under edge-feature norm weights
    """
    smiles = Chem.MolToSmiles(mol)
    data = type('obj', (object,), {'smiles': smiles})()
    featurizer = Gen39AtomFeatures()
    data = featurizer(data)
    node_features = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    N = node_features.size(0)

    # ---- Build graph with TWO weights per edge: hop (1.0) and w_feat (||e||2)
    G = nx.Graph()
    for i in range(edge_index.size(1)):
        u, v = edge_index[:, i].tolist()
        e = edge_attr[i].detach().cpu().numpy()
        G.add_edge(u, v, hop=1.0, w_feat=float(np.linalg.norm(e)), edge_feat=e)

    # ---- D_edge: shortest path under w_feat
    D_edge = np.zeros((N, N))
    for i in range(N):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight='w_feat')
        for j in range(N):
            D_edge[i, j] = lengths.get(j, np.inf)

    # ---- D_feat: Euclidean over atom features
    node_np = node_features.detach().cpu().numpy()
    D_feat = cdist(node_np, node_np, metric='euclidean')

    # ---- D_topo: hop-count shortest path (pure topology)
    D_topo = np.zeros((N, N))
    for i in range(N):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight='hop')
        for j in range(N):
            D_topo[i, j] = lengths.get(j, np.inf)

    # ---- Safe per-component normalization (ignore inf)
    def _safe_norm(M):
        M = np.array(M, dtype=float)
        M = np.nan_to_num(M, nan=1e6, posinf=1e6)
        mmax = np.max(M[M < 1e6]) if np.any(M < 1e6) else 1.0
        return M / (mmax + 1e-12)

    D_topo = _safe_norm(D_topo)
    D_feat = _safe_norm(D_feat)
    D_edge = _safe_norm(D_edge)

    D_aug = D_topo + alpha * D_feat + beta * D_edge
    D_aug = np.nan_to_num(D_aug, nan=1e6, posinf=1e6)
    D_aug = D_aug / (D_aug.max() + 1e-12)
    return D_aug


# def compute_augmented_graph_distance(mol, 
#                                       alpha=1.0, 
#                                       beta=1.0):
#     """
#     Compute an augmented distance matrix D_aug ∈ ℝ^{N×N} for atoms in a molecule,
#     integrating three components:
#         - topological path distance based on edge weights (D_topo)
#         - node feature distance via Euclidean metric (D_feat)
#         - edge-aggregated path distance based on edge attributes (D_edge)

#     D_aug = D_topo + α·D_feat + β·D_edge

#     Args:
#         mol (rdkit.Chem.Mol): Input RDKit molecule object.
#         alpha (float): Weight for node feature distance.
#         beta (float): Weight for edge feature path distance.

#     Returns:
#         D_aug (ndarray): Normalized augmented distance matrix of shape (N, N).
#     """

#     smiles = Chem.MolToSmiles(mol)
#     data = type('obj', (object,), {'smiles': smiles})()  # mock object
#     featurizer = Gen39AtomFeatures()
#     data = featurizer(data)
#     node_features=data.x
#     edge_index=data.edge_index
#     edge_attr=data.edge_attr


#     N = node_features.size(0)

#     # Step 1: Build weighted NetworkX graph from edge_index and edge_attr
#     G = nx.Graph()
#     edge_weights = {}
#     for i in range(edge_index.size(1)):
#         u, v = edge_index[:, i].tolist()
#         edge_feat = edge_attr[i].detach().cpu().numpy()
#         weight = np.linalg.norm(edge_feat)  # ||e||₂
#         G.add_edge(u, v, weight=weight, edge_feat=edge_feat)
#         edge_weights[(u, v)] = edge_feat
#         edge_weights[(v, u)] = edge_feat  # Undirected

#     # Step 2: Compute edge-aggregated shortest path matrix D_edge ∈ ℝ^{N×N}
#     D_edge = np.zeros((N, N))
#     for i in range(N):
#         for j in range(N):
#             if i == j:
#                 D_edge[i, j] = 0
#             else:
#                 try:
#                     path = nx.shortest_path(G, source=i, target=j)
#                     total_edge_feat = 0
#                     for k in range(len(path) - 1):
#                         e_feat = edge_weights.get((path[k], path[k + 1]), None)
#                         if e_feat is not None:
#                             total_edge_feat += np.linalg.norm(e_feat)
#                     D_edge[i, j] = total_edge_feat
#                 except nx.NetworkXNoPath:
#                     D_edge[i, j] = np.inf

#     # Step 3: Node feature Euclidean distance D_feat ∈ ℝ^{N×N}
#     node_feat_np = node_features.detach().cpu().numpy()
#     D_feat = cdist(node_feat_np, node_feat_np, metric='euclidean')

#     # Step 4: Dijkstra topological distance D_topo ∈ ℝ^{N×N}
#     D_topo = np.zeros((N, N))
#     for i in range(N):
#         lengths = nx.single_source_dijkstra_path_length(G, i, weight='weight')
#         for j in range(N):
#             D_topo[i, j] = lengths.get(j, np.inf)

#     # Step 5: Combine components into D_aug and normalize
#     D_aug = D_topo + alpha * D_feat + beta * D_edge
#     D_aug = np.nan_to_num(D_aug, nan=1e6, posinf=1e6)
#     D_aug = D_aug / D_aug.max()

#     return D_aug


from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN

def compute_AE_tanimoto_distance(mol, radius=2, nBits=2048):
    """
    Compute the Tanimoto-based atom distance matrix using Atom Environment (AE) fingerprints.
    
    D[i, j] = 1 - Tanimoto(AE_i, AE_j)

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.
        radius (int): Radius for atom environment.
        nBits (int): Bit size for Morgan fingerprints.

    Returns:
        dist_matrix (ndarray): Atom-wise distance matrix of shape (N_atoms, N_atoms).
    """
    num_atoms = mol.GetNumAtoms()
    atom_fps = []

    # Step 1: Extract local atom environments and compute fingerprints
    for atom_idx in range(num_atoms):
        env_ids = FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        env_mol = Chem.PathToSubmol(mol, env_ids)
        Chem.GetSymmSSSR(env_mol)  # Ensure RingInfo is initialized
        fp = AllChem.GetMorganFingerprintAsBitVect(env_mol, radius, nBits=nBits)
        atom_fps.append(fp)

    # Step 2: Compute pairwise Tanimoto similarities
    sim_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(num_atoms):
            sim = DataStructs.TanimotoSimilarity(atom_fps[i], atom_fps[j])
            sim_matrix[i, j] = sim

    # Step 3: Convert similarity to distance
    dist_matrix = 1.0 - sim_matrix
    return dist_matrix



from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
import numpy as np

def compute_embed3d_distance(mol, embed=True, optimize=True, maxIters=200):


    if mol is None:
        raise ValueError("Invalid SMILES")

    # 保持不加 H
    if embed:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if optimize:
        AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters)

    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])

    dist_matrix = cdist(coords, coords, metric='euclidean')
    return dist_matrix #/ dist_matrix.max()

