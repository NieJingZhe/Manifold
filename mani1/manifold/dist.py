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
def prob_low_dim(Y, a, b, eps=1e-12):
    """
    返回低维umap矩阵 Q_ij = 1 / (1 + a * (d_ij^2)^b)
    Y: (N, d) 的 torch.Tensor
    a, b: torch 或 float，标量
    """
    # 计算欧式距离矩阵 d_ij
    a = torch.as_tensor(a, dtype=Y.dtype, device=Y.device)
    b = torch.as_tensor(b, dtype=Y.dtype, device=Y.device)

    D = torch.cdist(Y, Y, p=2)    # (N, N)
    s = D.pow(2) + eps            # d_ij^2

    # 相似度矩阵
    aff = 1.0 / (1.0 + a * torch.pow(s, b))

    # 对角线强制为 0
    aff.fill_diagonal_(0.0)

    return aff


