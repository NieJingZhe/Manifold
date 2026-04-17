import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool
from torch_scatter import scatter_add, scatter_mean
import numpy as np
from .gnnconv import GNN_node,MLP
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, squareform
from rdkit import Chem
from rdkit.Geometry import Point3D
from .losses import CE
from sklearn.manifold import SpectralEmbedding
from .dist2coords import coords2dict_mds,coords2dict_tch
def compute_high_dim_adj(edge_index: torch.LongTensor,
                         num_nodes: int,
                         sigma_H: float) -> torch.Tensor:
    A = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)  # shape [N,N]
    dist_matrix = shortest_path(A, directed=False, unweighted=True)  # numpy [N,N]
    W = np.exp(- dist_matrix / sigma_H)
    # turn to torch.Tensor
    return torch.from_numpy(W).float()


def compute_low_dim_adj(pos: torch.Tensor,
                        sigma_L: float) -> torch.Tensor:

    diff = pos.unsqueeze(1) - pos.unsqueeze(0)    # [N,1,D] - [1,N,D] -> [N,N,D]
    dist = diff.norm(dim=-1)                      # [N,N]
    pos_np = pos.detach().cpu().numpy()
    # c_dist = cdist(pos_np, pos_np)
    # if dist==c_dist:
    #     print("dist and cdist are the same")
    W = torch.exp(- dist / sigma_L)
    return W
def prob_low_dim(Y):
    """
    返回低维相似度矩阵 Q_ij = 1 / (1 + a * (d_ij^2)^b)
    （与原来思路一致，但写法更清晰、数值稳定）
    """
    D = euclidean_distances(Y, Y)        # d_ij
    s = np.square(D) + EPS               # s = d_ij^2
    # 注意：a, b 在外部已通过 curve_fit 得到并存在变量中
    aff = 1.0 / (1.0 + a * np.power(s, b))
    np.fill_diagonal(aff, 0.0)
    return aff    # 每对 (i,j) 的“概率”/相似度（在 (0,1]）


def center_and_rescale(Y, target_rms=1.0):
    """每个 epoch：居中并把坐标 RMS 缩放到 target_rms，防止数值漂移或坍缩。"""
    Y = Y - Y.mean(dim=0, keepdim=True)
    rms = torch.sqrt((Y**2).mean())
    if rms < 1e-8:
        return Y
    return Y * (target_rms / rms)

class GNNEncoder(nn.Module):
    def __init__(self, args, config):
        super(GNNEncoder, self).__init__()
        self.args = args
        self.config = config    
        emb_dim = args.emb_dim
        self.gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
                            drop_ratio=args.dropout, gnn_type=args.gnn_type)
        self.mlp =MLP(input_dim=emb_dim, hidden_dim=args.mlp_hidden, output_dim=3,num_layers=args.mlp_layer)        
        if args.pooling_type == 'mean':
            self.pool = global_mean_pool
        elif args.pooling_type == 'add':
            self.pool = global_add_pool
        elif args.pooling_type =='max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling method: {args.pooling_type}")


    def forward(self, data,epoch):

        node_feat = self.gnn(data)
        graph_feat = self.pool(node_feat, data.batch)
        if self.args.train_model=='teacher':
            #W_H = compute_high_dim_adj(data.edge_index, data.num_nodes, self.args.sigma_H)
            
            pred_pos = self.mlp(node_feat)  # [N, 3]
            pred_pos = center_and_rescale(pred_pos, target_rms=1.0)

            CE_list = CE(data.P, pred_pos)
            pos_loss = F.mse_loss(pred_pos, data.pos)  # pos_pred, pos_true: [N, 3]
            #device = node_feat.device
            # W_H = W_H.to(device)
            # W_L = compute_low_dim_adj(pred_pos, self.args.sigma_L).to(device)
            mani_loss = sum([ce.sum() for ce in CE_list])
        elif self.args.train_model =="D1":        
            if epoch == 0:
                model = SpectralEmbedding(n_components=3, n_neighbors = 3,
                          affinity='precomputed', random_state=0)
                Y = model.fit_transform(data.P1)
            Q = prob_low_dim(Y)
            CE_list = CE(data.P1,Q )
            mani_loss = sum([ce.sum() for ce in CE_list])
            pos_loss = 0
            #pred_pos = coords2dict_mds(data.dist_matrix)
        elif self.args.train_model =="D2":
            if epoch == 0:
                model = SpectralEmbedding(n_components=3, n_neighbors = 3,
                          affinity='precomputed', random_state=0)
                Y = model.fit_transform(data.P2)
            Q = prob_low_dim(Y)
            CE_list = CE(data.P2,Q )
            mani_loss = sum([ce.sum() for ce in CE_list])
            pos_loss = 0
        elif self.args.train_model =="D3":
            if epoch == 0:
                model = SpectralEmbedding(n_components=3, n_neighbors = 3,
                          affinity='precomputed', random_state=0)
                Y = model.fit_transform(data.P3)
            Q = prob_low_dim(Y)
            CE_list = CE(data.P3,Q )
            mani_loss = sum([ce.sum() for ce in CE_list])
            pos_loss = 0
        # ----------------------
        # 将预测坐标写入 RDKit mol 的 conformer
        # ----------------------
        mol_list = []
        combined_mol = None


        if hasattr(data, 'batch'):
            batch = data.batch
            if batch.numel() == 0:
                n_mols = 0
            else:
                n_mols = int(batch.max().item()) + 1
        else:
 
            n_mols = len(getattr(data, 'rdmol', []))
            batch = None


        pred_pos_np = pred_pos.detach().cpu().numpy()

        for mi in range(n_mols):

            if batch is not None:
                atom_idx = (batch == mi).nonzero(as_tuple=True)[0]
            else:

                atom_idx = None

            if atom_idx is None or atom_idx.numel() == 0:

                continue

            coords = pred_pos_np[atom_idx.cpu().numpy()]  # shape: [n_atoms_i, 3]


            orig_mol = None
            if hasattr(data, 'rdmol'):
                try:
                    orig_mol = data.rdmol[mi]
                except Exception:
                    orig_mol = None
            if orig_mol is None:

                raise ValueError(
                    f"data.rdmol 缺失或 data.rdmol[{mi}] 为 None。无法为分子索引 {mi} 创建构象。"

                    f"预计该分子的原子数为 {coords.shape[0]}。请在 data 中提供 rdmol 列表(每个元素是 rdkit.Chem.Mol),"

                    "或在模型参数中启用 allow_placeholder 来允许自动创建占位分子(不推荐,可能导致原子序数/拓扑不匹配)。"
                )

            else:

                mol = Chem.Mol(orig_mol)


            conf = Chem.Conformer(mol.GetNumAtoms())
            for ai in range(min(mol.GetNumAtoms(), coords.shape[0])):
                x, y, z = float(coords[ai, 0]), float(coords[ai, 1]), float(coords[ai, 2])
                conf.SetAtomPosition(int(ai), Point3D(x, y, z))

            # 移除旧 conformer 并添加新 conformer
            try:
                mol.RemoveAllConformers()
            except Exception:

                pass
            mol.AddConformer(conf, assignId=True)

            mol_list.append(mol)

        return pred_pos, graph_feat, pos_loss, mani_loss, mol_list









