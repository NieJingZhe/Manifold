import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool
from torch_scatter import scatter_add, scatter_mean
import numpy as np
from torch.nn.functional import binary_cross_entropy_with_logits 
from .gnnconv import GNN_node,MLP
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist, squareform
from rdkit import Chem
from rdkit.Geometry import Point3D
from manifold import center_and_rescale,prob_low_dim,compute_embed3d_distance
from sklearn.manifold import SpectralEmbedding
from manifold import coords2dict_mds,coords2dict_tch,compute_low_dim_adj,prob_low_dim
from sklearn.metrics.pairwise import euclidean_distances
from typing import Optional, Tuple, Dict, Any, List
from manifold.kernels import KERNEL, UMAPLowKernel, GaussianKernel, StudentTKernel, pairwise_dist, UMAPRowExpKernel, UMAPRowFamilyKernel, SmoothKRowExpKernel, find_ab_params
from torch_geometric.nn.conv.pna_conv import PNAConv

class GNNEncoder(nn.Module):
    def __init__(self, args, config,deg = None):
        super(GNNEncoder, self).__init__()
        self.args = args
        self.config = config    
        emb_dim = args.emb_dim
        self.gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
                            drop_ratio=args.dropout, gnn_type=args.gnn_type,deg = deg)
        self.mlp =MLP(input_dim=emb_dim, hidden_dim=args.mlp_hidden, output_dim=3,num_layers=args.mlp_layer)
        #这里的Z、rho、sigma都是未来可能运用到高维umap核可能用到的东西所以保留
        self.mlp4Z = MLP(input_dim=emb_dim, hidden_dim=args.mlp_hidden, output_dim=3,num_layers=args.mlp_layer)
        self.mlp4rho = MLP(input_dim=emb_dim, hidden_dim=args.mlp_hidden, output_dim=1,num_layers=args.mlp_layer)
        self.mlp4sigma = MLP(input_dim=emb_dim, hidden_dim=args.mlp_hidden, output_dim=1,num_layers=args.mlp_layer)
        #self.rowhead = RowAdaptiveHead(family='umap', a=1.6, b=0.8, enforce_positive='softplus')
        #attention mechanism，可能采用attention机制
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=args.attn_heads, batch_first=True)
        self.sim_proj = nn.Linear(emb_dim, 1)
        self.i = 0

        #这也是目前没用到的东西
        if args.pooling_type == 'mean':
            self.pool = global_mean_pool
        elif args.pooling_type == 'add':
            self.pool = global_add_pool
        elif args.pooling_type =='max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling method: {args.pooling_type}")
    

    def forward(self, data,epoch):
        #目前只考虑用teacher的方式，因为以前考虑过直接生成坐标在这里优化所以保存了pred_pos等坐标相关的变量
        #D1、D2、D3是其他的可能作为我们外面的概率图的内容，所以保留
        node_feat = self.gnn(data)
        graph_feat = self.pool(node_feat, data.batch)
        if self.args.train_model=='teacher':
            #W_H = compute_high_dim_adj(data.edge_index, data.num_nodes, self.args.sigma_H)
            
            Zi = self.mlp4Z(node_feat)  # [N, Zi]
            #pred_pos = center_and_rescale(pred_pos, target_rms=1.0)
 
            Phat = prob_low_dim(Zi, data.a.mean(), data.b.mean())                      
            Q = prob_low_dim(data.pos, data.a.mean(), data.b.mean())
            CE = F.binary_cross_entropy(Phat,Q,reduction='mean')
            if self.i == 0:
                print("这分别是Phat和Q的shape：",Phat.shape,Q.shape)
                self.i=1

            if self.args.verbose==True:
                print("Phat stats:", Phat.min().item(), Phat.mean().item(), Phat.max().item())
                print("Q stats:", Q.min().item(), Q.mean().item(), Q.max().item())
                print("CE stats:", CE.min().item(), CE.mean().item(), CE.max().item())

            pred_pos = torch.zeros(Zi.shape[0],3)# 以全0的方式，暂时屏蔽这里的坐标
            pred_pos = pred_pos.to(data.pos.device)
            pos_loss = F.mse_loss(pred_pos, data.pos)  # pos_pred, pos_true: [N, 3]
            #device = node_feat.device
            # W_H = W_H.to(device)
            # W_L = compute_low_dim_adj(pred_pos, self.args.sigma_L).to(device)
            mani_loss = CE
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
    


class InferenceModel(nn.Module):
    #还没写完的推理model
    def __init__(self, args, config,Pstar):
        super().__init__()
        self.args = args
        self.config = config    
                #获取Pstar的长度
        n_samples = Pstar.shape[0]
        #随机生成n_samples个长度为3的向量
        self.pred_pos = torch.rand(n_samples, 3)
        self.Pstar = Pstar
    def forward(self):
        Q =compute_low_dim_adj(self.pred_pos)

        CE = F.binary_cross_entropy(self.Pstar,Q,reduction='mean')
        return CE,self.pred_pos


        
