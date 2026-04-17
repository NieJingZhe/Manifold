import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool,global_add_pool,global_max_pool
from torch_scatter import scatter_add, scatter_mean
import numpy as np
from .gnnconv import GNN_node,MLP
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import shortest_path


def compute_high_dim_adj(edge_index: torch.LongTensor,
                         num_nodes: int,
                         sigma_H: float) -> torch.Tensor:

    # 
    A = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)  # shape [N,N]
    # 
    dist_matrix = shortest_path(A, directed=False, unweighted=True)  # numpy [N,N]
    # 3. 
    W = np.exp(- dist_matrix / sigma_H)
    # turn to torch.Tensor
    return torch.from_numpy(W).float()


def compute_low_dim_adj(pos: torch.Tensor,
                        sigma_L: float) -> torch.Tensor:


    diff = pos.unsqueeze(1) - pos.unsqueeze(0)    # [N,1,D] - [1,N,D] -> [N,N,D]
    dist = diff.norm(dim=-1)                      # [N,N]

    W = torch.exp(- dist / sigma_L)
    return W

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


    def forward(self, data):
        node_feat = self.gnn(data)
        graph_feat = self.pool(node_feat, data.batch)
        W_H = compute_high_dim_adj(data.edge_index, data.num_nodes, self.args.sigma_H)

        #print(f"node_feat shape: {node_feat.shape}")
        
        pred_pos = self.mlp(node_feat)
        W_L = compute_low_dim_adj(pred_pos, self.args.sigma_L)

        #print(f"data.pos shape: {data.pos.shape}")
        #print(f"pred_pos shape: {pred_pos.shape}")

        pos_loss = F.mse_loss(pred_pos, data.pos)  # pos_pred, pos_true: [N, 3]
        device = node_feat.device  # or data.x.device
        W_H = W_H.to(device)
        W_L = W_L.to(device)
        mani_loss= torch.sum((W_H - W_L) ** 2)
        return pred_pos, graph_feat, pos_loss,mani_loss






