

# 实验目标与假设

* **目标**：验证在分子活性/属性预测中，加入“概率图 $P$”（空间相邻强度）可**显著提升**基线 GNN 的表现。
* **假设**：$P$ 提供了**距离相关的非局部联系**（top-k 近邻/强相互作用），对分类（AUC/PRC）与回归（RMSE/MAE）均有正增益。

---

# 数据集与指标（建议）

* MoleculeNet 小分子：**BBBP、BACE、Tox21**（分类）；**ESOL、FreeSolv、Lipo**（回归）。
* 切分：**Scaffold split（5 folds）**。
* 指标：ROC-AUC / PRC-AUC（分类），RMSE / MAE（回归）。做 **平均±标准差**，并加 **bootstrap 显著性**。

---

# 变体（Ablation）

| 变体 | 边集                      | 是否用 $P$ | 说明                                                      |
| -- | ----------------------- | ------- | ------------------------------------------------------- |
| V0 | 仅化学键                    | 否       | **2D 基线**（原子/键特征）                                       |
| V1 | 仅化学键                    | 是       | 在“键”这条边上**附加 $P_{ij}$** 作为 **edge feature**             |
| V2 | 键 ∪ **$P$ 的 top-k 非键边** | 是       | **稀疏密集化**：把 $P$ 最强的 k 条非键边加进来，edge\_attr 含 $P$、logit$P$ |
| V3 | **仅 $P$ 的 top-k 边**     | 是       | 纯空间图，验证“无化学键也能有信息”                                      |

> 两类 $P$：
> **老师 $Q^\*$**（由 RDKit ETKDG 坐标 + UMAP 低维核得到，做“上界”）；
> **学生 $\hat P$**（你的 GNN 学的行自适应核 + union 或固定核度量），**训练好后**用于下游任务，体现“可用性”。

---

# 与模型的结合方式（3 个插桩位）

1. **消息缩放（最稳）**：消息 $m_{ij} \leftarrow \underbrace{g(P_{ij})}_{\text{如 } \sigma(\beta\cdot\mathrm{logit}(P))}\cdot \phi(h_j, e_{ij})$
2. **注意力偏置**（Transformer/GAT 风格）：score$_{ij}$$+=$ $\beta\cdot\mathrm{logit}(P_{ij})$
3. **边集扩充**：把 $P$ 的 top-k 非键边并入图（V2/V3）

---

# 关键实现：构图 & 模型（PyTorch Geometric）

## 1) 低维核（老师图 Q\*) & RDKit 构象

```python
# --- UMAP 低维核（和你训练一致）---
def umap_low_kernel(D, a, b, eps=1e-12):
    Q = 1.0 / (1.0 + a * np.clip(D, eps, None)**(2*b))
    np.fill_diagonal(Q, 0.0)
    return np.clip(Q, 1e-12, 1-1e-12)

# --- 用 RDKit 生成一个参考构象并计算 Q* ---
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def coords_from_rdkit_etkdg(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    conf = mol.GetConformer()
    N = mol.GetNumAtoms()
    Y = np.array([[conf.GetAtomPosition(i).x,
                   conf.GetAtomPosition(i).y,
                   conf.GetAtomPosition(i).z] for i in range(N)], dtype=float)
    return mol, Y

def pairwise_euclid(Y, eps=1e-12):
    diff = Y[:, None, :] - Y[None, :, :]
    return np.sqrt((diff**2).sum(-1) + eps)
```

## 2) 用 $P$ 构图（V0–V3）

```python
import torch
from torch_geometric.data import Data

def build_graph_with_P(mol, atom_feat, bond_index, bond_feat,
                       P=None, topk=8, add_nonbond=False, only_P=False):
    """
    mol: RDKit Mol（仅用于取 N）; atom_feat: [N, F]
    bond_index: [2, E_bond]（化学键无向->重复两条）; bond_feat: [E_bond, F_e]
    P: [N, N] 概率图（None 表示不使用）
    topk: 每个节点从 P 里选的非键 top-k
    add_nonbond: 是否把 P 的非键边加进来（V2/V3）
    only_P: 是否只用 P 边（V3）
    """
    N = atom_feat.shape[0]
    # 1) 起始边集：V0/V1/V2 用“化学键”，V3 跳过
    edge_src, edge_dst = ([], [])
    e_attr = []
    if not only_P:
        ei = bond_index
        edge_src += ei[0].tolist()
        edge_dst += ei[1].tolist()
        e_attr.append(bond_feat)

    # 2) 由 P 选非键 top-k 边
    if P is not None and (add_nonbond or only_P):
        # mask 掉已有键（防重复）
        adj_bond = torch.zeros((N, N), dtype=torch.bool)
        if not only_P:
            adj_bond[ei[0], ei[1]] = True
        P_t = torch.tensor(P, dtype=torch.float32)
        for i in range(N):
            scores = P_t[i] * (~adj_bond[i])  # 非键
            scores[i] = 0.0
            if topk > 0:
                idx = torch.topk(scores, k=min(topk, N-1), largest=True).indices
                for j in idx.tolist():
                    edge_src.append(i); edge_dst.append(j)
        # 为 P 边准备 edge_attr：[P, logitP]
        src = torch.tensor(edge_src); dst = torch.tensor(edge_dst)
        P_e = P_t[src, dst].unsqueeze(1)
        logit = torch.log(P_e) - torch.log(1 - P_e + 1e-12)
        P_edge_feat = torch.cat([P_e, logit], dim=1)  # [E_p, 2]
        e_attr.append(P_edge_feat)

    edge_index = torch.stack([torch.tensor(edge_src, dtype=torch.long),
                              torch.tensor(edge_dst, dtype=torch.long)], dim=0)
    edge_attr = torch.cat(e_attr, dim=0) if len(e_attr) > 0 else None
    data = Data(x=torch.tensor(atom_feat, dtype=torch.float32),
                edge_index=edge_index, edge_attr=edge_attr)
    return data
```

> **使用约定**：
>
> * **V0**：`P=None, add_nonbond=False, only_P=False`
> * **V1**：在 bond 边上把 `P_bond = P[i,j]` 拼进 `bond_feat`（或用上面 `P_edge_feat` 直接堆叠）
> * **V2**：`P=..., add_nonbond=True, only_P=False`
> * **V3**：`P=..., only_P=True`

## 3) 模型（GINE + $P$ 消息缩放）

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

class PGate(nn.Module):
    """把 edge_attr 中的 [P, logitP] -> 一个 [0,1] 门控系数"""
    def __init__(self, in_dim, use_p=True, use_logit=True):
        super().__init__()
        self.use_p, self.use_logit = use_p, use_logit
        dim = (1 if use_p else 0) + (1 if use_logit else 0)
        if dim == 0:
            self.g = None
        else:
            self.g = nn.Sequential(
                nn.Linear(dim, 16), nn.ReLU(),
                nn.Linear(16, 1)
            )
        self.beta = nn.Parameter(torch.tensor(1.0))  # 温度

    def forward(self, edge_attr):
        if self.g is None or edge_attr is None:
            return None  # 不做门控
        cols = []
        if self.use_p: cols.append(edge_attr[:, 0:1])
        if self.use_logit: cols.append(edge_attr[:, 1:2] if edge_attr.size(1) > 1 else torch.log(edge_attr[:,0:1] + 1e-12)-torch.log(1-edge_attr[:,0:1]+1e-12))
        z = torch.cat(cols, dim=1)
        s = self.g(z) * self.beta
        gate = torch.sigmoid(s)           # (0,1)
        return gate                       # [E,1]

class GINEWithP(nn.Module):
    def __init__(self, xdim, edim, hidden=128, layers=4, out_dim=1, task='cls'):
        super().__init__()
        self.task = task
        self.emb_x = nn.Linear(xdim, hidden)
        self.emb_e = nn.Linear(edim, hidden)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden)))
            for _ in range(layers)
        ])
        self.pgate = PGate(in_dim=edim, use_p=True, use_logit=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, out_dim))

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = F.relu(self.emb_x(x))
        e = F.relu(self.emb_e(ea)) if ea is not None else None
        gate = self.pgate(ea)             # [E,1] or None

        for conv in self.convs:
            # GINEConv 的 edge_attr 会进入 MLP；我们在卷积后做“消息缩放”
            h_old = h
            h = conv(h, ei, e)            # [N, H]
            if gate is not None:
                # 把 gate 作为“边消息的缩放”注入：用 scatter_add 手工注入
                # 这里用一个简化：再跑一遍消息仅用于缩放（也可以自定义 MessagePassing）
                m = conv.nn(h_old[ei[1]] + e)          # GINE 内部消息（近似）
                m = gate * m                           # 缩放
                # 聚合
                N, H = h.size(0), h.size(1)
                agg = torch.zeros_like(h)
                agg.index_add_(0, ei[0], m)
                h = F.relu(h_old + agg)                # 残差
            else:
                h = F.relu(h)

        g = global_mean_pool(h, batch)
        y = self.head(g)
        if self.task == 'cls':
            return y  # logits
        else:
            return y  # 回归值
```

> 上面为了**易用**，用“**二次消息近似缩放**”把 `gate(P)` 注入（工程上你也可以派生 `MessagePassing`，将 `gate` 直接乘在 `message()` 上，等价更优雅）。

---

# 训练循环（with/without $P$ 可切换）

```python
def run_training(model, loaders, task='cls', epochs=80, lr=1e-3, wd=1e-5, device='cuda'):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_epoch = None, -1
    for ep in range(1, epochs+1):
        # --- train
        model.train()
        for batch in loaders['train']:
            batch = batch.to(device)
            pred = model(batch)
            if task == 'cls':
                loss = F.binary_cross_entropy_with_logits(pred.view(-1), batch.y.float())
            else:
                loss = F.mse_loss(pred.view(-1), batch.y.float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        # --- eval
        model.eval()
        # 省略：compute AUC/RMSE on val/test
        # 保存 best
    return model
```

---

# 实验流程建议（一步到位）

1. **老师 $Q^\*$ 版本（上界）**

   * 用 ETKDG 生成 $Y^\*$ → $D^\*$ → $Q^\*$（UMAP 低维核）
   * V0\~V3 跑一遍，记录指标；**预计 V1/V2 明显优于 V0**，V3 次之（无化学键信息）。

2. **学生 $\hat P$ 版本（真实可用）**

   * 用你训练好的 **P 预测器**（route-A 或 route-B）对训练/验证/测试集分子生成 $\hat P$（**冻结**）。
   * 同样跑 V1/V2/V3，对比 V0；**预计 $\hat P$ 也能带来显著增益**，但略低于 $Q^\*$。

3. **公平性与避免泄漏**

   * 所有 $P$ 都在**训练/验证/测试切分内分别计算**（学生 $\hat P$ 用已训练好的模型前向 **且不看测试标签**）。
   * 记录 wall-clock（with vs without $P$），展示**效率代价**很小（只是在构图时多了个 top-k/特征维度）。

4. **统计显著性**

   * 5-fold scaffold split 重复 3 次（不同种子）；对每个任务报 **均值±标准差**；
   * 采用 **bootstrap**（1k 次）估 AUC/RMSE 的置信区间，并给出 V0 vs V1/V2 的 **p-value**。

5. **可视化**

   * 画若干分子：叠加 $P$ 的 top-k 非键边（线宽/透明度 ∝ $P_{ij}$），**直观显示“空间近邻”**；
   * 画 **Attention/门控** 热力图，说明模型确实在用 $P$。

---

# 期望结果（如何写在论文里）

* 表 1（分类）/表 2（回归）：**V0 < V1 < V2**（显著），**V3** 介于 V0/V1 之间。
* $Q^\*$ 版本的提升最大（上界），$\hat P$ 版本次之但仍显著优于 V0。
* P-gate 的 ablation：去掉 gate 或只用 $P$ 而不用 logit$P$，效果略降。
* 结论：**空间概率图 $P$ 对下游预测有稳定增益**，证明“**空间表示**”的必要性；你的**生成方法提供了可泛化的 $P$**，即使不直接给 3D，也能提高性能。

