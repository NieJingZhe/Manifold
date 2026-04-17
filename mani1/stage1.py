# stage1.py
import os
import time
import json
import argparse
import warnings

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, global_mean_pool, GINConv, PNAConv

from dataset.drugdataset import QM9Dataset
from manifold import prob_low_dim

import wandb
from tqdm import tqdm
from torch.utils.data import Subset
import optuna

# -------- utils ----------
def collate_with_idx(batch):
    return Batch.from_data_list(batch)


# -------- Model ----------
class Net(torch.nn.Module):
    def __init__(self, args, deg, edge_attr_dim=0):
        super(Net, self).__init__()
        self.args = args
        emb_in = getattr(args, "node_emb_dim", 9)
        emb_dim = args.emb_dim
        Zi = args.Zi
        self.K = args.K
        self.node_emb = Linear(emb_in, emb_dim)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(self.args.num_layers):
            if self.args.conv == 'PNA':
                conv = PNAConv(in_channels=emb_dim, out_channels=emb_dim,
                               aggregators=aggregators,
                               scalers=scalers, deg=deg,
                               edge_dim=3, post_layers=1)
            elif self.args.conv == 'GIN':
                conv = GINConv(Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim)))
            else:
                raise ValueError(f"Unknown conv type: {self.args.conv}")
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(emb_dim))

        self.mlp = Sequential(Linear(emb_dim, 2 * emb_dim), ReLU(),
                              Linear(2 * emb_dim, 2 * emb_dim), ReLU(),
                              Linear(2 * emb_dim, Zi))

        self.heads = ModuleList([
            Sequential(Linear(Zi, Zi), ReLU(), Linear(Zi, Zi))
            for _ in range(self.K)
        ])

    def forward(self, x, edge_index, edge_attr, batch):

        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()
        x = self.node_emb(x)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            if self.args.conv == 'PNA':
                h = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            else:
                h = F.relu(batch_norm(conv(x, edge_index)))
            x = h + x
            x = F.dropout(x, self.args.dropout)

        x = self.mlp(x)  # (N_total_nodes, Zi)


        batch_ids = batch  # alias
        num_graphs = int(batch_ids.max().item()) + 1
        P_preds_list = []
        for i in range(num_graphs):
            node_mask = (batch_ids == i)
            H_nodes = x[node_mask]  # (n_i, Zi)
            n_i = H_nodes.size(0)
            if n_i == 0:

                P_preds_list.append(torch.zeros((self.K, 0, 0), device=x.device))
                continue

            # 每个 head 生成 Zk -> prob_low_dim -> (n_i,n_i)
            P_stack = []
            for k in range(self.K):
                Zk = self.heads[k](H_nodes)  # (n_i, Zi)
                qk = prob_low_dim(Zk, self.args.umap_a, self.args.umap_b)  # (n_i, n_i), in [0,1]
                P_stack.append(qk)
            # stack -> (K, n_i, n_i)
            P_stack = torch.stack(P_stack, dim=0)
            P_preds_list.append(P_stack)

        return P_preds_list


# -------- Trainer ----------
class Trainer:
    def __init__(self, args, config):
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.cfg = config

        self.dataset = QM9Dataset(root=self.cfg.data_root, name=self.cfg.dataset)
        orig_train_idx = list(self.dataset.train_index)
        orig_valid_idx = list(self.dataset.valid_index)
        orig_test_idx = list(self.dataset.test_index)

        seed = getattr(self.args, "seed", 42)
        g = torch.Generator().manual_seed(seed)

        # 新划分（保持你原来的逻辑）
        new_train_n = min(8, len(orig_test_idx))
        new_test_n = min(1000, len(orig_train_idx))

        perm_test = torch.randperm(len(orig_test_idx), generator=g).tolist()
        perm_train = torch.randperm(len(orig_train_idx), generator=g).tolist()

        new_train_idx = [orig_test_idx[i] for i in perm_test[:new_train_n]]
        new_valid_idx = orig_valid_idx.copy()
        new_test_idx = [orig_train_idx[i] for i in perm_train[:new_test_n]]

        if self.args.debug:
            self.train_set = Subset(self.dataset, new_train_idx)
            self.valid_set = Subset(self.dataset, new_train_idx)
            self.test_set = Subset(self.dataset, new_train_idx)
        else:
            self.train_set = Subset(self.dataset, orig_train_idx)
            self.valid_set = Subset(self.dataset, orig_valid_idx)
            self.test_set = Subset(self.dataset, orig_test_idx)

        self.train_loader = DataLoader(self.train_set, batch_size=self.args.bs, shuffle=True, collate_fn=collate_with_idx)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.args.bs, shuffle=False, collate_fn=collate_with_idx)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.bs, shuffle=False, collate_fn=collate_with_idx)

        exp_id = getattr(self.args, "exp_id", None)
        if exp_id is None or str(exp_id).strip() == "":
            self.save_dir = os.path.join("./saved_model", str(self.args.Qtype))
        else:
            if os.path.sep in exp_id or exp_id.startswith(".") or exp_id.startswith("/"):
                self.save_dir = os.path.abspath(exp_id)
            else:
                self.save_dir = os.path.join("./saved_model", exp_id)
        os.makedirs(self.save_dir, exist_ok=True)

        # deg histogram
        deg = torch.zeros(10, dtype=torch.long)
        for data in self.train_set:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        # model 放到 device
        self.model = Net(self.args, deg).to(self.device)

        # training hyperparams
        self.K = getattr(self.args, "K", 5)
        self.lambda_div = getattr(self.args, "lambda_div", 0.0)
        self.sigma_div = getattr(self.args, "sigma_div", 0.1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-6)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                           patience=max(1, int(self.args.epoch / 10)), min_lr=self.args.lr / 5.0)
        self.best = (0, 0, 0, None)

    def compute_loss(self, P_preds_list, batch, criterion, loss_type="BCE"):

        device = self.device
        per_graph_losses = []
        sims = []

        # iterate graphs in the current Batch; batch.idx 存储 dataset idx（你数据集里已存在）
        for i, ds_idx in enumerate(batch.idx):
            ds_idx = int(ds_idx)
            q_meta = self.dataset.q_meta_list[ds_idx]
            q_list = q_meta['q_list']  # list of (n_i, n_i) tensors on cpu
            bw = q_meta['boltzmannweights']
            bw = bw.to(device)
            sorted_idx = torch.argsort(bw, descending=True).to(device)
            num_conf = len(q_list)
            K_match = min(self.K, num_conf)

            P_stack = P_preds_list[i].to(device)  # (K, n_i, n_i)
            n_i = P_stack.size(1)

            loss_graph = 0.0
            sims_local = []

            for k in range(K_match):
                t_idx = int(sorted_idx[k].cpu().item())
                Q_target = q_list[t_idx].to(device)
                # ensure size match (假设 q_list 的 n_i 与当前 graph 节点数一致)
                Pk = P_stack[k]

                if self.args.meanW:
                    w = 1.0 / float(num_conf)
                else:
                    w = float(bw[t_idx].cpu().item())

                # 选择 loss 计算方式
                if loss_type == "BCE":
                    c = criterion(Pk, Q_target)
                elif loss_type == "MSE":
                    c = criterion(Pk, Q_target)
                elif loss_type == "KL":
                    # KLDivLoss expects log-prob input
                    c = criterion(torch.log(Pk.clamp(min=1e-8)), Q_target)
                else:
                    c = criterion(Pk, Q_target)

                loss_graph = loss_graph + w * c

                # cosine sim for logging（flatten）
                p_flat = Pk.reshape(-1)
                q_flat = Q_target.reshape(-1)
                sims_local.append(F.cosine_similarity(p_flat, q_flat, dim=0))

            # diversity penalty 可后续实现（目前保持为0）
            loss_graph = loss_graph + self.lambda_div * 0.0

            per_graph_losses.append(loss_graph)
            if len(sims_local) > 0:
                sims.append(torch.stack(sims_local).mean())
            else:
                sims.append(torch.tensor(0.0, device=device))

        # aggregate
        batch_loss = torch.stack(per_graph_losses).mean() if len(per_graph_losses) > 0 else torch.tensor(0.0, device=device)
        batch_sim = torch.stack(sims).mean() if len(sims) > 0 else torch.tensor(0.0, device=device)
        return batch_loss, per_graph_losses, batch_sim

    def train(self, epoch):
        self.model.train()
        total_loss_sum, total_graphs = 0.0, 0

        if self.args.loss == 'MSE':
            criterion = torch.nn.MSELoss()
        elif self.args.loss == 'KL':
            criterion = torch.nn.KLDivLoss(reduction='batchmean')
        else:
            criterion = torch.nn.BCELoss()

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch:02d} batches", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            P_preds_list = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # list per graph
            batch_loss, per_graph_losses, _ = self.compute_loss(P_preds_list, batch, criterion, loss_type=self.args.loss)

            batch_loss.backward()
            self.optimizer.step()

            total_loss_sum += sum([l.item() for l in per_graph_losses])
            total_graphs += len(per_graph_losses)

        return total_loss_sum / total_graphs if total_graphs else 0.0

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        total_loss_sum, total_graphs = 0.0, 0
        total_sim_sum = 0.0

        if self.args.loss == 'MSE':
            criterion = torch.nn.MSELoss()
        elif self.args.loss == 'KL':
            criterion = torch.nn.KLDivLoss(reduction='batchmean')
        else:
            criterion = torch.nn.BCELoss()

        for batch in loader:
            batch = batch.to(self.device)
            P_preds_list = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            _, per_graph_losses, batch_sim = self.compute_loss(P_preds_list, batch, criterion, loss_type=self.args.loss)
            total_loss_sum += sum([l.item() for l in per_graph_losses])
            total_sim_sum += batch_sim.item() * len(per_graph_losses)
            total_graphs += len(per_graph_losses)

        avg_loss = total_loss_sum / total_graphs if total_graphs else 0.0
        avg_sim = total_sim_sum / total_graphs if total_graphs else 0.0
        return avg_loss, avg_sim

    def forward(self):
        trial = getattr(self.args, "optuna_trial", None)
        start_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        tqdm.write(f"Training start time: {start_time}")

        for epoch in range(1, self.args.epoch + 1):
            tqdm.write(f"\n=== Epoch {epoch:02d} start ===")
            train_loss_start, train_score = self.test(self.train_loader)
            val_loss, val_score = self.test(self.valid_loader)
            test_loss, test_score = self.test(self.test_loader)

            train_loss = self.train(epoch)
            print("train loss is ", train_loss)
            self.scheduler.step(val_loss)

            tqdm.write(f"Epoch: {epoch:02d}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, Test: {test_loss:.6f},TrainStart: {train_loss_start:.6f}")
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                "test/loss": test_loss,
                "valid/cosine_similarity": val_score,
                "test/cosine_similarity": test_score,
                "train/start_loss": train_loss_start,
                "lr": self.optimizer.param_groups[0]['lr']
            }, step=epoch)

            if trial is not None:
                trial.report(val_score, epoch)
                if trial.should_prune():
                    tqdm.write(f"Trial pruned at epoch {epoch}, val_score={val_score:.6f}")
                    raise optuna.exceptions.TrialPruned()

            if val_score > self.best[0]:
                self.best = (val_score, test_score, epoch)

                time_dir = os.path.join(self.save_dir, start_time)
                os.makedirs(time_dir, exist_ok=True)

                save_model_path = os.path.join(time_dir, "best_model.pt")
                torch.save(self.model.state_dict(), save_model_path)

                args_path = os.path.join(time_dir, "train_args.pt")
                torch.save(vars(self.args), args_path)

                artifact = wandb.Artifact("best_model", type="model", metadata={
                    "val_loss": float(val_loss), "test_loss": float(test_loss), "epoch": int(epoch)
                })
                artifact.add_file(save_model_path)
                wandb.log_artifact(artifact)

                tqdm.write("Best model updated")

        print(json.dumps({"val": self.best[0]}))
        return self.best[0]


# -------- helper entry points ----------
def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default=None, help="保存的目录的名称")
    parser.add_argument("--gpu", type=int, default=0, help="指定使用的GPU编号")
    parser.add_argument("--conv", type=str, default="PNA", choices=["PNA", "GIN"], help="选择图卷积类型")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--epoch", type=int, default=300, help="训练轮数")
    parser.add_argument("--bs", type=int, default=256, help="batchsize")
    parser.add_argument("--num_layers", type=int, default=4, help="图神经网络层数")
    parser.add_argument("--debug", action='store_true', help="进行debug实验用的参数")
    parser.add_argument("--root", type=str, default="data", help="root dir of dataset")
    parser.add_argument("--name", type=str, default="qm9", help="dataset name")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--emb_dim", type=int, default=80, help="节点嵌入维度")
    parser.add_argument("--node_emb_dim", type=int, default=9, help="输入节点特征维度")
    parser.add_argument("--Zi", type=int, default=64, help="节点最终嵌入维度")
    parser.add_argument("--loss", type=str, default="BCE", choices=["MSE", "KL", "BCE"], help="选择损失函数类型")
    parser.add_argument('--Qtype', type=str, default='Qmean', choices=['QWmean', 'Qmean', 'Qgreatest'])
    parser.add_argument('--dropout', type=float, default=0.3, help="dropout 概率")
    parser.add_argument('--K', type=int, default=5, help="输出 head 数（每个 head 预测一个 Q）")
    parser.add_argument('--lambda_div', type=float, default=0.0, help="diversity penalty 权重（可先设0）")
    parser.add_argument('--sigma_div', type=float, default=0.1, help="diversity penalty 的尺度")
    parser.add_argument('--umap_a', type=float, default=0.583, help="UMAP kernel a 参数")
    parser.add_argument('--umap_b', type=float, default=1.334, help="UMAP kernel b 参数")
    parser.add_argument('--meanW', action='store_true', help="是否对所有构象使用均匀权重")
    return parser


def trainer(args, config):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    T = Trainer(args=args, config=config)
    return T.forward()


class Config:
    dataset = "qm9"
    data_root = "/home/liyong/data/mani_data"
    bs = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.005
    weight_decay = 1e-6
    epochs = 100
    project = "my_qm9_project"
    entity = None
    exp_id = "./saved_model"


if __name__ == "__main__":
    parser = make_argparser()
    args = parser.parse_args()

    if getattr(args, "exp_id", None) is None or str(args.exp_id).strip() == "":
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        args.exp_id = f"run_{timestamp}"
        print(f"[info] no exp_id provided. auto-generated exp_id = {args.exp_id}")
    else:
        args.exp_id = str(args.exp_id).strip()
        print(f"[info] using exp_id = {args.exp_id}")

    base_config = {k: v for k, v in Config.__dict__.items() if not k.startswith("__") and not callable(v)}
    merged_config = {**base_config, **vars(args)}

    wandb.init(
        project=merged_config["project"],
        entity=merged_config["entity"],
        config=merged_config,
        name=args.exp_id,
    )

    trainer(args, Config)
    wandb.finish()
