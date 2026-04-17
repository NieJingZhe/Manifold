# train.py
import os
import time
import warnings
import argparse

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm
from torch_geometric.data import Batch
from torch.utils.data import Subset
from tqdm import tqdm
import wandb
from torch_geometric.nn import GINConv, GPSConv

from torch_geometric.transforms import AddLaplacianEigenvectorPE
from models.model import Net

from dataset.drugdataset import QM9Dataset,DrugsDataset

from manifold import prob_low_dim

# ---------- collate ----------
def collate_with_idx(batch):
    return Batch.from_data_list(batch)

# ---------- Trainer ----------
class Trainer:
    def __init__(self, args, config,ctx):
        wandb.init(dir="/mnt/liyong/wandb_runs",
        project=ctx.merged_config["project"],
        entity=ctx.merged_config["entity"],
        config=ctx.merged_config,
        name=f'{ctx.start_time}_{ctx.name}',
    )
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.cfg = config

        # 加载数据集
        if args.dataset == 'QM9':
            self.dataset = QM9Dataset(root=self.cfg.data_root, name=self.cfg.dataset)
        elif args.dataset =='Drugs':
            self.dataset  =DrugsDataset(root=self.cfg.data_root)
        self.a = self.dataset.a
        self.b = self.dataset.b
        orig_train_idx = list(self.dataset.train_index)
        orig_valid_idx = list(self.dataset.valid_index)
        orig_test_idx = list(self.dataset.test_index)

        seed = getattr(self.args, "seed", 42)
        g = torch.Generator().manual_seed(seed)

        new_train_n = min(8, len(orig_test_idx))
        new_test_n = min(1000, len(orig_train_idx))

        perm_test = torch.randperm(len(orig_test_idx), generator=g).tolist()
        perm_train = torch.randperm(len(orig_train_idx), generator=g).tolist()

        new_train_idx = [orig_test_idx[i] for i in perm_test[:new_train_n]]
        new_valid_idx = orig_valid_idx.copy()
        new_test_idx = [orig_train_idx[i] for i in perm_train[:new_test_n]]

        if self.args.exchange:
            self.train_set = Subset(self.dataset, new_train_idx)
            self.valid_set = Subset(self.dataset, new_train_idx)
            self.test_set = Subset(self.dataset, new_train_idx)
        else:
            self.train_set = Subset(self.dataset, orig_train_idx)
            self.valid_set = Subset(self.dataset, orig_valid_idx)
            self.test_set = Subset(self.dataset, orig_test_idx)

        self.train_loader = DataLoader(self.train_set, batch_size=self.args.bs, shuffle=False, collate_fn=collate_with_idx)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.args.bs, shuffle=False, collate_fn=collate_with_idx)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.bs, shuffle=False, collate_fn=collate_with_idx)

        self.save_dir = getattr(self.cfg, "save_dir", "./saved_model")
        os.makedirs(self.save_dir, exist_ok=True)

        # deg histogram
        deg = torch.zeros(10, dtype=torch.long)
        for data in self.train_set:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        input_dim = 9 + args.lap_k 
        self.model = Net(self.args, deg=deg, data=None, input_dim=input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-6)

        if args.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                               patience=max(1, int(self.args.epoch / 10)), min_lr=self.args.lr / 5.0)
            self._scheduler_mode = "plateau"
        elif args.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, args.epoch), eta_min=args.eta_min)
            self._scheduler_mode = "step"
        elif args.scheduler == "cosine_restart":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=args.T0, T_mult=1, eta_min=args.eta_min)
            self._scheduler_mode = "step"
        else:
            self.scheduler = None
            self._scheduler_mode = None

        self.best = (float("100 "), float("100"), 0)

    def compute_loss(self, Phat, batch):
        """
        返回：
          batch_loss (标量 tensor)：最终用于 backward 的 loss（包含 sim term 与 outlier penalty）
          per_graph_losses_list (list of tensors)：每个图的基础 loss（未加 sim）
          batch_sim (scalar tensor)
        """
        # Qlist 与 Plist 的构造保持原样
        Q_list = [self.dataset.q_meta_list[int(idx)]['Qgreatest'].to(Phat.device) for idx in batch.idx]
        P_list = [Phat[batch.batch == i][:, batch.batch == i] for i in range(len(Q_list))]

        per_graph_losses = []
        # 逐图计算基础 loss（按选项）
        for P, Q in zip(P_list, Q_list):
            # 为了稳定起见，确保 P 和 Q 形状一致
            if P.numel() == 0 or Q.numel() == 0:
                per_graph_losses.append(torch.tensor(0.0, device=Phat.device))
                continue

            if self.args.loss_type == "bce":
                loss_val = F.binary_cross_entropy(P, Q)
            elif self.args.loss_type == "bce_mse":
                loss_bce = F.binary_cross_entropy(P, Q)
                loss_mse = F.mse_loss(P, Q)
                loss_val = loss_bce + self.args.alpha * loss_mse
            elif self.args.loss_type == "bce_huber":
                loss_bce = F.binary_cross_entropy(P, Q)
                huber = torch.nn.SmoothL1Loss(reduction='mean')  # SmoothL1 大致等价 Huber
                loss_huber = huber(P, Q)
                loss_val = loss_bce + self.args.alpha * loss_huber
            else:
                loss_val = F.binary_cross_entropy(P, Q)

            per_graph_losses.append(loss_val)

        # 转成 tensor，便于 trim / quantile 等操作
        if len(per_graph_losses) == 0:
            return torch.tensor(0.0, device=Phat.device), [], torch.tensor(0.0, device=Phat.device)

        losses_tensor = torch.stack(per_graph_losses)  # (G,)

        # optional: trimmed-mean（从上侧裁去一部分最差样本）
        trim_ratio = float(self.args.trim_ratio) if hasattr(self.args, "trim_ratio") else 0.0
        if trim_ratio > 0.0 and trim_ratio < 0.5:
            k = int(len(losses_tensor) * (1.0 - trim_ratio))
            if k < 1:
                k = 1
            sorted_losses, _ = torch.sort(losses_tensor)
            base_loss = sorted_losses[:k].mean()
        else:
            base_loss = losses_tensor.mean()

        # outlier penalty（基于损失分位点）：对大于 quantile 的 loss 额外二次惩罚
        outlier_w = float(getattr(self.args, "outlier_w", 0.0))
        outlier_penalty = torch.tensor(0.0, device=Phat.device)
        if outlier_w > 0.0:
            q = float(getattr(self.args, "outlier_q", 0.95))
            if 0.0 < q < 1.0:
                try:
                    threshold = torch.quantile(losses_tensor, q).item()
                except Exception:
                    threshold = float(losses_tensor.max().item())
                excess = torch.relu(losses_tensor - threshold)
                outlier_penalty = (excess ** 2).mean() * outlier_w

        # similarity term (cosine similarity between flattened P and Q，和原实现一致)
        sims = []
        for P, Q in zip(P_list, Q_list):
            p_flat = P.reshape(-1)
            q_flat = Q.reshape(-1)
            sim = F.cosine_similarity(p_flat, q_flat, dim=0)
            sims.append(sim)
        batch_sim = torch.stack(sims).mean() if len(sims) > 0 else torch.tensor(0.0, device=Phat.device)

        # 最终 loss：基础 loss + sim penalty + outlier_penalty
        final_loss = base_loss + self.args.sim_w * (1.0 - batch_sim) + outlier_penalty

        return final_loss, per_graph_losses, batch_sim


    def train_epoch(self, epoch):
        self.model.train()
        total_loss_sum, total_graphs = 0.0, 0
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch:02d} batches", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            Phat, gram, pair = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,a=self.a,b=self.b)

            batch_loss, per_graph_losses, batch_sim = self.compute_loss(Phat, batch)
            batch_loss.backward()
            self.optimizer.step()

            total_loss_sum += sum([l.item() for l in per_graph_losses])
            total_graphs += len(per_graph_losses)

        # scheduler step：根据类型分别调用
        if hasattr(self, "_scheduler_mode") and self._scheduler_mode == "plateau":
            # 这里需要 val loss，因此在外面训练循环会调用 scheduler.step(val_loss)
            pass
        else:
            # step per-epoch
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    # 某些 scheduler 需要参数，但前面我们已区分
                    pass

        return total_loss_sum / total_graphs if total_graphs else 0.0


    @torch.no_grad()
    def eval_split(self, loader):
        self.model.eval()
        total_loss_sum, total_graphs = 0.0, 0
        total_sim_sum = 0.0

        for batch in loader:
            batch = batch.to(self.device)
            Phat, gram, pair = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,a = self.a,b = self.b)
            _, per_graph_losses, batch_sim = self.compute_loss(Phat, batch)
            total_loss_sum += sum([l.item() for l in per_graph_losses])
            total_sim_sum += batch_sim.item() * len(per_graph_losses)
            total_graphs += len(per_graph_losses)

        avg_loss = total_loss_sum / total_graphs if total_graphs else 0.0
        avg_sim = total_sim_sum / total_graphs if total_graphs else 0.0
        return avg_loss, avg_sim

    def forward(self):
        start_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        tqdm.write(f"Training start time: {start_time}")

        # 保存上一次 best model 的 test 指标
        self.best_test_loss = None
        self.best_test_score = None

        for epoch in range(1, self.args.epoch + 1):
            tqdm.write(f"\n=== Epoch {epoch:02d} start ===")
            # 训练前 eval train_loader 的 start_loss 和 score
            train_loss_start, train_score = self.eval_split(self.train_loader)
            val_loss, val_score = self.eval_split(self.valid_loader)

            # 训练当前 epoch
            train_loss = self.train_epoch(epoch)

            # scheduler: 如果是 plateau，用验证 loss 做 step；
            # 否则对于基于 epoch 的 scheduler（cosine / cosine_restart），train_epoch 已经在每 epoch 内 step 过
            if getattr(self, "_scheduler_mode", None) == "plateau":
                if self.scheduler is not None:
                    # 用验证 loss 做 step（ReduceLROnPlateau 需要 val_loss）
                    self.scheduler.step(val_loss)
            else:
                # 对于 CosineAnnealingLR / CosineAnnealingWarmRestarts：已在 train_epoch 中 self.scheduler.step()
                pass

            # 判断是否更新 best model（基于 validation loss）
            improved = val_loss < self.best[0]
            if improved:
                test_loss, test_score = self.eval_split(self.test_loader)
                self.best_test_loss = test_loss
                self.best_test_score = test_score

                self.best = (val_loss, test_loss, epoch,val_score)
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

            # wandb log，test 使用上一次 best
            log_dict = {
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": val_loss,
                "train/start_loss": train_loss_start,
                "train/cosine_similarity": train_score,
                "valid/cosine_similarity": val_score,
                "test/loss": self.best_test_loss,
                "test/cosine_similarity": self.best_test_score,
                "lr": self.optimizer.param_groups[0]['lr']
            }
            wandb.log(log_dict, step=epoch)

        wandb.finish()
        return self.best[3]


# ---------- argparse & Config（与 stage1.py 风格一致，方便 optuna 调用） ----------

def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Drugs", help="指定数据集", choices=['QM9','Drugs'])
    parser.add_argument("--gpu", type=int, default=4, help="指定使用的GPU编号")
    parser.add_argument("--conv", type=str, default="GPS", choices=["GPS", "PNA", "GIN"], help="选择图卷积类型（训练用，仅做记录）")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--epoch", type=int, default=300, help="训练轮数")
    parser.add_argument("--bs", type=int, default=256, help="batchsize")
    parser.add_argument("--num_layers", type=int, default=4, help="图神经网络层数")
    parser.add_argument("--exchange", action='store_true', help="进行debug实验用的参数")
    parser.add_argument("--root", type=str, default="data", help="root dir of dataset")
    parser.add_argument("--name", type=str, default="qm9", help="dataset name")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--emb_dim", type=int, default=80, help="节点嵌入维度")
    parser.add_argument("--Zi", type=int, default=32, help="节点最终嵌入维度")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")
    parser.add_argument("--n_heads", type=int, default=4, help="GPS heads 数")
    parser.add_argument("--sim_w", type=float, default=0.3, help="相似度损失权重")
    parser.add_argument("--use_lap_pe", action="store_true", help="是否在 node feature 中拼接 Laplacian Eigenvector PE")
    parser.add_argument("--lap_k", type=int, default=3, help="LapPE 的特征向量数量 k（默认 3）")
    parser.add_argument("--project", type=str, default="my_qm9_project", help="wandb project name")
    parser.add_argument("--entity", type=str, default=None, help="wandb entity")

    # 新增：scheduler 与 loss 配置
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["plateau", "cosine", "cosine_restart"],
                        help="学习率调度器（plateau: ReduceLROnPlateau; cosine: CosineAnnealingLR; cosine_restart: CosineAnnealingWarmRestarts）")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="余弦退火的最小学习率（eta_min）")
    parser.add_argument("--T0", type=int, default=50, help="CosineAnnealingWarmRestarts 的 T_0（重启周期）")

    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce","bce_mse","bce_huber"],
                        help="训练时使用的损失组合（bce, bce_mse, bce_huber）")
    parser.add_argument("--alpha", type=float, default=1.0, help="当使用 bce_mse 时，MSE 部分的权重")
    parser.add_argument("--huber_delta", type=float, default=1.0, help="SmoothL1（Huber）中的 delta（近似）")
    parser.add_argument("--trim_ratio", type=float, default=0.0, help="裁剪比例，用于按图 loss 的 trimmed-mean（0-0.5）")
    parser.add_argument("--outlier_w", type=float, default=0.0, help="异常值惩罚项权重（0 表示不使用）")
    parser.add_argument("--outlier_q", type=float, default=0.95, help="用于判定异常 loss 的分位点（0-1）")
    return parser


class Config:
    dataset = "qm9"
    data_root = "/home/liyong/data/mani_data"
    bs = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.005
    weight_decay = 1e-6
    epochs = 1000
    project = "my_qm9_project"
    entity = None
    save_dir = "./saved_model"


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

    parser = make_argparser()
    args = parser.parse_args()


    # 合并 Config（仅用于 wandb 的 config 记录）
    base_config = {k: v for k, v in Config.__dict__.items() if not k.startswith("__") and not callable(v)}
    merged_config = {**base_config, **vars(args)}
    name = merged_config.get("name")
    start_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    from types import SimpleNamespace

    ctx = SimpleNamespace(
        merged_config=merged_config,
        start_time=start_time,
        name=name,
    )
    trainer = Trainer(args=args, config=Config,ctx=ctx)
    trainer.forward()
    
