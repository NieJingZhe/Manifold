"""Trainer module: contains Trainer class and training loop.
Depends on project modules: dataset.QM9Dataset, models.GNNEncoder, exputils functions, utils helpers.
"""
from datetime import datetime
import os
import logging
from typing import Any, Optional, Tuple, Dict

from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import QM9Dataset
from models import GNNEncoder
from exputils import describe_model, save_checkpoint, kabsch_alignment, mae_per_atom, get_best_rmsd
from utils.device import get_device
from utils.checkpoint import CheckpointManager
from utils.helpers import to_runconfig_from_args, compute_score_by_metric, get_arg, safe_float, save_molecule_views, _atomic_write_json
from torch_geometric.nn.conv.pna_conv import PNAConv
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 10, lower_better: bool = True):
        self.patience = patience
        self.lower_better = lower_better
        self.counter = 0
        self.best_score = float('inf') if lower_better else -float('-inf')
        self.best_model_state = None
        self.early_stop = False

    def step(self, score: float, model: torch.nn.Module) -> None:
        improved = score < self.best_score if self.lower_better else score > self.best_score
        if improved:
            self.best_score = score
            # copy state dict safely to CPU tensors
            self.best_model_state = {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Trainer:
    def __init__(self, args: Any, writer: SummaryWriter, logger_path: str):
        self.args = args
        self.cfg = to_runconfig_from_args(args)
        self.device = get_device(self.cfg.gpu)
        self.writer = writer
        self.logger_path = str(logger_path)
        self.checkpoint_mgr = CheckpointManager(self.logger_path)
        self.total_step = 0

        # dataset / loaders
        dataset = QM9Dataset(name=self.cfg.dataset, root=self.cfg.data_root, args=args)
        self.train_set = dataset[dataset.train_index]
        self.valid_set = dataset[dataset.valid_index]
        self.test_set = dataset[dataset.test_index]

        self.train_loader = DataLoader(self.train_set, batch_size=self.cfg.bs, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.cfg.bs, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.cfg.bs, shuffle=False)

        # model / optimizer / scheduler
        cfg_dict = {'model': {'model_level': 'graph'}}
        deg_hist = PNAConv.get_degree_histogram(self.train_loader) #如果想要有可重复性需要shuffle改为false，这样还可以将其保存为.pt直接读取  # dtype=torch.long
        self.GNNEncoder = GNNEncoder(args=args, config=cfg_dict,deg = deg_hist).to(self.device)
        self.opt = torch.optim.Adam(self.GNNEncoder.parameters(), lr=self.cfg.lr)
        self.scheduler = CosineAnnealingLR(self.opt, T_max = int(self.cfg.epoch * len(self.train_loader)), eta_min=self.cfg.eta_min)
 
        # early stopping
        self.early_stopper = EarlyStopping(patience=self.cfg.patience, lower_better=True)

        try:
            describe_model(self.GNNEncoder, path=self.logger_path)
        except Exception:
            logger.exception('describe_model 出错（可忽略）')

        self.last_found: Dict[str, Any] = dict(found=False, mol_pred=None, pos_gt_data=None, smiles_target=None)
        logger.info(f"Trainer initialized on device {self.device}")

    def run(self, optuna_trial: Optional[Any] = None) -> Tuple[Optional[float], Optional[int]]:
        best_valid_score = float('inf')
        best_test_score = float('inf')
        best_epoch: Optional[int] = None

        if getattr(self.args, 'checkpoint_path', None):
            try:
                ck = torch.load(self.args.checkpoint_path, map_location=self.device)
                self.GNNEncoder.load_state_dict(ck['model_state_dict'])
                self.opt.load_state_dict(ck['optimizer_state_dict'])
                logger.info(f"Loaded checkpoint from {self.args.checkpoint_path} (epoch {ck.get('epoch')})")
            except Exception:
                logger.exception('加载 checkpoint 失败，继续从头开始')
        # 训练主循环   
        try:
            for epoch in range(self.cfg.epoch):
                logger.info(f"Starting epoch {epoch}")
                self.train_epoch(epoch)

                valid_score = safe_float(self.test_step(self.valid_loader, val=True,epoch=epoch))
                self.writer.add_scalar(f"valid_{self.cfg.metric.lower()}", valid_score, epoch)
                logger.info(f"Epoch {epoch} valid {self.cfg.metric} = {valid_score:.6f}, best_test_so_far={best_test_score:.6f}")

                if self.cfg.early_stop:
                    self.early_stopper.step(valid_score, self.GNNEncoder)
                    if self.early_stopper.early_stop:
                        logger.info('Early stopping triggered; restoring best model state')
                        if self.early_stopper.best_model_state is not None:
                            self.GNNEncoder.load_state_dict(self.early_stopper.best_model_state)
                        break

                if valid_score < best_valid_score:
                    test_score = safe_float(self.test_step(self.test_loader, val=False,epoch=epoch))
                    best_valid_score = valid_score
                    best_test_score = test_score
                    best_epoch = epoch
                    logger.info(f"New best valid {best_valid_score:.6f} (epoch {best_epoch}), test={best_test_score:.6f}")
                    try:
                        self.checkpoint_mgr.save_best(self.GNNEncoder, self.opt, epoch, best_test_score)
                    except Exception:
                        logger.exception('保存 best checkpoint 失败')
            #训练主循环结束，这里主要是一些保存和超参数优化
                if epoch % 10 == 0:
                    try:
                        self.checkpoint_mgr.save_epoch(self.GNNEncoder, self.opt, epoch, best_test_score)
                    except Exception:
                        logger.exception('保存 epoch checkpoint 失败')

                if optuna_trial is not None:
                    try:
                        optuna_trial.report(None if best_valid_score == float('inf') else best_valid_score, epoch)
                        if optuna_trial.should_prune():
                            metrics = {
                                'status': 'pruned',
                                'best_valid_score': None if best_valid_score == float('inf') else best_valid_score,
                                'best_test_score': None if best_test_score == float('inf') else best_test_score,
                                'best_epoch': best_epoch,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            _atomic_write_json(os.path.join(self.logger_path, 'metrics.json'), metrics)
                            raise optuna.exceptions.TrialPruned()
                    except Exception:
                        logger.exception('Optuna 报告失败（忽略）')

            final_metrics = {
                'status': 'finished',
                'best_valid_score': None if best_valid_score == float('inf') else best_valid_score,
                'best_test_score': None if best_test_score == float('inf') else best_test_score,
                'best_epoch': best_epoch,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            try:
                _atomic_write_json(os.path.join(self.logger_path, 'metrics.json'), final_metrics)
            except Exception:
                logger.exception('写入 final metrics 失败')

            return (None if best_valid_score == float('inf') else best_valid_score, best_epoch)

        except Exception as e:
            logger.exception('训练过程中发生异常')
            err_metrics = {
                'status': 'failed',
                'error': repr(e),
                'best_valid_score': None if best_valid_score == float('inf') else best_valid_score,
                'best_test_score': None if best_test_score == float('inf') else best_test_score,
                'best_epoch': best_epoch,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            try:
                _atomic_write_json(os.path.join(self.logger_path, 'metrics.json'), err_metrics)
            except Exception:
                pass
            raise

    def train_epoch(self, epoch: int) -> None:
        self.GNNEncoder.train()
        pbar = tqdm(self.train_loader, desc=f"Train E[{epoch}]")
        total_loss = 0.0
        total_graphs = 0

        for batch in pbar:
            loss_val, n_graphs = self.train_batch(batch, epoch)   # now returns (batch_mean_loss, n_graphs)
            total_loss += loss_val * n_graphs
            total_graphs += n_graphs
            pbar.set_postfix_str(f"loss={loss_val:.4f}")

        avg = total_loss / max(1, total_graphs)
        self.writer.add_scalar('loss_per_epoch', avg, epoch)
        logger.info(f"Epoch {epoch} average loss {avg:.6f}")


    def train_batch(self, batch: Any, epoch: int) -> Tuple[float, int]:
        batch = batch.to(self.device)
        pred_pos, _, pos_loss, mani_loss, mol_list = self.GNNEncoder(batch, epoch=epoch)

        loss = mani_loss * self.args.mani_w + pos_loss * self.args.pos_w
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.GNNEncoder.parameters(), 5)
        self.opt.step()

       
        self.writer.add_scalar('loss', float(loss.detach().cpu()), self.total_step)
        self.writer.add_scalar('pos-loss', float(pos_loss.detach().cpu()), self.total_step)
        self.writer.add_scalar('mani-loss', float(mani_loss.detach().cpu()), self.total_step)
        self.writer.add_scalar('lr', self.opt.param_groups[0]['lr'], self.total_step)
        self.scheduler.step()
        self.total_step += 1

        # 本 batch 的图数（兼容 PyG Batch）
        n_graphs = getattr(batch, 'num_graphs', None)
        if n_graphs is None:
            n_graphs = len(mol_list)

        return float(loss.detach().cpu()), int(n_graphs)
        




    @torch.no_grad()
    def test_step(self, loader: DataLoader, val: bool, epoch: Optional[int] = None) -> float:
        """返回 dataset-level 的 CE（per-graph average mani_loss）"""
        self.GNNEncoder.eval()
        total_mani = 0.0
        total_graphs = 0
        y_pred = []
        y_gt = []

        for batch in loader:
            batch = batch.to(self.device)
            _, _, _, maniloss, mol_list = self.GNNEncoder(batch, epoch=epoch)

            y_pred.extend(mol_list)
            y_gt.extend(batch.rdmol)

            n_graphs = getattr(batch, 'num_graphs', None)
            if n_graphs is None:
                n_graphs = len(mol_list)
                print("n_graphs 计算方式采用 len(mol_list)")        
            total_mani += maniloss * n_graphs
            total_graphs += n_graphs

        if total_graphs == 0:
            raise RuntimeError("Empty dataset in test()")

        avg_mani_loss = total_mani / total_graphs
        score = float(avg_mani_loss)

        tag = 'val-score' if val else 'test-score'
        if epoch is not None:
            self.writer.add_scalar(tag, score, epoch)
        else:
            self.writer.add_scalar(tag, score, self.total_step)
        return score
