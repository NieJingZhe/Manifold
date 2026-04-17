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
from utils.optuna import report_optuna_and_maybe_prune, write_final_metrics, handle_training_exception
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
        self.model = GNNEncoder(args=args, config=cfg_dict).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.scheduler = CosineAnnealingLR(self.opt, T_max=self.cfg.epoch, eta_min=self.cfg.eta_min)

        # early stopping
        self.early_stopper = EarlyStopping(patience=self.cfg.patience, lower_better=True)

        try:
            describe_model(self.model, path=self.logger_path)
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
                self.model.load_state_dict(ck['model_state_dict'])
                self.opt.load_state_dict(ck['optimizer_state_dict'])
                logger.info(f"Loaded checkpoint from {self.args.checkpoint_path} (epoch {ck.get('epoch')})")
            except Exception:
                logger.exception('加载 checkpoint 失败，继续从头开始')

        try:
            for epoch in range(self.cfg.epoch):
                logger.info(f"Starting epoch {epoch}")
                self.train_epoch(epoch)

                valid_score = safe_float(self.test(self.valid_loader, val=True))
                self.writer.add_scalar(f"valid_{self.cfg.metric.lower()}", valid_score, epoch)
                logger.info(f"Epoch {epoch} valid {self.cfg.metric} = {valid_score:.6f}, best_test_so_far={best_test_score:.6f}")

                # early stop
                if self.cfg.early_stop:
                    self.early_stopper.step(valid_score, self.model)
                    if self.early_stopper.early_stop:
                        logger.info('Early stopping triggered; restoring best model state')
                        if self.early_stopper.best_model_state is not None:
                            self.model.load_state_dict(self.early_stopper.best_model_state)
                        if self.last_found.get('found', False):
                            now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
                            out_dir = f"rdmol/{now_str}_{self.last_found.get('smiles_target')}"
                            try:
                                save_molecule_views(out_dir, self.last_found.get('mol_pred'),
                                                    pos_gt_data=self.last_found.get('pos_gt_data'),
                                                    smiles_target=self.last_found.get('smiles_target'),
                                                    args_to_save=self.args)
                            except Exception:
                                logger.exception('保存 mid conformer 出错')
                        break

                # update best & save
                if valid_score < best_valid_score:
                    test_score = safe_float(self.test(self.test_loader, val=False))
                    best_valid_score = valid_score
                    best_test_score = test_score
                    best_epoch = epoch
                    logger.info(f"New best valid {best_valid_score:.6f} (epoch {best_epoch}), test={best_test_score:.6f}")
                    try:
                        self.checkpoint_mgr.save_best(self.model, self.opt, epoch, best_test_score)
                    except Exception:
                        logger.exception('保存 best checkpoint 失败')
                #
                if epoch % 10 == 0:
                    try:
                        self.checkpoint_mgr.save_epoch(self.model, self.opt, epoch, best_test_score)
                    except Exception:
                        logger.exception('保存 epoch checkpoint 失败')

                # --- NEW: delegate optuna report + prune to utils.metrics ---
                report_optuna_and_maybe_prune(optuna_trial, best_valid_score, epoch, self.logger_path)

            # --- NEW: write final metrics ---
            write_final_metrics(self.logger_path, best_valid_score, best_test_score, best_epoch)
            return (None if best_valid_score == float('inf') else best_valid_score, best_epoch)

        except Exception as e:
            # 记录异常栈
            logger.exception('训练过程中发生异常')
            # 统一调用集中化的异常处理器（会写 failed metrics）
            try:
                handle_training_exception(self.logger_path, e, best_valid_score, best_test_score, best_epoch)
            except Exception:
                logger.exception('调用 handle_training_exception 失败')
            # 重新抛出，保持原来行为（上层可以感知 / 测试框架可捕获）
            raise


    def train_epoch(self, epoch: int) -> None:
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"Train E[{epoch}]")
        epoch_loss = 0.0
        n_batches = 0

        for batch in pbar:
            loss_val = self.train_batch(batch, epoch)
            epoch_loss += loss_val
            n_batches += 1
            pbar.set_postfix_str(f"loss={loss_val:.4f}")
        avg = epoch_loss / max(1, n_batches)
        self.writer.add_scalar('loss_per_epoch', avg, epoch)
        logger.info(f"Epoch {epoch} average loss {avg:.6f}")

    def train_batch(self, batch: Any, epoch: int) -> float:
        batch = batch.to(self.device)
        pred_pos, _, pos_loss, mani_loss, mol_list = self.model(batch, epoch=epoch)

        loss = pos_loss * get_arg(self.args, 'pos_w', 1.0) + mani_loss * get_arg(self.args, 'mani_w', 1.0)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.opt.step()
        self.scheduler.step()

        if get_arg(self.args, 'get_image', False):
            smiles_target = get_arg(self.args, 'smiles', None)
            if smiles_target is not None and not self.last_found.get('found', False):
                for i, smiles in enumerate(batch.smiles):
                    if smiles == smiles_target and not self.last_found.get('found', False):
                        try:
                            mol_pred = batch.rdmol[i]
                            conf = Chem.Conformer(mol_pred.GetNumAtoms())
                            node_mask = (batch.batch == i)
                            for atom_idx, (x, y, z) in enumerate(pred_pos[node_mask].tolist()):
                                conf.SetAtomPosition(atom_idx, Point3D(x, y, z))
                            mol_pred.AddConformer(conf, assignId=True)
                            pos_gt_data = batch.pos[node_mask].cpu().numpy()
                            self.last_found.update(found=True, mol_pred=mol_pred, pos_gt_data=pos_gt_data, smiles_target=smiles_target)
                        except Exception:
                            logger.exception('保存目标 conformer 出错')
                        break

        self.writer.add_scalar('loss', float(loss.detach().cpu()), self.total_step)
        self.writer.add_scalar('pos-loss', float(pos_loss.detach().cpu()), self.total_step)
        self.writer.add_scalar('mani-loss', float(mani_loss.detach().cpu()), self.total_step)
        self.writer.add_scalar('lr', self.opt.param_groups[0]['lr'], self.total_step)

        self.total_step += 1
        return float(loss.detach().cpu())

    @torch.no_grad()
    def test(self, loader: DataLoader, val: bool) -> float:
        self.model.eval()
        y_pred = []
        y_gt = []
        last_maniloss = None
        for batch in loader:
            batch = batch.to(self.device)
            _, _, _, maniloss, mol_list = self.model(batch, epoch=1)
            y_pred.extend(mol_list)
            y_gt.extend(batch.rdmol)
            last_maniloss = maniloss

        if self.cfg.metric.upper() == 'CE':
            if last_maniloss is None:
                raise RuntimeError('No mani loss computed for CE metric.')
            score = float(last_maniloss)
        else:
            score = compute_score_by_metric(y_pred, y_gt, self.cfg.metric)

        tag = 'val-score' if val else 'test-score'
        self.writer.add_scalar(tag, score, self.total_step)
        return float(score)
