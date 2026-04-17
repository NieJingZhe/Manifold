import os
import logging
from tqdm import tqdm
from munch import Munch, munchify
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
import matplotlib.pyplot as plt
from args_parse import args_parser
from exputils import get_best_rmsd, mae_per_atom, initialize_exp, set_seed, get_dump_path, describe_model, save_checkpoint, load_checkpoint, visualize_mol, kabsch_alignment, merge_args_from_paths
from dataset import QM9Dataset
from models import GNNEncoder#,EGNNEncoder
from sklearn.manifold import TSNE
logger = logging.getLogger()
import pickle
import copy
import json
from datetime import datetime
import time
import optuna
#from align3D_score import score_alignment

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class EarlyStopping:
    def __init__(self, patience=10, lower_better=True):
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf') if lower_better else -float('inf')
        self.best_model = None
        self.early_stop = False
        self.lower_better = lower_better

    def step(self, score, model):
        improved = score < self.best_score if self.lower_better else score > self.best_score
        if improved:
            self.best_score = score
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class Runner:
    def __init__(self, args, writer, logger_path):
        self.last_conformer_save_epoch = 0
        self.args = args
        self.device = torch.device(f'cuda')
        dataset = QM9Dataset(name=args.dataset, root=args.data_root)
        self.train_set = dataset[dataset.train_index]
        self.valid_set = dataset[dataset.valid_index]
        self.test_set = dataset[dataset.test_index]
        self.lower_better = 1
        self.train_loader = DataLoader(self.train_set, batch_size=args.bs, shuffle=True, drop_last=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=args.bs, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=args.bs, shuffle=False)
        cfg = Munch()
        cfg.model = Munch()
        cfg.model.model_level = 'graph'
        self.early_stopper = EarlyStopping(patience=args.patience, lower_better=True)
        self.model = GNNEncoder(args=args, config=cfg).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.opt, T_max=args.epoch, eta_min=args.eta_min)
        self.total_step = 0
        self.writer = writer
        describe_model(self.model, path=logger_path)
        self.logger_path = logger_path
        self.cfg = cfg

    def run(self, optuna_trial=None):
        """
        运行训练流程。接收可选的 optuna_trial 用于上报与剪枝。
        返回 (best_valid_score, best_epoch) 元组（正常结束时）。
        剪枝时抛出 optuna.exceptions.TrialPruned。
        其他异常会继续抛出。
        """
        best_epoch = None
        if self.lower_better == 1:
            best_valid_score, best_test_score = float('inf'), float('inf')
        else:
            best_valid_score, best_test_score = -1, -1

        try:
            if self.args.checkpoint_path is not None:
                checkpoint = torch.load(self.args.checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                best_test_score = checkpoint.get('best_test_score', best_test_score)
                print("use previous checkpoint", "epoch is", epoch, "best_test_score is", best_test_score)

            for e in range(self.args.epoch):
                #train
                self.train_step(e)

                # valid
                valid_score = self.test_step(self.valid_loader)
                try:
                    valid_val = float(valid_score)
                except:
                    try:
                        valid_val = float(valid_score.item())
                    except:
                        valid_val = float('inf')

                metric_name = f"valid_{self.args.metric.lower()}"
                self.writer.add_scalar(metric_name, valid_val, e)
                logger.info(f"E={e}, Metrics:{self.args.metric},    valid={valid_val:.5f},test-score={best_test_score:.5f}")
                logger.info(f"lr={self.opt.param_groups[0]['lr']:.6f}")

                # early stopping
                if self.args.early_stop==True:
                    self.early_stopper.step(valid_val, self.model)
                    if self.early_stopper.early_stop:
                        logger.info("Early stopping triggered.")
                        self.model.load_state_dict(self.early_stopper.best_model)
                        try:
                            mol_gt = copy.deepcopy(mol_pred)
                            mol_gt.RemoveAllConformers()
                            conf_gt = Chem.Conformer(mol_gt.GetNumAtoms())
                            for atom_idx, (x, y, z) in enumerate(pos_gt_data.tolist()):
                                conf_gt.SetAtomPosition(atom_idx, Point3D(x, y, z))
                            mol_gt.AddConformer(conf_gt, assignId=True)

                            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
                            output_dir = f"rdmol/{now_str}_{smiles_target}"
                            os.makedirs(output_dir, exist_ok=True)

                            with open(os.path.join(output_dir, "mol_view.pkl"), "wb") as f:
                                pickle.dump(mol_pred, f)
                            with open(os.path.join(output_dir, "mol_view_gt.pkl"), "wb") as f:
                                pickle.dump(mol_gt, f)
                            Chem.MolToPDBFile(mol_pred, os.path.join(output_dir, "mol_view.pdb"))
                            Chem.MolToPDBFile(mol_gt, os.path.join(output_dir, "mol_view_gt.pdb"))
                        except Exception:
                            pass
                        break

                # 更新最佳并保存 checkpoint
                if valid_val < best_valid_score:
                    test_score = self.test_step(self.test_loader)
                    try:
                        test_val = float(test_score)
                    except:
                        try:
                            test_val = float(test_score.item())
                        except:
                            test_val = float('inf')
                    best_valid_score = valid_val
                    best_test_score = test_val
                    best_epoch = e  # 记录最佳发生的 epoch

                    logger.info(f"best_valid_score = {best_valid_score:.6f} (epoch {best_epoch})")
                    logger.info(f"Metrics:{self.args.metric},  UPDATE test-score={best_test_score:.5f}")
                    save_checkpoint(self.model, self.opt, e, best_test_score, os.path.join(self.logger_path, 'best.pth'))

                if e % 10 == 0:
                    save_checkpoint(self.model, self.opt, e, best_test_score, os.path.join(self.logger_path, f'epoch{e}.pth'))

                # 上报给 optuna 并检查剪枝
                if optuna_trial is not None:
                    try:
                        optuna_trial.report(best_valid_score if best_valid_score != float('inf') else None, e)
                        if optuna_trial.should_prune():
                            metrics = {
                                "status": "pruned",
                                "best_valid_score": best_valid_score if best_valid_score != float('inf') else None,
                                "best_test_score": best_test_score if best_test_score != float('inf') else None,
                                "best_epoch": best_epoch,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            _atomic_write_json(os.path.join(self.logger_path, "metrics.json"), metrics)
                            raise optuna.exceptions.TrialPruned()
                    except optuna.exceptions.TrialPruned:
                        raise
                    except Exception:
                        pass

            logger.info(f"test-score={best_test_score:.5f},learning rate={self.opt.param_groups[0]['lr']:.6f}")

            # 正常结束，写 final metrics 并返回 (best_valid_score, best_epoch)
            final_metrics = {
                "status": "finished",
                "best_valid_score": best_valid_score if best_valid_score != float('inf') else None,
                "best_test_score": best_test_score if best_test_score != float('inf') else None,
                "best_epoch": best_epoch,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            _atomic_write_json(os.path.join(self.logger_path, "metrics.json"), final_metrics)

            return best_valid_score, best_epoch

        except optuna.exceptions.TrialPruned:
            logger.info("Trial was pruned by Optuna.")
            # 在被剪枝时已经写过 metrics.json（上面逻辑有写），向上抛出以便 Optuna 处理
            raise
        except Exception as e:
            logger.exception("Exception during training")
            err_metrics = {
                "status": "failed",
                "error": repr(e),
                "best_valid_score": best_valid_score if best_valid_score != float('inf') else None,
                "best_test_score": best_test_score if best_test_score != float('inf') else None,
                "best_epoch": best_epoch,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            try:
                _atomic_write_json(os.path.join(self.logger_path, "metrics.json"), err_metrics)
            except Exception:
                pass
            raise

    def train_step(self, epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"E [{epoch}]")
        found = False
        mol_pred = None
        pos_gt_data = None
        smiles_target = self.args.smiles
        total_loss_epoch = 0
        num_batches = 0

        for data in pbar:
            data = data.to(self.device)
            
            pred_pos, graph_feat, pos_loss, mani_loss,mol_list = self.model(data,epoch=epoch)

            if self.args.get_image:
                for i, smiles in enumerate(data.smiles):
                    if smiles == smiles_target and not found:

                        mol_pred = data.rdmol[i]

                        conf = Chem.Conformer(mol_pred.GetNumAtoms())
                        node_mask = (data.batch == i)
                        for atom_idx, (x, y, z) in enumerate(pred_pos[node_mask].tolist()):
                            conf.SetAtomPosition(atom_idx, Point3D(x, y, z))
                        mol_pred.AddConformer(conf, assignId=True)
                        pos_gt_data = data.pos[node_mask].cpu().numpy()
                        found = True
                        break


            loss = pos_loss * self.args.pos_w + mani_loss * self.args.mani_w
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.opt.step()
            self.scheduler.step()
            pbar.set_postfix_str(f"loss={loss.item():.4f}, mani_loss={mani_loss:.4f}")
            self.writer.add_scalar('loss', loss.item(), self.total_step)
            self.writer.add_scalar('pos-loss', pos_loss.item(), self.total_step)
            self.writer.add_scalar('mani-loss', mani_loss.item(), self.total_step)
            self.writer.add_scalar('lr', self.opt.param_groups[0]['lr'], epoch)
            total_loss_epoch += loss.item()
            num_batches += 1
            self.total_step += 1
        avg_loss_epoch = total_loss_epoch / num_batches
        self.writer.add_scalar('loss_per_epoch', avg_loss_epoch, epoch)
        # when the last epoch
        if epoch == self.args.epoch-1  and found:
            # generate mol_gt
            mol_gt = copy.deepcopy(mol_pred)
            mol_gt.RemoveAllConformers()
            conf_gt = Chem.Conformer(mol_gt.GetNumAtoms())
            for atom_idx, (x, y, z) in enumerate(pos_gt_data.tolist()):
                conf_gt.SetAtomPosition(atom_idx, Point3D(x, y, z))
            mol_gt.AddConformer(conf_gt, assignId=True)

            # make output dir

            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
            output_dir = f"rdmol/{now_str}_{smiles_target}"
            os.makedirs(output_dir, exist_ok=True)

            # save mol and .pdb files
            with open(os.path.join(output_dir, "mol_view.pkl"), "wb") as f:
                pickle.dump(mol_pred, f)
            with open(os.path.join(output_dir, "mol_view_gt.pkl"), "wb") as f:
                pickle.dump(mol_gt, f)
            Chem.MolToPDBFile(mol_pred, os.path.join(output_dir, "mol_view.pdb"))
            Chem.MolToPDBFile(mol_gt, os.path.join(output_dir, "mol_view_gt.pdb"))

            # save args
            with open(os.path.join(output_dir, "train_args.json"), "w") as f:
                json.dump(vars(self.args), f, indent=4)
            print(" Saved mol view and args to", output_dir)


        if self.args.get_image and found and (epoch % 10 == 0) and (epoch > self.last_conformer_save_epoch):
            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
            output_dir_mid = f"rdmol/epoch{epoch}_{now_str}_{smiles_target}"
            os.makedirs(output_dir_mid, exist_ok=True)

            with open(os.path.join(output_dir_mid, "mol_view_epoch.pkl"), "wb") as f:
                pickle.dump(mol_pred, f)
            Chem.MolToPDBFile(mol_pred, os.path.join(output_dir_mid, "mol_view_epoch.pdb"))

            self.last_conformer_save_epoch = epoch
            print(f"💾 Saved mid-training conformer at epoch {epoch} to {output_dir_mid}")

    @torch.no_grad()
    def test_step(self, loader):
        self.model.eval()
        y_pred, y_gt = [], []
        for data in loader:
            data = data.to(self.device)

            pred_pos, _, _, _,mol_list = self.model(data,epoch=1)
            
            '''y_pred.append(pred_pos)         # shape: [num_atoms, 3]
            y_gt.append(data.pos)           # ground truth positions

        # 把 list of tensors → tensor
        y_pred = torch.cat(y_pred, dim=0)  # shape: [total_atoms, 3]
        y_gt = torch.cat(y_gt, dim=0)      # same shape'''
            y_pred.extend(mol_list)
            y_gt.extend(data.rdmol)
        if self.args.metric == 'MAE':
            y_pred,_,_,score = kabsch_alignment(y_pred, y_gt)
            score = mae_per_atom(y_pred, y_gt)
        elif self.args.metric == 'RMSD':
            score_sum = 0
            for i in range(len(y_pred)):
                score = get_best_rmsd(y_pred[i], y_gt[i])
                score_sum = score_sum + score
            score = score_sum / len(y_pred)
            #y_pred,_,_,score = kabsch_alignment(y_pred, y_gt)
            #score = torch.sqrt(F.mse_loss(y_pred, y_gt))
        elif self.args.metric == 'score_alignment':
            score=score_alignment(y_pred,y_gt)
        else:
            raise ValueError(f"metric is {self.args.metric}. We need MAE or RMSD or score_alignment.")
        return score

def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=4)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp, path)

def main(args=None, optuna_trial=None):
    if args is None:
        args = args_parser()
    try:
        args = merge_args_from_paths(args,attr_name="paras_path")
    except Exception:
        logger = logging.getLogger()
        logger.exception("merge_args_from_paths 出错，继续使用原始 args。")
    torch.cuda.set_device(int(args.gpu))

    # if bayesian optimization is enabled, return without running the main code
    if hasattr(args, 'run_bayesian_optimization') and args.run_bayesian_optimization:
        return

    logger = initialize_exp(args)
    set_seed(args.random_seed)
    logger_path = get_dump_path(args)
    writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    runner = Runner(args, writer, logger_path)
    try:
        result = runner.run(optuna_trial=optuna_trial)
    finally:
        try:
            writer.close()
        except Exception:
            pass

    # runner.run 返回 (best_valid_score, best_epoch) 或抛异常
    return result

if __name__ == '__main__':
    main()
