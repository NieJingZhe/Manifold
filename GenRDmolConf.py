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
from exputils import initialize_exp, set_seed, get_dump_path, describe_model ,save_checkpoint,load_checkpoint,visualize_mol
from models import GNNEncoder,resolve_internal_clashes_batch  
from dataset import QM9Dataset
from sklearn.manifold import TSNE
logger = logging.getLogger()
import pickle
import copy
import json
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import AllChem

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def generate_optimal_conformer(data, forcefield, smiles_target) -> torch.Tensor:
    coords_list = []
    device = data.pos.device

    for i, mol in enumerate(data.rdmol_embedded):
        current_smiles = data.smiles[i]

        # 只对目标smiles做保存操作
        save_this = (current_smiles == smiles_target)

        # 复制分子
        mol_copy = Chem.Mol(mol)

        # 力场优化
        if forcefield == 'uff':
            status = AllChem.UFFOptimizeMolecule(mol_copy, maxIters=1000)
        elif forcefield == 'mmff':
            status = AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=1000)
        else:
            raise ValueError(f"Unknown forcefield: {forcefield}")

        if status != 0:
            print(f"⚠️ {forcefield} 优化未完全收敛 (status={status})")

        # 提取坐标
        conf = mol_copy.GetConformer(0)
        atom_coords = []
        for atom_idx in range(mol_copy.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            atom_coords.append([pos.x, pos.y, pos.z])

        coords_tensor = torch.tensor(atom_coords, dtype=torch.float32, device=device)
        coords_list.append(coords_tensor)

        # 只保存一次对应的 pdb 文件
        if save_this:
            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]
            output_dir = f"rdmol/{forcefield}_{now_str}_{smiles_target}"
            os.makedirs(output_dir, exist_ok=True)
            pdb_path = os.path.join(output_dir, f"mol_view_{forcefield}.pdb")
            Chem.MolToPDBFile(mol_copy, pdb_path)
            print(f"Successfully saved conformer for {smiles_target} to {pdb_path}")

    all_coords = torch.cat(coords_list, dim=0)
    return all_coords



class Runner:
    def __init__(self, args, writer, logger_path):
        self.forcefield = args.forcefield
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
        self.model = GNNEncoder(args=args, config=cfg).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = CosineAnnealingLR(self.opt, T_max=args.epoch, eta_min=args.eta_min)
        self.total_step = 0
        self.writer = writer
        describe_model(self.model, path=logger_path)
        self.logger_path = logger_path
        self.cfg = cfg

    def run(self):
        
            self.train_step()
            valid_score = self.test_step(self.valid_loader)
            test_score = self.test_step(self.test_loader)
            logger.info( f" valid={valid_score:.5f},test-score={test_score:.5f}")
   

    def train_step(self, ):

        pbar = tqdm(self.train_loader)
        for data in pbar:
            data = data.to(self.device)
            pred_pos=generate_optimal_conformer(data,self.args.forcefield,smiles_target=self.args.smiles)

            
    
    @torch.no_grad()
    def test_step(self, loader):
        self.model.eval()
        y_pred, y_gt = [], []

        for data in loader:
            data = data.to(self.device)
            pred_pos=generate_optimal_conformer(data,self.args.forcefield,self.args.smiles)
            y_pred.append(pred_pos)         # shape: [num_atoms, 3]
            y_gt.append(data.pos)           # ground truth positions

        # 把 list of tensors → tensor
        y_pred = torch.cat(y_pred, dim=0)  # shape: [total_atoms, 3]
        y_gt = torch.cat(y_gt, dim=0)      # same shape
        if self.args.metric == 'MAE':
            score = F.l1_loss(y_pred, y_gt)
        elif self.args.metric == 'RMSD':
            score = torch.sqrt(F.mse_loss(y_pred, y_gt))
        else:
            raise ValueError(f"metric is {self.args.metric}. We need MAE or RMSD.")
        return score

def main(args=None):
    if args is None:
        args = args_parser()
    
    torch.cuda.set_device(int(args.gpu))
    
    # if bayesian optimization is enabled, return without running the main code
    if hasattr(args, 'run_bayesian_optimization') and args.run_bayesian_optimization:
        return
    
    logger = initialize_exp(args)
    set_seed(args.random_seed)
    logger_path = get_dump_path(args)
    writer = SummaryWriter(log_dir=os.path.join(logger_path, 'tensorboard'))

    runner = Runner(args, writer, logger_path)
    runner.run()
    writer.close()

if __name__ == '__main__':
    main()
