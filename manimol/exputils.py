import os
import re
import sys
import time
import json
import torch
import pickle
import random
import getpass
import logging
import argparse
import subprocess
import numpy as np
from datetime import timedelta, date, datetime
from rdkit import Chem
import torch
import importlib
import re
from typing import List, Any

import importlib
import re
import torch

import numpy as np
import warnings
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch_geometric.data import Data
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import Mol,GetPeriodicTable
from rdkit.Chem.Draw import rdMolDraw2D as MD2
from rdkit.Chem.rdmolops import RemoveHs

logger = logging.getLogger(__name__)



def merge_args_from_paths(args, attr_name="args_paras"):  # 修改: 增加 attr_name 参数
    """
    从 args.<attr_name> 指定的路径读取参数文件并合并
    attr_name: 在 args 中存放路径的参数名
    """
    if not hasattr(args, attr_name):
        return args

    ap = getattr(args, attr_name)
    if not ap:
        return args

    paths = []
    if isinstance(ap, (list, tuple)):
        items = list(ap)
    else:
        items = [ap]

    for p in items:
        p = Path(p)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            for sub in sorted(p.iterdir()):
                if sub.is_file():
                    paths.append(sub)
        else:
            logger.warning(f"路径不存在: {p}")

    for path in paths:
        logger.info(f"加载参数文件: {path}")
        params = _load_single_path(path)
        for k, v in params.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                logger.warning(f"参数 {k} 不在原始 args 中，已跳过")

    return args
def get_best_rmsd(probe, ref):
    probe = RemoveHs(probe)
    ref = RemoveHs(ref)
    rmsd = MA.GetBestRMS(probe, ref)
    return rmsd

def _conf_to_numpy_positions(conf, n_atoms):
    """把 RDKit Conformer -> numpy (n_atoms,3)"""
    coords = np.zeros((n_atoms, 3), dtype=float)
    for i in range(n_atoms):
        p = conf.GetAtomPosition(i)
        coords[i, 0] = float(p.x)
        coords[i, 1] = float(p.y)
        coords[i, 2] = float(p.z)
    return coords

def aggregate_conformers_by_smiles_from_loader(model, loader, device,
                                               max_pred_per_smiles=None,
                                               max_ref_per_smiles=None,
                                               include_pred=True,
                                               include_ref=True,
                                               verbose=True):
    """
    直接从 DataLoader（你的 loader）里遍历：
      - 使用 model(data) 得到 pred_pos（节点级坐标）
      - 从 data.rdmol 中提取该样本自带的真实 conformer（通常每个 sample 的 rdmol 含一个真实构象）
    将相同 smiles 的样本聚合为一组：
      - 模板（template mol）取第一个出现的样本的 rdmol（会用 Chem.Mol() 复制为模板并 RemoveAllConformers）
      - 所有真实构象（来自不同样本的 rdmol）合并为 pos_ref 列表
      - 所有预测构象（来自 model 预测的 pred_pos）合并为 pos_gen 列表（并把这些 pred 加到 deepcopy 的 mol 上，**但我们不会把 deepcopy 返回而是返回 pos_gen 数组**）
    返回:
      data_list: list of torch_geometric.data.Data，每个 Data 拥有 keys:
          'rdmol'  -> template mol (no conformers)
          'smiles' -> smiles string
          'pos_ref'-> numpy array (num_ref, n_atoms, 3)
          'pos_gen'-> numpy array (num_gen, n_atoms, 3)
    注意:
      - 不会修改 loader 里的原始 data.rdmol（所有写入都是在 Chem.Mol(copy) 上进行）
      - 需要保证 pred_pos / data.pos 的 node order 与 data.rdmol 的 atom order 对应
    """
    model.eval()
    # 存放中间聚合数据（使用 lists）
    aggr = {}  # smiles -> dict with keys: 'template_mol', 'ref_positions'(list), 'gen_positions'(list)
    with torch.no_grad():
        for batch in loader:
            # 把 batch 放到 device（与你 Runner.train/test 的风格一样）
            batch = batch.to(device)
            # 前向得到预测位置（你的 model 返回 pred_pos, ..., pred_pos shape 每个 batch 是 [total_nodes_in_batch, 3]）
            pred_pos_tensor, _, _, _ = model(batch)
            pred_pos_np = pred_pos_tensor.detach().cpu().numpy()
            gt_pos_np = batch.pos.detach().cpu().numpy()   # ground truth node positions
            batch_idx = batch.batch.detach().cpu().numpy() # 节点 -> 图索引映射

            # rdmol & smiles 在 Data 中通常是 Python object 列表
            # 稍作兼容性处理
            try:
                rdmols = list(batch.rdmol)
                smiles_list = list(batch.smiles)
            except Exception:
                # 处理某些包装形式：尽量构建 list
                n_graphs = int(batch.num_graphs) if hasattr(batch, 'num_graphs') else max(batch_idx) + 1
                rdmols = [batch.rdmol[i] for i in range(n_graphs)]
                smiles_list = [batch.smiles[i] for i in range(n_graphs)]

            n_graphs = len(rdmols)
            for i in range(n_graphs):
                smi = smiles_list[i]
                template = rdmols[i]
                if template is None:
                    if verbose:
                        warnings.warn(f"[aggregate] sample idx {i} has None rdmol, skip.")
                    continue

                # node mask for this graph
                node_mask = (batch_idx == i)
                coords_pred = pred_pos_np[node_mask]  # (n_atoms, 3)
                coords_ref_node = gt_pos_np[node_mask]  # (n_atoms, 3) -- single ref from this sample

                # initialize aggr entry if not exist
                if smi not in aggr:
                    # copy template mol for topology but remove all conformers to be a clean template
                    template_copy = Chem.Mol(template)
                    template_copy.RemoveAllConformers()
                    aggr[smi] = {
                        'template_mol': template_copy,
                        'ref_positions': [],   # list of numpy (n_atoms,3)
                        'gen_positions': []    # list of numpy (n_atoms,3)
                    }

                # check atom counts consistent with template
                n_atoms_template = aggr[smi]['template_mol'].GetNumAtoms()
                # if template_copy has 0 atoms (rare), fallback to original template's atom count
                if n_atoms_template == 0:
                    n_atoms_template = template.GetNumAtoms()

                # add reference conformer(s) from this sample's rdmol (rdmol may have one or more conformers)
                if include_ref:
                    # extract conformers from the original rdmol object (not the cleaned template)
                    orig_confs = template.GetConformers()
                    for conf in orig_confs:
                        conf_coords = _conf_to_numpy_positions(conf, template.GetNumAtoms())
                        if conf_coords.shape[0] != n_atoms_template:
                            # 不匹配则警告并跳过此 conformer
                            warnings.warn(f"[aggregate] ref atom count mismatch for smiles {smi}: template {n_atoms_template}, conf {conf_coords.shape[0]}. Skipping this ref conformer.")
                            continue
                        # 可选截断
                        if (max_ref_per_smiles is None) or (len(aggr[smi]['ref_positions']) < max_ref_per_smiles):
                            aggr[smi]['ref_positions'].append(conf_coords)

                # add the ground-truth node positions (data.pos) if present and if not redundant
                # Note: often template.GetConformers() already contains the true conformer; adding both may duplicate.
                # We still append data.pos (gt) to ensure we collected the model's ground-truth used in batch.
                if include_ref:
                    if coords_ref_node.shape[0] != n_atoms_template:
                        warnings.warn(f"[aggregate] gt node count mismatch for smiles {smi}: template {n_atoms_template}, gt {coords_ref_node.shape[0]}. Skipping this gt.")
                    else:
                        if (max_ref_per_smiles is None) or (len(aggr[smi]['ref_positions']) < max_ref_per_smiles):
                            aggr[smi]['ref_positions'].append(coords_ref_node.copy())

                # add predicted conformer built from pred_pos
                if include_pred:
                    if coords_pred.shape[0] != n_atoms_template:
                        warnings.warn(f"[aggregate] pred atom count mismatch for smiles {smi}: template {n_atoms_template}, pred {coords_pred.shape[0]}. Skipping this pred.")
                    else:
                        if (max_pred_per_smiles is None) or (len(aggr[smi]['gen_positions']) < max_pred_per_smiles):
                            aggr[smi]['gen_positions'].append(coords_pred.copy())

    # Finished scanning loader - now convert aggr -> Data list
    data_list = []
    for smi, v in aggr.items():
        ref_list = v['ref_positions']
        gen_list = v['gen_positions']
        template_mol = v['template_mol']

        # 去重/合并策略可选：这里保留原顺序和所有收集到的 conformer
        if len(ref_list) == 0 and len(gen_list) == 0:
            # skip empty
            continue

        # convert to numpy arrays with expected shapes
        if len(ref_list) > 0:
            pos_ref_arr = np.stack(ref_list, axis=0)  # (num_ref, n_atoms, 3)
        else:
            pos_ref_arr = np.zeros((0, template_mol.GetNumAtoms(), 3), dtype=float)

        if len(gen_list) > 0:
            pos_gen_arr = np.stack(gen_list, axis=0)  # (num_gen, n_atoms, 3)
        else:
            pos_gen_arr = np.zeros((0, template_mol.GetNumAtoms(), 3), dtype=float)

        # 构造 Data，注意这里我们直接用 numpy arrays（get_rmsd_confusion_matrix 中会 reshape）
        d = Data()
        d['rdmol'] = template_mol
        d['smiles'] = smi
        d['pos_ref'] = pos_ref_arr
        d['pos_gen'] = pos_gen_arr

        data_list.append(d)

    if verbose:
        print(f"[aggregate] aggregated {len(data_list)} unique smiles into Data items.")
    return data_list

class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger

def kabsch_alignment(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=0)
    centroid_Q = torch.mean(Q, dim=0)

    # Optimal translation
    t = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
        Vt[:, -1] *= -1.0

    # Optimal rotation
    R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    P_aligned = torch.matmul(p, R.transpose(0, 1)) + centroid_Q
    # RMSD
    rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(0, 1)) - q)) / P.shape[0])

    return P_aligned, R, t, rmsd

def mae_per_atom(y_pred, y_gt):
    # y_pred, y_gt: [N, 3]
    diff = y_pred - y_gt
    dist = torch.norm(diff, dim=1)  # 每个原子的欧氏距离
    return dist.mean()

def initialize_exp(params):
    """
    Initialize the experiment:
    - dump parameters
    - create a logger
    """
    # dump parameters
    exp_folder = get_dump_path(params)
    json.dump(vars(params), open(os.path.join(exp_folder, 'params.pkl'), 'w'), indent=4)

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(exp_folder, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))

    logger.info("The experiment will be stored in %s\n" % exp_folder)
    logger.info("Running command: %s" % command)
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0
    assert not params.dump_path in ('', None), \
        'Please choose your favorite destination for dump.'
    dump_path = params.dump_path

    # create the sweep path if it does not exist
    when = date.today().strftime('%m%d-')
    sweep_path = os.path.join(dump_path, when + params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an random ID for the job if it is not given in the parameters.
    if params.exp_id == '':
        # exp_id = time.strftime('%H-%M-%S')
        exp_id = datetime.now().strftime('%H-%M-%S.%f')[:-3]
        exp_id += ''.join(random.sample('abcdefghijklmnopqrstuvwxyz', 3))
        # chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        # while True:
        #     exp_id = ''.join(random.choice(chars) for _ in range(10))
        #     if not os.path.isdir(os.path.join(sweep_path, exp_id)):
        #         break
        params.exp_id = exp_id

    # create the dump folder / update parameters
    exp_folder = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(exp_folder):
        subprocess.Popen("mkdir -p %s" % exp_folder, shell=True).wait()
    return exp_folder


def describe_model(model, path, name='model'):
    file_path = os.path.join(path, f'{name}.describe')
    with open(file_path, 'w') as fout:
        print(model, file=fout)


def set_seed(seed):
    """
    Freeze every seed for reproducibility.
    torch.cuda.manual_seed_all is useful when using random generation on GPUs.
    e.g. torch.cuda.FloatTensor(100).uniform_()
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(model, save_dir, epoch=None, model_name='model'):
    model_to_save = model.module if hasattr(model, "module") else model
    if epoch is None:
        save_path = os.path.join(save_dir, f'{model_name}.pkl')
    else:
        save_path = os.path.join(save_dir, f'{model_name}-{epoch}.pkl')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model_to_save.state_dict(), save_path)


def load_model(path, map_location):
    return torch.load(path, map_location=map_location)

def save_checkpoint(model, optimizer, epoch, best_test_score, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_score': best_test_score ,
    }, filepath)

def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['best_test_score']

def visualize_mol(mol, size=(300, 300), surface=False, opacity=0.5):
    """Draw molecule in 3D
    
    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    # assert style in ('line', 'stick', 'sphere', 'carton')
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({'stick':{}, 'sphere':{'radius':0.35}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer


