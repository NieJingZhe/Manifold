# mol_utils.py
# 将args中的输入的smiles中对应的结构保存到rdkit的conformer（这个在求RMSE时也需要用到）中，保存为pdb格式用以可视化，如果有其他更好的方式也可以采用其他的方式

import os
import pickle
import copy
import json
import logging
from rdkit import Chem
from rdkit.Geometry import Point3D

# 初始化日志器（确保能使用logger.exception）
logger = logging.getLogger(__name__)



def save_molecule_views(output_dir, mol_pred, pos_gt_data=None, smiles_target=None, args_to_save=None, save_pdb=True):
    """
    Save mol_pred and its ground-truth conformer (if pos_gt_data provided).
    Creates:
      - mol_view.pkl
      - mol_view_gt.pkl (if pos_gt_data)
      - mol_view.pdb and mol_view_gt.pdb (if save_pdb True)
      - train_args.json (if args_to_save provided)
    # MOD 3: 将重复保存分子/生成 mol_gt 的逻辑抽出来，避免重复造轮子。
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save predicted mol
    try:
        with open(os.path.join(output_dir, "mol_view.pkl"), "wb") as f:
            pickle.dump(mol_pred, f)
    except Exception:
        logger.exception("Failed to pickle mol_pred")

    # If ground-truth positions provided, construct and save a GT conformer mol
    if pos_gt_data is not None:
        try:
            mol_gt = copy.deepcopy(mol_pred)
            mol_gt.RemoveAllConformers()
            conf_gt = Chem.Conformer(mol_gt.GetNumAtoms())
            for atom_idx, (x, y, z) in enumerate(pos_gt_data.tolist()):
                conf_gt.SetAtomPosition(atom_idx, Point3D(x, y, z))
            mol_gt.AddConformer(conf_gt, assignId=True)
            with open(os.path.join(output_dir, "mol_view_gt.pkl"), "wb") as f:
                pickle.dump(mol_gt, f)
        except Exception:
            logger.exception("Failed to create/save mol_gt")

    # Save pdb files (if possible)
    try:
        if save_pdb:
            Chem.MolToPDBFile(mol_pred, os.path.join(output_dir, "mol_view.pdb"))
            if pos_gt_data is not None:
                Chem.MolToPDBFile(mol_gt, os.path.join(output_dir, "mol_view_gt.pdb"))
    except Exception:
        logger.exception("Failed to write pdb files")

    # Save args if provided (this keeps reproducibility)
    if args_to_save is not None:
        try:
            with open(os.path.join(output_dir, "train_args.json"), "w") as f:
                json.dump(vars(args_to_save), f, indent=4)
        except Exception:
            logger.exception("Failed to save args to json")



class MolSaver:
    """
    保存分子 conformer 的小工具类：
      - build_mol_with_pred: 根据预测坐标生成带 conformer 的分子副本（不会修改原对象）
      - save_final: 保存训练结束（或 early-stop）时的 mol_pred + gt conformer + train args
      - save_epoch: 保存中间 epoch 的 mol（只保存 mol_pred 的 epoch 视图）
    """
    def __init__(self, base_dir='rdmol'):
        self.base_dir = base_dir
        # 记录最后一次保存的 epoch，防止重复保存
        self.last_conformer_save_epoch = -1
        os.makedirs(self.base_dir, exist_ok=True)

    def _now_str(self):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]

    def _make_output_dir(self, prefix, smiles):
        now_str = self._now_str()
        dirn = os.path.join(prefix, f"{now_str}_{smiles}")
        os.makedirs(dirn, exist_ok=True)
        return dirn

    def build_mol_with_pred(self, mol, pred_pos, node_mask):
        """
        返回一个 deepcopy(mol) 并把 pred_pos[node_mask] 作为一个新的 conformer 添加进去。
        pred_pos: Tensor/ndarray shape [N_atoms_total, 3]
        node_mask: boolean mask (或 LongTensor 索引) 标识属于该分子的原子行
        """
        mol_new = copy.deepcopy(mol)
        try:
            conf = Chem.Conformer(mol_new.GetNumAtoms())
            coords = pred_pos[node_mask].tolist()
            for atom_idx, (x, y, z) in enumerate(coords):
                conf.SetAtomPosition(atom_idx, Point3D(x, y, z))
            # remove existing conformers (defensive)
            try:
                mol_new.RemoveAllConformers()
            except Exception:
                pass
            mol_new.AddConformer(conf, assignId=True)
        except Exception:
            # 如果任何步骤失败，返回 deepcopy(mol)（不包含预测坐标）
            mol_new = copy.deepcopy(mol)
        return mol_new

    def _write_pickle(self, path, obj):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def _write_json_args(self, dirn, args):
        if args is None:
            return
        try:
            with open(os.path.join(dirn, "train_args.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
        except Exception:
            pass

    def save_final(self, mol_pred, pos_gt=None, smiles="unknown", args=None, save_pdb=True, save_pickle=True):
        """
        保存最终视图：mol_pred (+ mol_gt 如果 pos_gt 给出) 以及 train args。
        返回 output_dir 路径（或 None if failed）
        """
        try:
            # 构建 mol_gt（如果有）
            mol_gt = None
            if pos_gt is not None:
                try:
                    mol_gt = copy.deepcopy(mol_pred)
                    mol_gt.RemoveAllConformers()
                    conf_gt = Chem.Conformer(mol_gt.GetNumAtoms())
                    for atom_idx, (x, y, z) in enumerate(pos_gt.tolist()):
                        conf_gt.SetAtomPosition(atom_idx, Point3D(x, y, z))
                    mol_gt.AddConformer(conf_gt, assignId=True)
                except Exception:
                    mol_gt = None

            out_dir = self._make_output_dir(self.base_dir, smiles)

            if save_pickle:
                try:
                    self._write_pickle(os.path.join(out_dir, "mol_view.pkl"), mol_pred)
                    if mol_gt is not None:
                        self._write_pickle(os.path.join(out_dir, "mol_view_gt.pkl"), mol_gt)
                except Exception:
                    pass

            if save_pdb:
                try:
                    Chem.MolToPDBFile(mol_pred, os.path.join(out_dir, "mol_view.pdb"))
                    if mol_gt is not None:
                        Chem.MolToPDBFile(mol_gt, os.path.join(out_dir, "mol_view_gt.pdb"))
                except Exception:
                    pass

            # 保存 args（如果提供）
            self._write_json_args(out_dir, args)

            return out_dir
        except Exception:
            return None

    def save_epoch(self, mol_pred, epoch, smiles="unknown", save_pdb=True, save_pickle=True):
        """
        保存中间 epoch 的 mol_pred（例如 'rdmol/epoch{epoch}_{time}_{smiles}/...'）
        只在 epoch > last_conformer_save_epoch 时保存并更新 last_conformer_save_epoch。
        返回 output_dir 或 None。
        """
        try:
            if epoch <= self.last_conformer_save_epoch:
                return None
            prefix = os.path.join(self.base_dir, f"epoch{epoch}")
            out_dir = self._make_output_dir(prefix, smiles)

            if save_pickle:
                try:
                    self._write_pickle(os.path.join(out_dir, "mol_view_epoch.pkl"), mol_pred)
                except Exception:
                    pass
            if save_pdb:
                try:
                    Chem.MolToPDBFile(mol_pred, os.path.join(out_dir, "mol_view_epoch.pdb"))
                except Exception:
                    pass

            self.last_conformer_save_epoch = epoch
            return out_dir
        except Exception:
            return None


from typing import List, Optional, Union, Sequence
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

def write_pred_pos_to_conformers(
    pred_pos: torch.Tensor,
    data,
    rdmol_attr: str = "rdmol",
    batch_attr: str = "batch",
    allow_placeholder: bool = False,
    placeholder_atom: str = "C"
) -> List[Chem.Mol]:
    """
    将预测坐标写入 RDKit Mol 的 conformer 并返回分子列表，用于最终RMSD检测。

    ---- 输入 ----
    - pred_pos: torch.Tensor, shape [N_total_atoms, 3], dtype float32/float64. 可以在任意 device，
                函数内部会用 .detach().cpu() 读取数值用于写入 RDKit。
    - data: 一个对象，期望包含以下之一：
        1) data.batch: torch.LongTensor/torch.Tensor (长度 = N_total_atoms)，表示每个原子所属分子索引；
           同时 data.rdmol: 序列 (list/tuple) 包含每个分子的 rdkit.Chem.Mol。
        2) 或者仅 data.rdmol: 序列 (list/tuple)，此时函数按 rdmol 中每个分子的 atom-count
           依次把 pred_pos 切成块（前提：总原子数与 pred_pos 行数一致）。
    - rdmol_attr: data 中 rdkit mol 列表的属性名，默认 "rdmol"
    - batch_attr: data 中 batch 的属性名，默认 "batch"
    - allow_placeholder: 若 True，当 data.rdmol 缺失或某元素为 None 时，自动创建占位分子（仅用于可视化/调试）
    - placeholder_atom: 占位分子使用的原子符号（如 "C"）

    ---- 输出 ----
    - mol_list: List[rdkit.Chem.Mol]
        返回已复制并附加新 conformer 的分子列表。仅包含那些成功写入构象的分子。
        注意：原始 data.rdmol 不会被原地修改（我们对每个 orig_mol 取了 Chem.Mol(orig_mol) 的副本）。

    ---- 行为 ----
    - 完全使用 torch 来处理索引/切片（没有 numpy 依赖）。
    - 当 batch 存在时，以 batch 决定每个分子的 atom indices；否则按 data.rdmol 的 atom counts 顺序分块。
    - 若某分子的 atom_idx 为空会被跳过。
    - 若 rdmol 缺失且 allow_placeholder=False，会抛出 ValueError（包含说明）。
    """
    
    if not isinstance(pred_pos, torch.Tensor):
        raise TypeError("pred_pos 必须是 torch.Tensor，当前类型: {}".format(type(pred_pos)))

    if pred_pos.ndim != 2 or pred_pos.size(1) != 3:
        raise ValueError("pred_pos 必须形状为 [N,3]，当前形状: {}".format(tuple(pred_pos.shape)))

    # 将 pred_pos 保持为 tensor，但为读取数值时准备 cpu 副本（不转换为 numpy 全数组）
    pred_pos_cpu = pred_pos.detach().cpu()

    # 读取 batch 与 rdmol_list（均不做 numpy 转换）
    batch = getattr(data, batch_attr, None)
    rdmol_list: Optional[Sequence] = getattr(data, rdmol_attr, None)

    if batch is not None:
        if not isinstance(batch, torch.Tensor):
            # 若 batch 是其他可迭代（例如 list/np array），尝试转为 torch tensor
            batch = torch.tensor(batch, dtype=torch.long)
        else:
            batch = batch.detach().cpu().long()
        # 计算分子数
        if batch.numel() == 0:
            n_mols = 0
        else:
            n_mols = int(batch.max().item()) + 1
        batch_arr = batch
    else:
        # 没有 batch：需要 rdmol_list 来推断分子数和每个分子的原子数
        if rdmol_list is None:
            raise ValueError(f"data 中既没有属性 '{batch_attr}' 也没有属性 '{rdmol_attr}'。无法推断分子分块。")
        try:
            n_mols = len(rdmol_list)
        except Exception:
            raise ValueError(f"data.{rdmol_attr} 必须是可索引序列 (list/tuple)。 当前类型: {type(rdmol_list)}")
        batch_arr = None

    mol_list: List[Chem.Mol] = []

    # 当没有 batch 时，预先计算每个分子的 atom counts（纯 Python 调用 RDKit 接口）
    if batch_arr is None:
        sizes = []
        for i in range(n_mols):
            mol_i = rdmol_list[i] if i < len(rdmol_list) else None
            if mol_i is None:
                sizes.append(0)
            else:
                sizes.append(mol_i.GetNumAtoms())
        # 验证总数是否与 pred_pos 行数匹配（可选，不严格要求）
        total = sum(sizes)
        if total != pred_pos_cpu.size(0):
            # 不必强制报错，但提醒用户（用异常会中断流程，遵循原逻辑可选择抛出）
            raise ValueError(f"按 data.rdmol 推断的总原子数 {total} 与 pred_pos 行数 {pred_pos_cpu.size(0)} 不一致。")

        # 计算每个分子的起止索引（torch-friendly）
        starts = [0]
        for s in sizes[:-1]:
            starts.append(starts[-1] + s)

    # 遍历每个分子，取对应 atom indices 并写入 conformer
    for mi in range(n_mols):
        if batch_arr is not None:
            # 找到属于该分子的 atom 索引（torch.where -> 1D long tensor）
            atom_idx = torch.where(batch_arr == mi)[0]
        else:
            s = sizes[mi]
            if s == 0:
                atom_idx = torch.empty((0,), dtype=torch.long)
            else:
                start = starts[mi]
                atom_idx = torch.arange(start, start + s, dtype=torch.long)

        if atom_idx.numel() == 0:
            # 没有原子，跳过
            continue

        coords = pred_pos_cpu[atom_idx]  # 仍是 torch.Tensor on cpu, shape [n_atoms_i, 3]

        # 获取原始分子或创建占位（如果允许）
        orig_mol = None
        if rdmol_list is not None:
            try:
                orig_mol = rdmol_list[mi]
            except Exception:
                orig_mol = None

        if orig_mol is None:
            if not allow_placeholder:
                raise ValueError(
                    f"data.{rdmol_attr}[{mi}] 缺失或为 None。无法为分子索引 {mi} 写入构象。"
                    " 如需自动创建占位分子，请将 allow_placeholder=True。"
                )
            # 创建占位分子：只包含相应数量的原子（无键）
            mol_rw = Chem.RWMol()
            n_atoms_i = int(coords.size(0))
            for _ in range(n_atoms_i):
                mol_rw.AddAtom(Chem.Atom(placeholder_atom))
            orig_mol = mol_rw.GetMol()

        # 复制 mol，避免修改原始对象
        mol = Chem.Mol(orig_mol)

        # 新建 conformer 并写入坐标（写入到 min(n_atoms_in_mol, coords.shape[0])）
        n_write = min(mol.GetNumAtoms(), int(coords.size(0)))
        conf = Chem.Conformer(mol.GetNumAtoms())
        # coords 是 torch.Tensor; 逐个取 float，RDKit 需要 Python float
        for ai in range(n_write):
            # coords[ai] 是 tensor([x,y,z])
            x = float(coords[ai, 0].item())
            y = float(coords[ai, 1].item())
            z = float(coords[ai, 2].item())
            conf.SetAtomPosition(int(ai), Point3D(x, y, z))

        # 移除旧 conformer 并添加新 conformer
        try:
            mol.RemoveAllConformers()
        except Exception:
            pass
        mol.AddConformer(conf, assignId=True)

        mol_list.append(mol)

    return mol_list





