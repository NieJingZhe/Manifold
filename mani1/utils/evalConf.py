#evalConf
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
import os
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem import rdMolAlign as MA
def GetBestRMSD(probe, ref, prbId=None, refId=None):
    # 这里不去氢，假设外面已经去氢
    rmsd = MA.GetBestRMS(probe, ref, prbId=prbId, refId=refId)
    return rmsd

def get_rmsd_confusion_matrix(mol_gen: Chem.Mol, mol_ref: Chem.Mol, useFF=False):
    mol_gen = RemoveHs(mol_gen)
    mol_ref = RemoveHs(mol_ref)
    num_gen = mol_gen.GetNumConformers()
    num_ref = mol_ref.GetNumConformers()
    
    rmsd_confusion_mat = -1 * np.ones([num_ref, num_gen], dtype=np.float32)
    
    for i, conf_gen in enumerate(mol_gen.GetConformers()):
        if useFF:
            tmp_mol = Chem.Mol(mol_gen)
            tmp_mol.RemoveAllConformers()
            tmp_mol.AddConformer(conf_gen, assignId=True)
            MMFFOptimizeMolecule(tmp_mol)
            mol_to_use = tmp_mol
        else:
            mol_to_use = mol_gen
        
        for j, conf_ref in enumerate(mol_ref.GetConformers()):
            rmsd = GetBestRMSD(
                mol_to_use, mol_ref,
                prbId=conf_gen.GetId(),
                refId=conf_ref.GetId()
            )
            rmsd_confusion_mat[j, i] = rmsd
    
    return rmsd_confusion_mat

def evaluate_conf(mol_gen: Chem.Mol, mol_ref: Chem.Mol, useFF=False, threshold=0.5):
    """
    基于 RMSD 计算生成分子是否接近参考分子。
    
    返回:
        fraction_within_threshold: fraction，RMSD ≤ threshold 的参考构象比例
        mean_rmsd: 平均最小 RMSD
    """
    rmsd_confusion_mat = get_rmsd_confusion_matrix(mol_gen, mol_ref, useFF=useFF)
    rmsd_ref_min = rmsd_confusion_mat.min(axis=-1)  # 每个参考构象对应最接近的生成构象
    fraction_within_threshold = (rmsd_ref_min <= threshold).mean()
    mean_rmsd = rmsd_ref_min.mean()
    return rmsd_confusion_mat,fraction_within_threshold, mean_rmsd


import os
import numpy as np

def save_rmsd_text(matrix, fraction, mean_rmsd, out_dir, mol_name=None):
    os.makedirs(out_dir, exist_ok=True)
    
    # 文件名可带 mol_name，默认使用固定文件名
    matrix_file = f"{out_dir}/rmsd_matrix.txt" if mol_name is None else f"{out_dir}/rmsd_matrix_{mol_name}.txt"
    stats_file = f"{out_dir}/rmsd_stats.txt" if mol_name is None else f"{out_dir}/rmsd_stats_{mol_name}.txt"
    readable_file = f"{out_dir}/rmsd_matrix_readable.txt" if mol_name is None else f"{out_dir}/rmsd_matrix_readable_{mol_name}.txt"

    # 保存矩阵（追加模式）
    with open(matrix_file, "ab") as f:  # 使用二进制追加模式 np.savetxt 需要
        np.savetxt(f, matrix, fmt="%.6f")
        f.write(b"\n")  # 每个分子之间加个空行

    # 保存统计量（追加模式）
    with open(stats_file, "a") as f:
        f.write(f"{mol_name or 'mol'}:\n")
        f.write(f"fraction_within_threshold = {fraction:.6f}\n")
        f.write(f"mean_rmsd = {mean_rmsd:.6f}\n\n")

    # 保存可读矩阵（追加模式）
    with open(readable_file, "a") as f:
        f.write(f"RMSD Confusion Matrix for {mol_name or 'mol'}\nRows = reference conformers, Columns = generated conformers\n\n")
        for i, row in enumerate(matrix):
            row_str = " ".join(f"{x:.4f}" for x in row)
            f.write(f"Ref_{i}: {row_str}\n")
        f.write("\n")  # 分子之间加空行

