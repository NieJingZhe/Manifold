from rdkit import Chem
import numpy as np
import os

def write_multi_conformers_sdf(mol, coords_list, out_path, add_hs=False):
    """
    将多个坐标写入同一个分子并保存为 SDF（每个 coords: np.array (N,3)）。
    参数:
      mol: rdkit.Chem.Mol
      coords_list: list of np.array (N,3)
      out_path: 输出 .sdf 路径（会覆盖）
      add_hs: 如果 True 则对 mol 做 Chem.AddHs 后再写入
    返回:
      写入的构象个数
    注意:
      coords 的 atom 顺序必须与 mol 的原子顺序一致；若 mol 需要氢原子，请先确保 mol 与 coords 的原子数一致。
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if add_hs:
        mol = Chem.AddHs(mol)

    mol_copy = Chem.Mol(mol)  # 复制分子对象，避免修改原始 mol
    mol_copy.RemoveAllConformers()
    N = mol_copy.GetNumAtoms()
    if N == 0:
        raise ValueError("输入分子的原子数为 0，请确认 mol 有效。")

    writer = Chem.SDWriter(out_path)
    try:
        for coords in coords_list:
            coords = np.asarray(coords, dtype=float)
            if coords.shape != (N, 3):
                raise ValueError(f"coords.shape={coords.shape} 与分子原子数 {N} 不匹配。")

            conf = Chem.Conformer(N)
            for i in range(N):
                x, y, z = float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])
                conf.SetAtomPosition(i, (x, y, z))

            cid = mol_copy.AddConformer(conf, assignId=True)  # 返回的 cid 与 conf.GetId() 等价
            # 关键：逐个写入，每次指定 confId，确保写入多个构象
            writer.write(mol_copy, confId=cid)
    finally:
        writer.close()

    return len(coords_list),mol_copy
