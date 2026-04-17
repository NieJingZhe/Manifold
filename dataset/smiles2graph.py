# import dgl
import numpy as np
import rdkit
import torch
from rdkit import Chem
import os
from rdkit.Chem import AllChem
from models import (
    KERNEL, UMAPLowKernel, GaussianKernel, StudentTKernel, pairwise_dist,
    UMAPRowExpKernel, UMAPRowFamilyKernel, SmoothKRowExpKernel, find_ab_params,
)
from models import KERNEL, pairwise_dist  


def get_atom_features(atom):
    # The usage of features is along with the Attentive FP.
    feature = np.zeros(39)

    # Symbol
    symbol = atom.GetSymbol()
    symbol_list = ['H','B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',  'I', 'At']
    if symbol in symbol_list:
        loc = symbol_list.index(symbol)
        feature[loc] = 1
    else:
        feature[15] = 1

    # Degree
    degree = atom.GetDegree()
    if degree > 5:
        print("atom degree larger than 5. Please check before featurizing.")
        raise RuntimeError

    feature[16 + degree] = 1

    # Formal Charge
    charge = atom.GetFormalCharge()
    feature[22] = charge

    # radical electrons
    radelc = atom.GetNumRadicalElectrons()
    feature[23] = radelc

    # Hybridization
    hyb = atom.GetHybridization()
    hybridization_list = [rdkit.Chem.rdchem.HybridizationType.SP,
                          rdkit.Chem.rdchem.HybridizationType.SP2,
                          rdkit.Chem.rdchem.HybridizationType.SP3,
                          rdkit.Chem.rdchem.HybridizationType.SP3D,
                          rdkit.Chem.rdchem.HybridizationType.SP3D2]
    if hyb in hybridization_list:
        loc = hybridization_list.index(hyb)
        feature[loc + 24] = 1
    else:
        feature[29] = 1

    # aromaticity
    if atom.GetIsAromatic():
        feature[30] = 1

    # hydrogens
    hs = atom.GetNumImplicitHs()
    feature[31 + hs] = 1

    # chirality, chirality type
    if atom.HasProp('_ChiralityPossible'):
        # TODO what kind of error
        feature[36] = 1

        try:
            chi = atom.GetProp('_CIPCode')
            chi_list = ['R', 'S']
            loc = chi_list.index(chi)
            feature[37 + loc] = 1
        except KeyError:
            feature[37] = 0
            feature[38] = 0

    return feature


def get_bond_features(bond):
    feature = np.zeros(10)

    # bond type
    type = bond.GetBondType()
    bond_type_list = [rdkit.Chem.rdchem.BondType.SINGLE,
                      rdkit.Chem.rdchem.BondType.DOUBLE,
                      rdkit.Chem.rdchem.BondType.TRIPLE,
                      rdkit.Chem.rdchem.BondType.AROMATIC]
    if type in bond_type_list:
        loc = bond_type_list.index(type)
        feature[0 + loc] = 1
    else:
        print("Wrong type of bond. Please check before feturization.")
        raise RuntimeError

    # conjugation
    conj = bond.GetIsConjugated()
    feature[4] = conj

    # ring
    ring = bond.IsInRing()
    feature[5] = ring

    # stereo
    stereo = bond.GetStereo()
    stereo_list = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                   rdkit.Chem.rdchem.BondStereo.STEREOANY,
                   rdkit.Chem.rdchem.BondStereo.STEREOZ,
                   rdkit.Chem.rdchem.BondStereo.STEREOE]
    if stereo in stereo_list:
        loc = stereo_list.index(stereo)
        feature[6 + loc] = 1
    else:
        print("Wrong stereo type of bond. Please check before featurization.")
        raise RuntimeError

    return feature

def get_atom_vdw_radii(mol):
    """计算每个原子的范德华半径"""
    periodic_table = rdkit.Chem.GetPeriodicTable()
    vdw_radii = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        # 获取范德华半径 (单位: Angstrom)
        radius = periodic_table.GetRvdw(atomic_num)
        vdw_radii.append(radius)
    return torch.tensor(vdw_radii, dtype=torch.float).view(-1, 1)


def smile2graph4GEOM(data):
    mol = data['rdmol']
    # if (mol is None):
    #     return None
    src = []
    dst = []
    atom_feature = []
    bond_feature = []

    for atom in mol.GetAtoms():
        one_atom_feature = get_atom_features(atom)
        atom_feature.append(one_atom_feature)
    atom_feature = np.array(atom_feature)
    atom_feature = torch.tensor(atom_feature).float()
    vdw_radii = get_atom_vdw_radii(mol)
    if len(mol.GetBonds()) > 0:  # mol has bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            one_bond_feature = get_bond_features(bond)
            src.append(i)
            dst.append(j)
            bond_feature.append(one_bond_feature)
            src.append(j)
            dst.append(i)
            bond_feature.append(one_bond_feature)

        src = torch.tensor(src).long()
        dst = torch.tensor(dst).long()
        bond_feature = np.array(bond_feature)
        bond_feature = torch.tensor(bond_feature).float()
        edge_index = torch.vstack([src, dst])
        # graph_cur_smile = dgl.graph((src, dst), num_nodes=len(mol.GetAtoms()))
        # graph_cur_smile.ndata['x'] = atom_feature
        # graph_cur_smile.edata['x'] = bond_feature
    else:
        edge_index = torch.empty((2, 0)).long()
        bond_feature = torch.empty((0, 10)).float()

    return atom_feature, edge_index, bond_feature, vdw_radii


def teacher_coords_from_smiles(mol, seed=42, optimize=True):
    """
    用 RDKit ETKDG 生成 3D 坐标（仅重原子，保持与你可视化的原子顺序一致）。
    如果优化=True，做一次 UFF 优化以稳定键长。
    """
    #mol = Chem.AddHs(mol)  # ETKDG 更稳
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    ok = AllChem.EmbedMolecule(mol, params)
    if ok != 0:
        print("ETKDG embedding failed.")
        return None, None
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        except Exception:
            pass
    conf = mol.GetConformer(0)
    idx_map = [conf.GetAtomPosition(a.GetIdx()) for a in mol.GetAtoms()]
    Y_true = np.array([[p.x, p.y, p.z] for p in idx_map], dtype=float)
    return mol, Y_true

def center_and_rescale(Y, target_rms=1.0):
    """每个 epoch：居中并把坐标 RMS 缩放到 target_rms，防止数值漂移或坍缩。"""
    Y = Y - Y.mean(axis=0, keepdims=True)
    rms = np.sqrt((Y**2).mean())
    if rms < 1e-8:
        return Y
    return Y * (target_rms / rms)

def q_from_Y(Y: np.ndarray, KERNEL):
    
    D = pairwise_dist(Y)
    Q = KERNEL.forward(D)
    return Q, D