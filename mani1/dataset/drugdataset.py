import os.path as osp
import pickle
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from rdkit import Chem
from typing import Dict, List, Any, Tuple
from torch_geometric.transforms import AddLaplacianEigenvectorPE
# 你原来的映射表与 mol_to_features 保留不变
x_map = {
    'atomic_num': list(range(0, 119)),
    'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': ['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'SP2D','OTHER'],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': ['misc', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'],
    'is_conjugated': [False, True],
}

def mol_to_features(smiles = None, mol = None, add_hs: bool = True):
    # 与你原版完全相同（此处省略重复实现，直接复用原函数）
    from rdkit import Chem
    if add_hs and smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = [
            x_map['atomic_num'].index(atom.GetAtomicNum()),
            x_map['chirality'].index(str(atom.GetChiralTag())),
            x_map['degree'].index(atom.GetTotalDegree()),
            x_map['formal_charge'].index(atom.GetFormalCharge()),
            x_map['num_hs'].index(atom.GetTotalNumHs()),
            x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()),
            x_map['hybridization'].index(str(atom.GetHybridization())),
            x_map['is_aromatic'].index(atom.GetIsAromatic()),
            x_map['is_in_ring'].index(atom.IsInRing()),
        ]
        xs.append(x)
    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e = [
            e_map['bond_type'].index(str(bond.GetBondType())),
            e_map['stereo'].index(str(bond.GetStereo())),
            e_map['is_conjugated'].index(bond.GetIsConjugated()),
        ]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if len(edge_indices) > 0 else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long) if len(edge_attrs) > 0 else torch.empty((0, 3), dtype=torch.long)

    perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort() if edge_index.numel() > 0 else torch.tensor([], dtype=torch.long)
    if perm.numel() > 0:
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

    return x, edge_index, edge_attr

# ==== 抽离出来的通用处理函数 ====
# 这个函数把一个 dataset: Dict[smiles, conformer-list] -> data_list, q_meta_list
# 保持与你原方法内的逻辑一致
def process_dataset_dict(dataset: Dict[str, List[Dict[str, Any]]], desc: str, add_hs: bool = False):
    """
    dataset: Dict[smiles: str, conformers: List[Dict]]
      每个 conformer dict 至少包含 'rdmol', 'boltzmannweight', 'totalenergy'
    desc: tqdm 描述
    
    返回: data_list, q_meta_list
    """
    from manifold.dist import prob_low_dim  # 确保在函数内导入以兼容模块结构
    data_list = []
    q_meta_list = []
    iterator = tqdm(dataset.items(), desc=desc)
    lapPE = AddLaplacianEigenvectorPE(k=3, attr_name=None,is_undirected=True)

    for smiles, conformers in iterator:
        if len(conformers) == 0:
            continue

        first_conf = conformers[0]
        rdmol = first_conf['rdmol']
        try:
            rdmol = Chem.RemoveHs(first_conf['rdmol'])
        except Chem.AtomValenceException:
            print(smiles)
            continue  
        boltz0 = first_conf['boltzmannweight']
        energy0 = first_conf['totalenergy']

        # 图结构
        x, edge_index, edge_attr = mol_to_features(smiles=smiles, mol=rdmol, add_hs=add_hs)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data = lapPE(data)

        conf0 = rdmol.GetConformer(0)
        pos0_noH = conf0.GetPositions()
        data.pos = torch.tensor(pos0_noH, dtype=torch.float)

        # 其他标量属性
        data.rdmol = Chem.RemoveHs(rdmol)
        data.smiles = smiles
        data.boltzmannweight = torch.tensor([float(boltz0)], dtype=torch.float)
        data.totalenergy = torch.tensor([float(energy0)], dtype=torch.float)

        # 计算每个构象的 Q（返回 n x n tensor）
        q_list = []
        boltz_weights = []
        pos_list = []
        for conf in conformers:
            try:
                
                conf_rdmol = Chem.RemoveHs(conf['rdmol'])
                conf['rdmol'] =conf_rdmol
            except Chem.AtomValenceException:
                print(smiles)
                continue          
            conf_pos = conf_rdmol.GetConformer(0).GetPositions()
            conf_pos_t = torch.tensor(conf_pos, dtype=torch.float)  # (n,3)
            pos_list.append(conf_pos_t)
            Q = prob_low_dim(conf_pos_t)
            q_list.append(Q)
            boltz_weights.append(float(conf['boltzmannweight']))

        #q_stack = torch.stack(q_list)   # C x n x n
        bw = torch.tensor(boltz_weights, dtype=torch.float)
        if bw.sum() == 0:
            bw = torch.ones_like(bw)
        bw = bw / bw.sum()
        max_idx = torch.argmax(bw).item()

        q_meta = {
            'q_list': q_list,
            'Qgreatest':q_list[0],
            # 'Qmean': torch.mean(q_stack, dim=0),
            # 'Qgreatest': q_stack[max_idx],
            # 'QWmean': torch.sum(q_stack * bw.view(-1,1,1), dim=0),
            # 'num_conformers': q_stack.size(0),
            # 'n_atoms': q_stack.size(1),
            'smiles': smiles,
            'boltzmannweights': bw,
            'positions': pos_list,
        }

        data_list.append(data)
        q_meta_list.append(q_meta)

    return data_list, q_meta_list

# ==== QM9Dataset 只保留轻量化类定义，process 调用通用函数 ====
class QM9Dataset(InMemoryDataset):
    def __init__(self, name='QM9', root='data', args=None, add_hs=False,
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        self.args = args
        self.add_hs = add_hs if args is None else getattr(args, 'add_hs', add_hs)
        super(QM9Dataset, self).__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        # 加载 processed 内容（若不存在会触发 process）
        self.data, self.slices, self.q_meta_list = torch.load(self.processed_paths[0], map_location='cpu')
        self.train_index, self.valid_index, self.test_index = pickle.load(open(self.processed_paths[1], 'rb'))

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed_noH_lapPE')

    @property
    def raw_file_names(self):
        # 根据实际调整
        return ['train-data40k_merged.pt', 'val-data_merged.pt', 'test-data_merged.pt']

    @property
    def processed_file_names(self):
        return ['data.pt', 'split.pt']

    def process(self):
        # 只负责文件加载、调用通用处理函数、合并与保存 —— 不包含重复逻辑
        raw_paths = [osp.join(self.raw_dir, p) for p in self.raw_file_names]
        train_raw = torch.load(raw_paths[0], map_location='cpu')
        valid_raw = torch.load(raw_paths[1], map_location='cpu')
        test_raw  = torch.load(raw_paths[2], map_location='cpu')
        
        train_data_list, train_q_meta = process_dataset_dict(train_raw, desc='process train', add_hs=self.add_hs)
        valid_data_list, valid_q_meta = process_dataset_dict(valid_raw, desc='process valid', add_hs=self.add_hs)
        test_data_list,  test_q_meta  = process_dataset_dict(test_raw,  desc='process test',  add_hs=self.add_hs)

        # 合并
        data_list = train_data_list + valid_data_list + test_data_list
        q_meta_list = train_q_meta + valid_q_meta + test_q_meta

        # 给每个 Data 添加 idx
        for i, data in enumerate(data_list):
            data.idx = i

        # 构造索引
        tlen = len(train_data_list)
        vlen = len(valid_data_list)
        ttlen = len(test_data_list)
        train_index = list(range(0, tlen))
        valid_index = list(range(tlen, tlen + vlen))
        test_index  = list(range(tlen + vlen, tlen + vlen + ttlen))

        # collate data_list
        data, slices = self.collate(data_list)

        # 保存
        torch.save((data, slices, q_meta_list), self.processed_paths[0])
        pickle.dump([train_index, valid_index, test_index], open(self.processed_paths[1], 'wb'))

    def __repr__(self):
        return f'QM9Dataset({self.name}, len={len(self)})'

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data.idx = torch.tensor(idx, dtype=torch.long)
        return data



class DrugsDataset(QM9Dataset):
    def __init__(self, name='Drugs', root='data', args=None, add_hs=False,
                 transform=None, pre_transform=None, pre_filter=None):
        # 通过传参覆盖 name/root/add_hs 即可复用 QM9Dataset 中的 process 实现
        super(DrugsDataset, self).__init__(name=name, root=root, args=args, add_hs=add_hs,
                                           transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    @property
    def raw_file_names(self):
        return ['train_merged.pt', 'valid_merged.pt', 'test_merged.pt']

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed_noH_lapPE')

    @property
    def processed_file_names(self):
        return ['data.pt', 'split.pt']
