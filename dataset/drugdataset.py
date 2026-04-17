import os
import os.path as osp
import json
import pickle


import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from .smiles2graph import smile2graph4GEOM,teacher_coords_from_smiles, center_and_rescale,q_from_Y
import torch_geometric
from models import KERNEL, UMAPLowKernel, GaussianKernel, StudentTKernel, pairwise_dist, UMAPRowExpKernel, UMAPRowFamilyKernel, SmoothKRowExpKernel, find_ab_params
from models.dist import compute_augmented_graph_distance_np, compute_AE_tanimoto_distance_np, compute_embed3d_distance_np
from .manifold import build_high_dim_probabilities
class QM9Dataset(InMemoryDataset):
    def __init__(self, name, root='data',dataset='QM9',
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        # self.dir_name = '_'.join(name.split('-'))
        self.type = type
        super(QM9Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
        self.train_index, self.valid_index, self.test_index = pickle.load(open(self.processed_paths[1], 'rb'))
        self.num_tasks = 1

    @property
    def raw_dir(self):

        return 'GEOM_Data'
    

    @property
    def raw_file_names(self):
      
        return 'GEOM_Data/QM9/train_converted_data1K.pt', 'GEOM_Data/QM9/val_converted_data100.pt', 'GEOM_Data/QM9/test_converted_data100.pt'


    @property
    def processed_dir(self):

        return 'processed_data'

    @property
    def processed_file_names(self):
        return 'data.pt','split.pt'

    def __subprocess(self, datalist):
        processed_data = []
        i = 0
        for datapoint in tqdm(datalist):

            smiles = datapoint['smiles']
            mol = datapoint['rdmol']

            # direct computation without try/except as requested
            mol_copy = Chem.Mol(mol)

            # teacher embedding / Y_true / P
            a, b = find_ab_params(min_dist=0.9, spread=1.0)
            KERNEL = UMAPLowKernel(a, b)
            mol_embedded, Y_true = teacher_coords_from_smiles(mol_copy, seed=42, optimize=True)
            if mol_embedded is not None and Y_true is not None:
                Y_true = center_and_rescale(Y_true, target_rms=1.0)
                P, _ = q_from_Y(Y_true, KERNEL)
  
                x, edge_index, edge_attr,vdw_radii = smile2graph4GEOM(datapoint)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, pos=datapoint['pos'],
                            boltzmannweight=datapoint['boltzmannweight'], idx=datapoint['idx'], rdmol=datapoint['rdmol'],
                            totalenergy=datapoint['totalenergy'], rdmol_embedded=mol_copy,vdw_radii=vdw_radii,
                            mol_embedded=mol_embedded, Y_true=Y_true, P=[P])
                P1 = build_high_dim_probabilities(mol_copy,data=data,dist_name='D1')
                P2 = build_high_dim_probabilities(mol_copy,data=data,dist_name='D2')
                P3 = build_high_dim_probabilities(mol_copy,data=data,dist_name='D3')
                data.P1 = P1
                data.P2 = P2
                data.P3 = P3
                a,b = find_ab_params(min_dist=0.5, spread=1.0)
                data.a = a
                data.b = b
                # attach matrices as list-wrapped CPU tensors to avoid collate stacking
                data.batch_num_nodes = data.num_nodes

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                processed_data.append(data)

        return processed_data, len(processed_data)

    def process(self):
        # data_list = []
        DEBUG_N =0  # <= 设置你想测试的样本数，改为 None 或 0 表示不限制

        train_data = torch.load('GEOM_Data/QM9/train_converted_data1K.pt',weights_only=False   )
        valid_data = torch.load('GEOM_Data/QM9/val_converted_data100.pt',weights_only=False)
        test_data  = torch.load('GEOM_Data/QM9/test_converted_data100.pt',weights_only=False)

        # 临时只取前 DEBUG_N 个用于测试
        if DEBUG_N and DEBUG_N > 0:
            train_data = train_data[:DEBUG_N]
            valid_data = valid_data[:DEBUG_N]
            test_data = test_data[:DEBUG_N]

        train_data_list, train_num = self.__subprocess(train_data)
        valid_data_list, valid_num = self.__subprocess(valid_data)
        test_data_list, test_num = self.__subprocess(test_data)
        data_list = train_data_list + valid_data_list + test_data_list
        train_index = list(range(train_num))
        valid_index = list(range(train_num, train_num + valid_num))
        test_index = list(range(train_num + valid_num, train_num + valid_num + test_num))
        print(f"Train: {train_num}, Valid: {valid_num}, Test: {test_num}")
        torch.save(self.collate(data_list), self.processed_paths[0])
        pickle.dump([train_index, valid_index, test_index], open(self.processed_paths[1], 'wb'))


    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
