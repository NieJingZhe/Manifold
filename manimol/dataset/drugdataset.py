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
from manifold.kernels import  UMAPLowKernel, GaussianKernel, StudentTKernel, pairwise_dist, UMAPRowExpKernel, UMAPRowFamilyKernel, SmoothKRowExpKernel, find_ab_params
from manifold.dist import compute_augmented_graph_distance_np, compute_AE_tanimoto_distance_np,compute_low_dim_adj
from .manifold import build_high_dim_probabilities
from models.model_P import PBuilderAll
class QM9Dataset(InMemoryDataset):
    def __init__(self, name, root='data',dataset='QM9',args=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        # self.dir_name = '_'.join(name.split('-'))
        self.args=args
        self.type = type
        super(QM9Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],weights_only=False)
        self.train_index, self.valid_index, self.test_index = pickle.load(open(self.processed_paths[1], 'rb'))
        self.num_tasks = 1

    @property
    def raw_dir(self):

        return 'data/GEOM_Data'
    

    @property
    def raw_file_names(self):
      #这里修改没什么用，因为我们直接load处理好的pt文件
        return '../data/GEOM_Data/QM9/train-data40k.pt', '../data/GEOM_Data/QM9/val-data.pt', '../data/GEOM_Data/QM9/test-data.pt'


    @property
    def processed_dir(self):

        return '../data/processed_data'

    @property
    def processed_file_names(self):
        return 'data.pt','split.pt'

    def __subprocess(self, datalist):
        processed_data = []
        bad_count = 0
        for datapoint in tqdm(datalist):
            smiles = datapoint.get('smiles', None)
            mol = datapoint.get('rdmol', None)
            if mol is None:
                bad_count += 1
                continue

            mol_copy = Chem.Mol(mol)

            # 直接捕获 EmbedMolecule 的 RuntimeError，遇到错误就跳过
            try:
                mol_embedded, Y_true = teacher_coords_from_smiles(mol_copy, seed=42, optimize=True)
            except RuntimeError as e:
                print(f"Skipping molecule {smiles} due to RDKit error: {e}")
                bad_count += 1
                continue

            if mol_embedded is not None and Y_true is not None:
                Y_true = center_and_rescale(Y_true, target_rms=1.0)
                a, b = find_ab_params(min_dist=self.args.min_dist, spread=self.args.spread)
                self.Pbuilder = PBuilderAll(topk=self.args.topk, temp=0.5, eps=1e-12)
                KERNEL = UMAPLowKernel(a=a, b=b)
                P, _ = q_from_Y(datapoint['pos'], KERNEL)

                if P.shape[0] != Y_true.shape[0]:
                    print(f"Warning: P is not equal to Y_true for {smiles}")

                x, edge_index, edge_attr, vdw_radii = smile2graph4GEOM(datapoint)
                data = Data(
                    x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, pos=datapoint['pos'],
                    boltzmannweight=datapoint.get('boltzmannweight', None), idx=datapoint.get('idx', None),
                    rdmol=datapoint['rdmol'], totalenergy=datapoint.get('totalenergy', None),
                    rdmol_embedded=mol_copy, vdw_radii=vdw_radii,
                    mol_embedded=mol_embedded, Y_true=Y_true, P=P.numpy()
                )

                # 构建高维概率
                data.P1 = build_high_dim_probabilities(mol_copy, data=data, dist_name='D1')
                data.P2 = build_high_dim_probabilities(mol_copy, data=data, dist_name='D2')
                data.P3 = build_high_dim_probabilities(mol_copy, data=data, dist_name='D3')
                data.a = a
                data.b = b
                print("这是a，b:",a,b)
                # attach matrices as list-wrapped CPU tensors to avoid collate stacking
                data.batch_num_nodes = data.num_nodes

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                processed_data.append(data)
            else:
                bad_count += 1

        print(f"__subprocess: processed {len(processed_data)} samples, skipped {bad_count} bad samples.")
        return processed_data, len(processed_data)



    def process(self):

        # DEBUG_N: <=0 or None means no limit; 否则只取前 DEBUG_N 个样本用于快速测试
        DEBUG_N = 0  # 想测试几十个就改为 20 / 30 / 50

        train_data = torch.load('/home/liyong/data/mani_data/origin_data/train-data40k.pt', weights_only=False)
        valid_data = torch.load('/home/liyong/data/mani_data/origin_data/val-data.pt', weights_only=False)
        test_data  = torch.load('/home/liyong/data/mani_data/origin_data/test-data.pt', weights_only=False)


        # 如果 DEBUG_N 有效则裁剪
        if DEBUG_N and DEBUG_N > 0:
            train_data = train_data[:DEBUG_N]
            valid_data = valid_data[:DEBUG_N]
            test_data = test_data[:DEBUG_N]

        # 处理数据
        train_data_list, _ = self.__subprocess(train_data)
        valid_data_list, _ = self.__subprocess(valid_data)
        test_data_list,  _ = self.__subprocess(test_data)

        # 使用实际长度来构造索引（避免不一致）
        tlen = len(train_data_list)
        vlen = len(valid_data_list)
        tstlen = len(test_data_list)

        print(f"After processing -> Train: {tlen}, Valid: {vlen}, Test: {tstlen}")

        data_list = train_data_list + valid_data_list + test_data_list

        # 如果没有任何样本，立刻报错并给出调试建议
        if len(data_list) == 0:
            raise RuntimeError(
                "No data samples were processed successfully. "
                "Please check the raw data files and preprocessing steps. "
               
            )

        # 基于真实长度创建索引
        train_index = list(range(0, tlen))
        valid_index = list(range(tlen, tlen + vlen))
        test_index  = list(range(tlen + vlen, tlen + vlen + tstlen))

        print(f"Final indices sizes -> train: {len(train_index)}, valid: {len(valid_index)}, test: {len(test_index)}")

        # 保存 processed dataset 与索引
        torch.save(self.collate(data_list), self.processed_paths[0])
        pickle.dump([train_index, valid_index, test_index], open(self.processed_paths[1], 'wb'))


    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
