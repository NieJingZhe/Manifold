# infer_simp.py
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from rdkit import Chem
from torch_geometric.utils import degree
from utils.SaveConf import write_multi_conformers_sdf
from manifold import prob_low_dim
from dataset.drugdataset import QM9Dataset
from trainGPS import Net
from dataset.drugdataset import mol_to_features
from torch_geometric.data import Data
from datetime import datetime
from utils.StableData import center_and_rescale
from utils.evalConf import evaluate_conf_extended, save_rmsd_text, get_pairwise_rmsd_matrix
from utils.stage2_details import correct_chirality
from utils.wandb_debug import init_wandb, log_losses_scalars

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def opt(phat_target, init_coords, prob_fn, steps=1000, LR=1e-4, l2=1e-7,
        repel_w=1e-3, device='cpu', track_loss=False,
        grad_clip=None, do_center_rescale=False, eps_rms=1e-6,
        early_stop_patience=50, args=None):
    phat = phat_target.to(device)
    pos = torch.tensor(init_coords, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = Adam([pos], lr=LR)
    loss_func = torch.nn.BCELoss()
    patience_limit = args.patience if (args is not None and getattr(args, "patience", None) is not None) else early_stop_patience
    losses = [] if track_loss else None
    best_loss = float('inf')
    best_pos = None
    patience_counter = 0
    early_stopped = False
    stop_step = steps - 1

    for step in range(steps):
        optimizer.zero_grad()
        pred = prob_fn(pos)
        loss_rec = loss_func(pred, phat)
        loss_l2 = l2 * torch.mean(pos.pow(2))
        dif = pos.unsqueeze(1) - pos.unsqueeze(0)
        d = torch.sqrt(torch.sum(dif * dif, dim=-1) + 1e-12)
        rep = repel_w * torch.sum(1.0 / (d + 1e-3))
        loss = loss_rec + loss_l2 + rep
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([pos], grad_clip)
        optimizer.step()

        current_loss = float(loss.item())
        if track_loss:
            losses.append(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_pos = pos.detach().cpu().numpy()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience_limit:
            early_stopped = True
            stop_step = step
            break

    if best_pos is None:
        best_pos = pos.detach().cpu().numpy()
        best_loss = float(loss.item())
    return best_pos, best_loss, losses, early_stopped, stop_step, None, None

# ---------------- 使用 RDKit 计算 pairwise RMSD 的辅助函数 ----------------
def build_mol_with_conformers(mol_template: Chem.Mol, confs):
    """
    给定一个 RDKit Mol 模板（含正确的 atom ordering），移除原有构象并把 confs (list of (N,3) arrays)
    加为新的构象，返回新的分子对象（copy）。
    confs 中每个 coords 必须与 mol_template 的 atom 数一致。
    """
    mol_copy = Chem.Mol(mol_template)  # 复制分子对象
    mol_copy.RemoveAllConformers()
    n_atoms = mol_copy.GetNumAtoms()
    for coords in confs:
        conf = Chem.Conformer(n_atoms)
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, (float(x), float(y), float(z)))
        mol_copy.AddConformer(conf, assignId=True)
    return mol_copy

def hierarchical_cluster_select_rdkit(all_inits, mol_template, rand_n, hier_threshold):
    """
    使用 RDKit 的 RMSD 计算（通过 utils.evalConf.get_pairwise_rmsd_matrix）
    进行层次聚类、选择 medoid 作为代表构象，并返回 candidate_records。
    - all_inits: list of numpy arrays (Ksample, N, 3)
    - mol_template: RDKit Mol（用于构造带多个 conformers 的分子）
    - rand_n: 目标构象数（上限/参考）
    - hier_threshold: 阈值（Å），>0 表示按距离切树，否则使用 maxclust 或 silhouette（可扩展）
    """
    K = len(all_inits)
    if K == 0:
        return []
    if K == 1:
        return [{'coords': all_inits[0].astype(np.float32), 'label': 'single', 'cluster_id': 0, 'cluster_size': 1, 'medoid_index': 0, 'cluster_avg_rmsd': 0.0}]

    # 构造带 K 个构象的分子对象，交给 evalConf 的 pairwise 函数（复用 RDKit 的 GetBestRMS）
    mol_with_confs = build_mol_with_conformers(mol_template, all_inits)
    # get_pairwise_rmsd_matrix 返回 NxN 的矩阵
    D = get_pairwise_rmsd_matrix(mol_with_confs, useFF=False, upper_triangular=False)
    # condensed distance vector 用于 linkage
    condensed = squareform(D)

    Z = linkage(condensed, method='average')

    # 如果给了阈值，优先按阈值切树
    if hier_threshold and hier_threshold > 0.0:
        thr = float(hier_threshold)
        labels = fcluster(Z, t=thr, criterion='distance')
        n_clusters = len(np.unique(labels))
        # 若簇太多，逐步放大阈值直到簇数 <= rand_n（最多尝试 8 次）
        tries = 0
        while n_clusters > rand_n and tries < 8:
            thr *= 1.3  # 逐步放大阈值
            labels = fcluster(Z, t=thr, criterion='distance')
            n_clusters = len(np.unique(labels))
            tries += 1
        # 若簇太少，强制按 maxclust 切为 rand_n 簇
        if n_clusters < rand_n:
            labels = fcluster(Z, t=rand_n, criterion='maxclust')
    else:
        # 不提供阈值：直接按目标簇数强制切分
        labels = fcluster(Z, t=rand_n, criterion='maxclust')

    unique_labels = sorted(set(labels))
    candidate_records = []
    for ul in unique_labels:
        idxs = [i for i, L in enumerate(labels) if L == ul]
        cluster_size = len(idxs)
        if cluster_size == 1:
            medoid_idx = idxs[0]
            medoid_coords = all_inits[medoid_idx]
            avg_rmsd = 0.0
        else:
            subD = D[np.ix_(idxs, idxs)]
            sumd = subD.sum(axis=1)
            local_med = int(np.argmin(sumd))
            medoid_idx = idxs[local_med]
            medoid_coords = all_inits[medoid_idx]
            # average pairwise RMSD inside this cluster (exclude diagonal)
            if cluster_size > 1:
                avg_rmsd = float(subD.sum() / (cluster_size * (cluster_size - 1)))
            else:
                avg_rmsd = 0.0

        candidate_records.append({
            'coords': medoid_coords.astype(np.float32),
            'cluster_id': int(ul),
            'cluster_size': int(cluster_size),
            'medoid_index': int(medoid_idx),
            'cluster_avg_rmsd': float(avg_rmsd),
            'label': f'hier_cluster_{int(ul)}'
        })
    return candidate_records

# ---------------------------------------------- 主流程 --------------------------------------
def main(args):
    import os, random
    
    if args.seed < 0 or args.seed > 2^31-1:
        args.seed = random.randint(1, 2**31 - 1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    ds = QM9Dataset(root=args.data_root, name=args.dataset)
    train_idx = list(ds.train_index)
    valid_idx = list(ds.valid_index)
    test_idx = list(ds.test_index)
    if args.debug:
        train_idx = train_idx[:5]
        valid_idx = valid_idx[:1]
        test_idx = test_idx[:5]
    to_process = []
    if 'train' in args.splits: to_process.append(('train', train_idx))
    if 'valid' in args.splits: to_process.append(('valid', valid_idx))
    if 'test' in args.splits: to_process.append(('test', test_idx))

    deg = torch.zeros(10, dtype=torch.long)
    for i in range(len(ds)):
        d = ds[i]
        dd = degree(d.edge_index[1], num_nodes=d.num_nodes, dtype=torch.long)
        deg += torch.bincount(dd, minlength=deg.numel())

    # ---------------- 强制用训练时保存的 train_args 覆盖当前 args ----------------
    
    args.model = os.path.join(args.model_dir, "best_model.pt")
    train_args = torch.load(os.path.join(args.model_dir, "train_args.pt"), map_location="cpu")
    from types import SimpleNamespace

    train_args = SimpleNamespace(**train_args)
    net = Net(train_args, deg).to(device)

    print(">> 加载模型：", args.model)
    ckpt = torch.load(args.model, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # 允许 shape 不一致的层跳过（strict=False）
    net.load_state_dict(state_dict, strict=True)
    print(">> checkpoint 加载完成")

    net.eval()
    for p in net.parameters(): p.requires_grad = False

    os.makedirs(args.out, exist_ok=True)
    phat_dir = os.path.join(args.out, "phat")
    sdf_dir = os.path.join(args.out, "sdf")
    os.makedirs(phat_dir, exist_ok=True)
    os.makedirs(sdf_dir, exist_ok=True)

    all_meta = []

    wandb_run = None
    if args.debug:
        run_name = f"stage2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run = init_wandb(project='stage 2', run_name=run_name, config=vars(args), entity=getattr(args, "entity", None))

    # 先计算并保存 phat
    for split_name, idxs in to_process:
        subset = Subset(ds, idxs)
        loader = DataLoader(subset, batch_size=args.bs, shuffle=False)
        for batch in tqdm(loader, desc=f"推理 Phat {split_name}"):
            batch = batch.to(device)
            with torch.no_grad():
                Phat_batch,gram,pair = net(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            for i, idx in enumerate(batch.idx):
                idx = int(idx)
                mask = (batch.batch == i)
                Phat_i = Phat_batch[mask][:, mask].detach().cpu()
                phat_path = os.path.join(phat_dir, f"{split_name}_{idx}.pt")
                torch.save(Phat_i, phat_path)

        # 对每个样本做随机初始化（Ksample 次），再用 RDKit RMSD + 层次聚类 得到代表构象
        for idx in tqdm(idxs, desc=f"优化坐标 {split_name}"):
            phat_path = os.path.join(phat_dir, f"{split_name}_{int(idx)}.pt")
            phat = torch.load(phat_path)
            N = phat.shape[0]
            data = ds[int(idx)]
            mol = getattr(data,'rdmol',None)
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Chem.KekulizeException:
                print(f"Can't kekulize {data.smiles}, skip.")
                continue 
            candidate_records = []

            # 如果使用数据集中提供的 gt Q 矩阵（可选）
            if args.gt:
                if args.Qtype == 'Qmean':
                    phat = ds.q_meta_list[data.idx]['Qmean']
                elif args.Qtype == 'QWmean':
                    phat = ds.q_meta_list[data.idx]['QWmean']
                elif args.Qtype == 'Qgreatest':
                    phat = ds.q_meta_list[data.idx]['Qgreatest']
                elif args.Qtype == 'distance':
                    phat = torch.cdist(data.pos, data.pos, p=2)
                    phat.fill_diagonal_(0.0)
            # 随机初始化：Ksample 次优化，收集优化后的 coords & loss
            all_inits = []
            all_losses = []
            for k in range(args.Ksample):
                rng = np.random.RandomState(args.seed + k + int(idx))
                init = rng.normal(loc=0.0, scale=args.rand_scale, size=(N, 3)).astype(np.float32)
                coords_opt, loss_val, losses, early_stopped, stop_step, sampled_positions, sampled_losses = opt(
                    phat_target=phat,
                    init_coords=init,
                    prob_fn=lambda pos: prob_low_dim(pos, args.prob_a, args.prob_b).to(device),
                    steps=args.steps, LR=args.LR, l2=args.l2, repel_w=args.repel, device=device,
                    track_loss=args.debug, args=args
                )
                corrected = False
                #coords_opt,corrected,coords_oppo = correct_chirality(coords_opt, mol)
                
                all_inits.append(coords_opt.astype(np.float32))
                all_losses.append(float(loss_val))

            # ---------- 这里改成基于 RDKit RMSD 的层次聚类 ----------
            if args.cluster_method == "hier":
                cand = hierarchical_cluster_select_rdkit(all_inits, mol_template=mol, rand_n=args.rand_n, hier_threshold=args.hier_threshold)
                # 把 medoid_index 对应的 final_loss 补上
                for c in cand:
                    midx = c.get('medoid_index', None)
                    c['final_loss'] = all_losses[midx] if midx is not None else float('nan')
                    c['losses'] = None
                candidate_records = cand
           
            # 输出 SDF（代表构象）
            coords_for_sdf = [rec['coords'] for rec in candidate_records]
            out_sdf = os.path.join(sdf_dir, f"{split_name}_{int(idx)}.sdf")
            n_written, mol_pred = write_multi_conformers_sdf(mol, coords_for_sdf, out_sdf)

            rmsd_mat, cov, mat, intra_ref, intra_gen = evaluate_conf_extended(mol_pred, data.rdmol, useFF=False, threshold=0.5)
            if corrected == True:
                output_oppo = evaluate_conf_extended(mol_pred, data.rdmol, useFF=False, threshold=0.5)
            rmsd_dir = os.path.join(args.out, "rmsd")

            early_info = []
            for rec in candidate_records:
                info = {
                    'label': rec.get('label', ''),
                    'early_stop': bool(rec.get('early_stop', False)),
                    'stop_step': int(rec.get('stop_step', -1)),
                    'final_loss': float(rec.get('final_loss', float('nan')))
                }
                if 'cluster_size' in rec:
                    info['cluster_size'] = int(rec['cluster_size'])
                if 'cluster_avg_rmsd' in rec:
                    info['cluster_avg_rmsd'] = float(rec['cluster_avg_rmsd'])
                early_info.append(info)
 
            save_rmsd_text(
                rmsd_mat, cov, mat, rmsd_dir,
                mol_name=data.smiles,
                split_idx=f"{split_name}_{int(idx)}",
                early_info=early_info,
                intra_ref_mat=intra_ref,
                intra_gen_mat=intra_gen,corrected=corrected,
                output_oppo=output_oppo if corrected==True else None
            )
            all_meta.append({'split': split_name, 'idx': int(idx), 'n_atoms': int(N), 'phat': phat_path, 'sdf': out_sdf, 'n_conf': int(n_written)})

        torch.save(all_meta, os.path.join(args.out, f"meta_{split_name}.pt"))

    torch.save(all_meta, os.path.join(args.out, "meta_all.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out/stage2_out")
    p.add_argument("--data_root", default="/home/liyong/data/mani_data")
    p.add_argument("--dataset", default="qm9")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--splits", nargs='+', default=['test'])
    p.add_argument('--Qtype',type=str,default="Qgreatest",choices=['QWmean','Qmean','Qgreatest','distance'],)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--LR", type=float, default=1)
    p.add_argument("--l2", type=float, default=0)
    p.add_argument("--repel", type=float, default=0)

    p.add_argument("--rand_n", type=int, default=1)          # 目标最终输出构象数（作为上限/参考）
    p.add_argument("--rand_scale", type=float, default=1.0)   # 随机初始化尺度
    p.add_argument("--prob_a", type=float, default=0.583)
    p.add_argument("--prob_b", type=float, default=1.334)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--gt", action="store_true")
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--model_dir", type=str, default='/home/liyong/mani/saved_model/2025-12-13-22-32-47')
    p.add_argument("--patience", type=int, default=50)
    p.add_argument('--exp_id', type=str, default=None)
    p.add_argument('--Ksample', type=int, default=1)        # 随机初始化次数
    p.add_argument('--model', type=str, default=None)        # 可通过 model_dir 赋值覆盖

    p.add_argument('--cluster_method', type=str, default='hier', choices=['hier','kmeans'],
                   help='聚类方法：hier=层次聚类(基于RDKit RMSD)，kmeans=坐标KMeans')
    p.add_argument('--hier_threshold', type=float, default=1.0,
                   help='层次聚类切树阈值（Å）。>0 表示按阈值切树；<=0 表示按 rand_n 强制切簇。经验值：小分子 0.2-0.6Å，中等 0.5-1.5Å，大分子 1.0-2.5Å')

    # 解析参数（保持原来的 argparse 定义）
    args = p.parse_args()

    # ------- 最简加载 model_dir 中的 train_args.pt 与 best_model.pt -------
    if args.model_dir:
        model_path = os.path.join(args.model_dir, "best_model.pt")
        train_args_path = os.path.join(args.model_dir, "train_args.pt")

        # 如果 train_args 文件存在则加载，并把字段写回 args（仅写入未由命令行提供的字段）
        if os.path.exists(train_args_path):
            train_args = torch.load(train_args_path, map_location="cpu")
            train_dict = train_args if isinstance(train_args, dict) else vars(train_args)
            for k, v in train_dict.items():
                setattr(args, k, v)


        if args.model is None:
            args.model = model_path

    # 输出检查（可选，便于确认）
    print("使用 model_dir:", args.model_dir)
    print("使用 model 文件:", args.model)
    # ------------------------------------------------------------------------

    # 生成输出路径（保持原逻辑）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.exp_id is None:
        args.out = os.path.join(args.out, f"run_{timestamp}")
    else:
        args.out = os.path.join(args.out, f"run_{timestamp}_{args.exp_id}")

    main(args)
