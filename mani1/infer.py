# infer.py（改：debug 下不保存本地 png/csv，只把每个 step 的 loss 上传到 wandb）
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.manifold import MDS
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from rdkit import Chem
from torch_geometric.utils import degree
from utils.SaveConf import write_multi_conformers_sdf
from manifold import prob_low_dim
from dataset.drugdataset import QM9Dataset
from train import Net
from dataset.drugdataset import mol_to_features
from torch_geometric.data import Data
from datetime import datetime
from utils.StableData import center_and_rescale
from utils.evalConf import evaluate_conf,save_rmsd_text 
# wandb helper（仅含初始化/结束与逐-step 标量上传）
from utils.wandb_debug import init_wandb, log_losses_scalars


def smiles2mol(smiles, device='cpu', add_hs=True):
    x, edge_index, edge_attr = mol_to_features(smiles=smiles, mol=None, add_hs=add_hs)

    x = x.to(device)
    edge_index = edge_index.to(device)
    if edge_attr is None or edge_attr.numel() == 0:
        edge_attr = None
    else:
        edge_attr = edge_attr.to(device)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def MdsInit(phat_tensor, dim=3, random_state=0):
    """用 sklearn.manifold.MDS 从 Phat 映射出初始坐标（先映射到距离矩阵）"""
    if isinstance(phat_tensor, torch.Tensor):
        P = phat_tensor.detach().cpu().numpy()
    else:
        P = np.array(phat_tensor, dtype=np.float32)

    scale = max(np.mean(P) * 2.0, 1e-3)
    D = (1.0 - P) * scale
    np.fill_diagonal(D, 0.0)

    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=random_state, n_init=1, max_iter=300)
    coords = mds.fit_transform(D)
    return coords.astype(np.float32)


def opt(phat_target, init_coords, prob_fn, steps=1000, lr=1e-4, l2=1e-7,
        repel_w=1e-3, device='cpu', track_loss=False,
        grad_clip=5.0, do_center_rescale=True, eps_rms=1e-6,
        early_stop_patience=50,args = None):
    """
    phat_target: 目标 Phat (N,N)
    init_coords: 初始化坐标 (N,3)
    prob_fn: pos -> predicted Phat
    track_loss: 是否记录每步 loss
    early_stop_patience: 早停步数阈值
    """
    phat = phat_target.to(device)
    pos = torch.tensor(init_coords, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = Adam([pos], lr=lr)
    mse = torch.nn.MSELoss()
    early_stop_patience=args.patience
    losses = [] if track_loss else None
    best_loss = float('inf')
    best_pos = None
    patience_counter = 0

    for step in range(steps):
        optimizer.zero_grad()
        pred = prob_fn(pos)
        loss_rec = mse(pred, phat)
        loss_l2 = l2 * torch.mean(pos.pow(2))
        dif = pos.unsqueeze(1) - pos.unsqueeze(0)
        d = torch.sqrt(torch.sum(dif * dif, dim=-1) + 1e-12)
        rep = repel_w * torch.sum(1.0 / (d + 1e-3))
        loss = loss_rec + loss_l2 + rep
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([pos], grad_clip)
        optimizer.step()

        if do_center_rescale:
            with torch.no_grad():
                pos = center_and_rescale(pos, eps=eps_rms)

        current_loss = float(loss.item())
        if track_loss:
            losses.append(current_loss)

        # 更新最优
        if current_loss < best_loss:
            best_loss = current_loss
            best_pos = pos.detach().cpu().numpy()
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停判定
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at step {step}, best loss={best_loss:.6f}")
            break

    # 如果早停没有触发，返回最后结果
    if best_pos is None:
        best_pos = pos.detach().cpu().numpy()
        best_loss = float(loss.item())

    return best_pos, best_loss, losses



def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    ds = QM9Dataset(root=args.data_root, name=args.dataset)
    train_idx = list(ds.train_index)
    valid_idx = list(ds.valid_index)
    test_idx = list(ds.test_index)
    if args.debug:
        print("DEBUG模式")
        train_idx = train_idx[:5]
        valid_idx = valid_idx[:1]
        test_idx = test_idx[:1]
    to_process = []
    if 'train' in args.splits: to_process.append(('train', train_idx))
    if 'valid' in args.splits: to_process.append(('valid', valid_idx))
    if 'test' in args.splits: to_process.append(('test', test_idx))

    deg = torch.zeros(10, dtype=torch.long)
    for i in range(len(ds)):
        d = ds[i]
        dd = degree(d.edge_index[1], num_nodes=d.num_nodes, dtype=torch.long)
        deg += torch.bincount(dd, minlength=deg.numel())

    net = Net(args, deg).to(device)
    state = torch.load(args.model, map_location=device)
    net.load_state_dict(state)
    net.eval()
    for p in net.parameters(): p.requires_grad = False

    os.makedirs(args.out, exist_ok=True)
    phat_dir = os.path.join(args.out, "phat")
    sdf_dir = os.path.join(args.out, "sdf")
    plots_dir = os.path.join(args.out, "plots")
    os.makedirs(phat_dir, exist_ok=True)
    os.makedirs(sdf_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    all_meta = []

    # 如果开启 debug，则初始化 wandb（project 名称：stage 2）
    wandb_run = None
    if args.debug:
        run_name = f"stage2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # 你也可以通过命令行传 entity
        wandb_run = init_wandb(project='stage 2', run_name=run_name, config=vars(args), entity=getattr(args, "entity", None))

    # 新增：如果传入了单分子 --smiles，则走单分子推理分支（不走 dataset 路径）
    if getattr(args, "smiles", None):
        smiles = args.smiles
        # 如果尚未初始化 wandb（debug 可能为 False），单分子分支也要初始化 wandb 用于上传 loss。
        if wandb_run is None:
            run_name = f"stage2_smiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_run = init_wandb(project='stage 2', run_name=run_name, config=vars(args), entity=getattr(args, "entity", None))

        print(f"Processing single SMILES: {smiles}")

        # 构建带氢的 rdkit Mol
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        # 构建 GNN 输入数据
        data = smiles2mol(smiles, device=device, add_hs=True)
        data = data.to(device)
        # 为单分子构建 batch 向量
        batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # 预测 Phat（用 net 输出）
        with torch.no_grad():
            Phat_batch = net(data.x, data.edge_index, data.edge_attr, batch_vec)
        # 对 single graph，取 mask 全为 True
        mask = torch.ones(data.x.size(0), dtype=torch.bool, device=device)
        Phat_i = Phat_batch[mask][:, mask].detach().cpu()
        Phat_i = (Phat_i + Phat_i.T) / 2.0

        # 保存 phat
        phat_path = os.path.join(phat_dir, "single_smiles.pt")
        torch.save(Phat_i, phat_path)

        N = Phat_i.shape[0]

        candidate_records = []

        # 随机初始化（始终记录并上传 loss）
        for r in range(args.rand_n):
            rng = np.random.RandomState(args.seed + r)
            init = rng.normal(loc=0.0, scale=args.rand_scale, size=(N, 3)).astype(np.float32)
            coords_opt, loss_val, losses = opt(
                phat_target=Phat_i,
                init_coords=init,
                prob_fn=lambda pos: prob_low_dim(pos, args.prob_a, args.prob_b).to(device),
                steps=args.steps, lr=args.lr, l2=args.l2, repel_w=args.repel, device=device,
                track_loss=True ,args = args # 单分子分支无论是否 debug，都记录每步 loss
            )
            candidate_records.append({'coords': coords_opt, 'final_loss': loss_val, 'losses': losses, 'label': f'rand_{r}'})

        # MDS 初始化（始终记录并上传 loss）
        for r in range(args.mds_n):
            init = MdsInit(Phat_i, dim=3, random_state=args.seed + r)
            perturb = np.random.normal(loc=0.0, scale=0.01*(r+1), size=init.shape).astype(np.float32)
            init = init + perturb

            coords_opt, loss_val, losses = opt(
                phat_target=Phat_i,
                init_coords=init,
                prob_fn=lambda pos: prob_low_dim(pos, args.prob_a, args.prob_b).to(device),
                steps=args.steps, lr=args.lr, l2=args.l2, repel_w=args.repel, device=device,
                track_loss=True,args = args  # 单分子分支无论是否 debug，都记录每步 loss
            )
            candidate_records.append({'coords': coords_opt, 'final_loss': loss_val, 'losses': losses, 'label': f'mds_{r}'})

        # 单分子分支：无论 debug 与否，都上传 loss 到 wandb（与 debug 模式一致）
        key = f"single_smiles"
        log_losses_scalars(candidate_records, key, run=wandb_run)


        # 写到 SDF（只取 coords）
        coords_for_sdf = [rec['coords'] for rec in candidate_records]
        out_sdf = os.path.join(sdf_dir, "single_smiles.sdf")
        n_written = write_multi_conformers_sdf(mol, coords_for_sdf, out_sdf)

        meta = {'split': 'smiles', 'idx': 0, 'n_atoms': int(N),
                'phat': phat_path, 'sdf': out_sdf, 'n_conf': int(n_written)}
        all_meta.append(meta)

        # 保存 meta 并退出
        torch.save(all_meta, os.path.join(args.out, "meta_all.pt"))
        print("完成 single smiles 处理。输出目录：", args.out)
        return

    # 如果没有传入单分子 smiles，则按 dataset 路径运行（原有逻辑）
    for split_name, idxs in to_process:
        print(f"处理 split={split_name}, 样本数={len(idxs)}")
        subset = Subset(ds, idxs)
        loader = DataLoader(subset, batch_size=args.bs, shuffle=False)

        for batch in tqdm(loader, desc=f"推理 Phat {split_name}"):
            batch = batch.to(device)
            with torch.no_grad():
                Phat_batch = net(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            for i, idx in enumerate(batch.idx):
                idx = int(idx)
                mask = (batch.batch == i)
                Phat_i = Phat_batch[mask][:, mask].detach().cpu()
                Phat_i = (Phat_i + Phat_i.T) / 2.0
                phat_path = os.path.join(phat_dir, f"{split_name}_{idx}.pt")
                torch.save(Phat_i, phat_path)

        # 优化并写 SDF
        for idx in tqdm(idxs, desc=f"优化坐标 {split_name}"):
            phat_path = os.path.join(phat_dir, f"{split_name}_{int(idx)}.pt")
            phat = torch.load(phat_path)
            N = phat.shape[0]

            data = ds[int(idx)]
            # smi = getattr(data, 'smiles', None)
            # mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            mol = getattr(data,'rdmol',None)
            candidate_records = []  # 每个记录包含 coords, final_loss, losses, label

            # 随机初始化
            for r in range(args.rand_n):
                rng = np.random.RandomState(args.seed + r + int(idx))
                init = rng.normal(loc=0.0, scale=args.rand_scale, size=(N, 3)).astype(np.float32)
                coords_opt, loss_val, losses = opt(
                    phat_target=prob_low_dim(data.pos,args.prob_a,args.prob_b),#改写为data中的ground truth  应该是 phat_target=phat
                    init_coords=init,
                    prob_fn=lambda pos: prob_low_dim(pos, args.prob_a, args.prob_b).to(device),
                    steps=args.steps, lr=args.lr, l2=args.l2, repel_w=args.repel, device=device,
                    track_loss=args.debug,args = args
                )
                candidate_records.append({'coords': coords_opt, 'final_loss': loss_val, 'losses': losses, 'label': f'rand_{r}'})

            # MDS 初始化
            for r in range(args.mds_n):
                init = MdsInit(phat, dim=3, random_state=args.seed + r)
                #perturb = np.random.normal(loc=0.0, scale=0.01*(r+1), size=init.shape).astype(np.float32)
                #init = init + perturb

                coords_opt, loss_val, losses = opt(
                    phat_target=prob_low_dim(data.pos, args.prob_a, args.prob_b).to(device),  #可以改为data.pos试试/phat
                    init_coords=init,
                    prob_fn=lambda pos: prob_low_dim(pos, args.prob_a, args.prob_b).to(device),
                    steps=args.steps, lr=args.lr, l2=args.l2, repel_w=args.repel, device=device,
                    track_loss=args.debug,args = args
                )
                candidate_records.append({'coords': coords_opt, 'final_loss': loss_val, 'losses': losses, 'label': f'mds_{r}'})

            # 如果 debug 模式，只上传每个 step 的 loss 到 wandb（不保存本地图/表）
            if args.debug:
                key = f"{split_name}_{int(idx)}"
                log_losses_scalars(candidate_records, key, run=wandb_run)

            # 写到 SDF（只取 coords）
            coords_for_sdf = [rec['coords'] for rec in candidate_records]
            out_sdf = os.path.join(sdf_dir, f"{split_name}_{int(idx)}.sdf")
            n_written,mol_pred = write_multi_conformers_sdf(mol, coords_for_sdf, out_sdf)
            rmsd_mat,cov,mat =evaluate_conf(mol_pred,data.rdmol,useFF=False,threshold=0.5) #这里的mat可能不太对，然后记得返回
            rmsd_dir = os.path.join(args.out, "rmsd")
            save_rmsd_text(rmsd_mat, cov, mat, rmsd_dir,data.smiles)
            meta = {'split': split_name, 'idx': int(idx), 'n_atoms': int(N),
                    'phat': phat_path, 'sdf': out_sdf, 'n_conf': int(n_written)}
            all_meta.append(meta)

        torch.save(all_meta, os.path.join(args.out, f"meta_{split_name}.pt"))

    torch.save(all_meta, os.path.join(args.out, "meta_all.pt"))
    print("完成。输出目录：", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="out/stage2_out")
    p.add_argument("--data_root", default="/home/liyong/data/mani_data")
    p.add_argument("--dataset", default="qm9")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--splits", nargs='+', default=['test','valid','train'])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--l2", type=float, default=1e-7)
    p.add_argument("--repel", type=float, default=1e-5)
    p.add_argument("--rand_n", type=int, default=0)
    p.add_argument("--mds_n", type=int, default=10)
    p.add_argument("--rand_scale", type=float, default=1.0)
    p.add_argument("--prob_a", type=float, default=0.583)
    p.add_argument("--prob_b", type=float, default=1.334)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true", help="调试模式：只取少量样本5/1/1，并将每个 step 的 loss 完整上传到 wandb")
    p.add_argument("--entity", type=str, default=None, help="wandb entity（可选）")
    p.add_argument("--model_dir", type=str, required=True,help="包含 best_model.pt 和 train_args.pt 的目录")
    p.add_argument("--smiles", type=str, default=None, help="如果提供单个 SMILES，则只对该分子进行预测和优化（覆盖 dataset 路径）")
    p.add_argument("--patience", type=int, default=50, help="早停耐心步数")

    args = p.parse_args()
    if args.model_dir:
        args.model = os.path.join(args.model_dir, "best_model.pt")
        train_args = torch.load(os.path.join(args.model_dir, "train_args.pt"))
        for k, v in (train_args.items() if isinstance(train_args, dict) else vars(train_args).items()):
            if getattr(args, k, None) is None:
                setattr(args, k, v)
        print(f"加载模型目录 {args.model_dir} -> 模型: {args.model}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out = os.path.join(args.out, f"run_{timestamp}")
    main(args)
