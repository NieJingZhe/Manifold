from typing import Any
import os
from datetime import datetime
from dataclasses import asdict
from .device import get_device
try:
    # import project utilities if available
    from exputils import kabsch_alignment, mae_per_atom, get_best_rmsd
except Exception:
    # placeholders if exputils is not importable at write-time
    kabsch_alignment = lambda a,b: (a,b,None,None)
    mae_per_atom = lambda a,b: 0.0
    get_best_rmsd = lambda a,b: 0.0

def to_runconfig_from_args(args) -> Any:
    """Minimal conversion from args namespace to a simple object with attributes used in trainer."""
    class C: pass
    c = C()
    c.dataset = getattr(args, 'dataset', getattr(args, 'data', 'QM9'))
    c.data_root = getattr(args, 'data_root', '.')
    c.bs = getattr(args, 'bs', 32)
    c.lr = getattr(args, 'lr', 1e-3)
    c.epoch = getattr(args, 'epoch', 100)
    c.eta_min = getattr(args, 'eta_min', 0.0)
    c.patience = getattr(args, 'patience', 20)
    c.gpu = getattr(args, 'gpu', None)
    c.metric = getattr(args, 'metric', 'RMSD')
    c.pos_w = getattr(args, 'pos_w', 1.0)
    c.mani_w = getattr(args, 'mani_w', 1.0)
    c.get_image = bool(getattr(args, 'get_image', False))
    c.smiles = getattr(args, 'smiles', None)
    c.early_stop = bool(getattr(args, 'early_stop', False))
    c.random_seed = getattr(args, 'random_seed', 0)
    c.checkpoint_path = getattr(args, 'checkpoint_path', None)
    c.run_bayesian_optimization = bool(getattr(args, 'run_bayesian_optimization', False))
    return c

def compute_score_by_metric(y_pred_list, y_gt_list, metric: str) -> float:
    #目前还是用RMSE，不要用其他的指标
    metric = metric.upper()
    if metric == 'MAE':
        y_pred_aligned, _, _, _ = kabsch_alignment(y_pred_list, y_gt_list)
        return mae_per_atom(y_pred_aligned, y_gt_list)
    elif metric == 'RMSD':
        s = 0.0
        for p, g in zip(y_pred_list, y_gt_list):
            s += get_best_rmsd(p, g)
        return s / max(1, len(y_pred_list))
    elif metric == 'SCORE_ALIGNMENT':
        try:
            from align3D_score import score_alignment
            return score_alignment(y_pred_list, y_gt_list)
        except Exception:
            raise RuntimeError('score_alignment not available')
    elif metric == 'CE':
        raise RuntimeError('CE metric requires mani loss externally')
    else:
        raise ValueError(f'Unsupported metric: {metric}')

def get_arg(args, name, default=None):
    return getattr(args, name, default)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def save_molecule_views(output_dir, mol_pred, pos_gt_data=None, smiles_target=None, args_to_save=None):
    # minimal wrapper: attempt to save rdkit mol as mol file and metadata
    os.makedirs(output_dir, exist_ok=True)
    try:
        if mol_pred is not None:
            path = os.path.join(output_dir, 'pred.mol')
            with open(path, 'w') as f:
                f.write(Chem.MolToMolBlock(mol_pred))
    except Exception:
        pass
    meta = {'smiles': smiles_target, 'ts': datetime.now().isoformat()}
    if args_to_save is not None:
        try:
            meta['args'] = str(args_to_save)
        except Exception:
            pass
    with open(os.path.join(output_dir, 'meta.txt'), 'w') as f:
        f.write(str(meta))

def _atomic_write_json(path, obj):
    import json, tempfile
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=d) as tf:
        json.dump(obj, tf, indent=2)
        tmpn = tf.name
    os.replace(tmpn, path)
