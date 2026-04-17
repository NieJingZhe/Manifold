# utils/wandb_debug.py （只包含需要的函数）
import wandb
from typing import List, Dict, Any

def init_wandb(project: str, run_name: str, config: dict = None, entity: str = None):
    """
    初始化 wandb run 并返回 run。调用方应保存返回值并传给 log 函数（或使用 wandb.run）。
    """
    run = wandb.init(project=project, name=run_name, config=config, entity=entity, reinit=True)
    return run

def log_losses_scalars(candidate_records: List[Dict[str, Any]], key: str, run=None):
    """
    为每个 conformer 按 label 单独命名并上传 per-step loss 与 final_loss。
    candidate_records: list of dict, 每个 dict 至少包含:
        - 'label' (str): e.g. 'randn0' 或 'mds0'（若不提供会自动生成 'candidate'）
        - 'losses' (list[float] or None): 每 step 的 loss
        - 'final_loss' (float or None): 最终 loss
    key: 全局样本标识，例如 "train_123"
    run: wandb.Run，可选（None 时使用 wandb.run）
    """
    if run is None:
        run = wandb.run
    if run is None:
        raise RuntimeError("No wandb run available. Call init_wandb(...) first or pass run object.")

    # 规范化输入：确保 losses 为 list，处理重复 label
    seen = {}
    items = []
    for rec in candidate_records:
        label = rec.get('label', 'candidate')
        # 处理重复 label：如果重复则后缀 _1 _2 ...
        if label in seen:
            seen[label] += 1
            label = f"{label}_{seen[label]}"
        else:
            seen[label] = 0

        losses = rec.get('losses') or []
        # 确保 losses 是 list of float
        losses = [float(x) for x in losses]
        items.append({'label': label, 'losses': losses})

    max_len = max((len(it['losses']) for it in items), default=0)

    # 按时间步逐步上传：一次调用 run.log(...) 包含所有在该步有值的 conformer
    for t in range(max_len):
        log_dict = {}
        for it in items:
            if t < len(it['losses']):
                metric = f"{key}/{it['label']}/loss"
                log_dict[metric] = it['losses'][t]
        if log_dict:
            run.log(log_dict)   # 不显式传 step，wandb 会自动维护全局 step（单调）


