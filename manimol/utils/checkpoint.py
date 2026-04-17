import os
from pathlib import Path
import torch
from typing import Any, Dict

def save_checkpoint(model, optimizer, epoch: int, best_test_score: float, path: str):
    # keep compatibility with exputils.save_checkpoint if present;
    # if not, fallback to simple torch.save
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_test_score': best_test_score
        }, path)
    except Exception:
        # last-resort: directory ensure then save
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_test_score': best_test_score
        }, str(p))

class CheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_best(self, model, optimizer, epoch: int, best_test_score: float):
        p = os.path.join(str(self.save_dir), 'best.pth')
        save_checkpoint(model, optimizer, epoch, best_test_score, p)
        return p

    def save_epoch(self, model, optimizer, epoch: int, best_test_score: float):
        p = os.path.join(str(self.save_dir), f'epoch{epoch}.pth')
        save_checkpoint(model, optimizer, epoch, best_test_score, p)
        return p
