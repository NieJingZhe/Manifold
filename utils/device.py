import torch
from typing import Optional

def get_device(gpu: Optional[int] = None) -> torch.device:
    if gpu is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        torch.cuda.set_device(int(gpu))
        return torch.device(f'cuda:{int(gpu)}')
    except Exception:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
