import torch
import torch.nn as nn
import torch.nn.functional as F

def center_and_rescale(Y, eps=1e-6):
    with torch.no_grad():
        Y -= Y.mean(dim=0, keepdim=True)
        # 统一一个 RMS 尺度，避免数值漂移
        rms = Y.pow(2).mean().sqrt()
        Y /= (rms + eps)
    return Y
