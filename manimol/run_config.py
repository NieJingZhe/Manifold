from dataclasses import dataclass
from typing import Optional

@dataclass
class RunConfig:
    dataset: str
    data_root: str
    bs: int
    lr: float
    epoch: int
    eta_min: float
    patience: int
    gpu: Optional[int]
    metric: str
    pos_w: float
    mani_w: float
    get_image: bool
    smiles: Optional[str]
    early_stop: bool
    random_seed: int
    checkpoint_path: Optional[str]
    run_bayesian_optimization: bool
