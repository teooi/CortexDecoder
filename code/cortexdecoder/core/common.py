import os
import random

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
AMP_DTYPE = torch.bfloat16


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def torch_load_cpu(path, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    kwargs.setdefault("weights_only", False)
    return torch.load(path, **kwargs)


def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def noop():
    pass
