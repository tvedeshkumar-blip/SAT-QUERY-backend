import os
import random
import numpy as np

def set_deterministic_seed(seed: int = 42, deterministic_cuda: bool = False):
    """
    Sets global deterministic random seeds across Python random, NumPy, and PyTorch.
    Ensures scientific reproducibility for training, evaluation, and data splits.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic_cuda:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed

if __name__ == "__main__":
    s = set_deterministic_seed(42)
    print(f"Deterministic seed set: {s}")
