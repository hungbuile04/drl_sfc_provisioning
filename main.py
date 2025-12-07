import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from pathlib import Path


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    return


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg: DictConfig):
    return


@hydra.main(version_base=None, config_path="conf", config_name="config")
def reconfigurability(cfg: DictConfig):
    return


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        sys.argv.pop(1)
        
        if mode == "test":
            test()
        elif mode == "reconfig":
            reconfigurability()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python main.py [train|test|reconfig]")
    else:
        # Default: train
        main()