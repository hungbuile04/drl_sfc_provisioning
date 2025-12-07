"""
DRL Agent for SFC Provisioning
"""
import torch
import torch.optim as optim
import numpy as np
from omegaconf import DictConfig

class DRLAgent:
    def __init__(self, device='cpu'):
        return