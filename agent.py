"""
DRL Agent for SFC Provisioning
"""
import torch
import torch.optim as optim
import numpy as np
from omegaconf import DictConfig
from dqn_model import DQNModel, ReplayBuffer


class DRLAgent:
    def __init__(self, cfg: DictConfig, device='cpu'):
        self.cfg = cfg
        self.device = device
        
        # Create networks
        self.policy_net = DQNModel(cfg).to(device)
        self.target_net = DQNModel(cfg).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                    lr=cfg.training.learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(cfg.training.replay_buffer_size)
        
        # Training parameters
        self.gamma = cfg.training.gamma
        self.batch_size = cfg.training.batch_size
        self.epsilon = cfg.training.epsilon_start
        self.epsilon_end = cfg.training.epsilon_end
        self.epsilon_decay = cfg.training.epsilon_decay
        
        self.action_dim = 2 * len(cfg.vnf_types) + 1
    
    def select_action(self, state1, state2, state3, training=True):
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                state1 = torch.FloatTensor(state1).unsqueeze(0).to(self.device)
                state2 = torch.FloatTensor(state2).unsqueeze(0).to(self.device)
                state3 = torch.FloatTensor(state3).unsqueeze(0).to(self.device)
                
                q_values = self.policy_net(state1, state2, state3)
                return q_values.argmax().item()
    
    def store_transition(self, state1, state2, state3, action, reward, 
                        next_state1, next_state2, next_state3, done):
        """Store transition in replay buffer"""
        self.memory.push(state1, state2, state3, action, reward,
                        next_state1, next_state2, next_state3, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        state1, state2, state3, action, reward, next_state1, next_state2, next_state3, done = \
            self.memory.sample(self.batch_size)
        
        state1 = state1.to(self.device)
        state2 = state2.to(self.device)
        state3 = state3.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state1 = next_state1.to(self.device)
        next_state2 = next_state2.to(self.device)
        next_state3 = next_state3.to(self.device)
        done = done.to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(state1, state2, state3)
        current_q_values = current_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state1, next_state2, next_state3)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = reward + (1 - done) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, path):
        """Save model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


class StateBuilder:
    """Helper class to build state representations"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
    
    def build_state1(self, dc):
        """Build state 1: Current DC information"""
        state = []
        
        # Installed VNFs count
        for vnf_type in self.cfg.vnf_types:
            state.append(dc.installed_vnfs[vnf_type])
        
        # Available VNFs for allocation
        for vnf_type in self.cfg.vnf_types:
            available = dc.installed_vnfs[vnf_type] - dc.allocated_vnfs[vnf_type]
            state.append(available)
        
        # Available storage (normalized)
        state.append(dc.storage_available / dc.storage_total)
        
        # Available CPU (normalized)
        state.append(dc.cpu_available / dc.cpu_total)
        
        return np.array(state, dtype=np.float32)
    
    def build_state2(self, env, dc_id):
        """Build state 2: SFC processing stages by current DC"""
        state = []
        
        for sfc_type in self.cfg.sfc_types:
            sfc_id = self.cfg.sfc_types.index(sfc_type)
            
            allocated = [0] * len(self.cfg.vnf_types)
            remaining = [0] * len(self.cfg.vnf_types)
            
            for sfc in env.sfc_requests:
                if sfc.sfc_type == sfc_type and dc_id in sfc.allocated_dcs:
                    for vnf in sfc.allocated_vnfs:
                        if sfc.allocated_dcs[sfc.allocated_vnfs.index(vnf)] == dc_id:
                            vnf_idx = self.cfg.vnf_types.index(vnf)
                            allocated[vnf_idx] += 1
                    
                    next_vnf = sfc.get_next_vnf()
                    if next_vnf:
                        vnf_idx = self.cfg.vnf_types.index(next_vnf)
                        remaining[vnf_idx] += 1
            
            state.extend([sfc_id] + allocated + remaining)
        
        return np.array(state, dtype=np.float32)
    
    def build_state3(self, env):
        """Build state 3: Overall pending SFC requests"""
        state = []
        
        for sfc_type in self.cfg.sfc_types:
            sfc_id = self.cfg.sfc_types.index(sfc_type)
            
            type_requests = [sfc for sfc in env.sfc_requests if sfc.sfc_type == sfc_type]
            request_count = len(type_requests)
            
            if type_requests:
                min_time = min([sfc.get_remaining_time(env.current_time) for sfc in type_requests])
            else:
                min_time = 0
            
            bandwidth = self.cfg.sfc_characteristics[sfc_type]['bandwidth']
            
            vnf_counts = [0] * len(self.cfg.vnf_types)
            for sfc in type_requests:
                next_vnf = sfc.get_next_vnf()
                if next_vnf:
                    vnf_idx = self.cfg.vnf_types.index(next_vnf)
                    vnf_counts[vnf_idx] += 1
            
            state.extend([sfc_id, request_count, min_time / 100.0, bandwidth / 100.0] + vnf_counts)
        
        return np.array(state, dtype=np.float32)