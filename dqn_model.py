"""
Deep Q-Network Model with Multiple Inputs and Attention Layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=-1)
        return x * attention_weights


class DQNModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(DQNModel, self).__init__()
        
        num_vnf_types = len(cfg.vnf_types)
        num_sfc_types = len(cfg.sfc_types)
        
        # Input 1: Current DC information [1 x (2*|V| + 2)]
        input1_dim = 2 * num_vnf_types + 2
        self.fc1_1 = nn.Linear(input1_dim, 64)
        self.fc1_2 = nn.Linear(64, 128)
        
        # Input 2: DC SFC processing stages [|S| x (1 + 2*|V|)]
        input2_dim = num_sfc_types * (1 + 2 * num_vnf_types)
        self.fc2_1 = nn.Linear(input2_dim, 128)
        self.fc2_2 = nn.Linear(128, 128)
        
        # Input 3: Overall pending SFC requests [|S| x (4 + |V|)]
        input3_dim = num_sfc_types * (4 + num_vnf_types)
        self.fc3_1 = nn.Linear(input3_dim, 128)
        self.fc3_2 = nn.Linear(128, 128)
        
        # Concatenated layer
        combined_dim = 128 + 128 + 128
        
        # Attention layer
        self.attention = AttentionLayer(combined_dim)
        
        # Fully connected layers
        self.fc4 = nn.Linear(combined_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 128)
        
        # Output layer: Actions [2*|V| + 1]
        output_dim = 2 * num_vnf_types + 1
        self.output = nn.Linear(128, output_dim)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state1, state2, state3):
        # Process input 1
        x1 = F.relu(self.fc1_1(state1))
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.dropout(x1)
        
        # Process input 2
        x2 = F.relu(self.fc2_1(state2))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.dropout(x2)
        
        # Process input 3
        x3 = F.relu(self.fc3_1(state3))
        x3 = F.relu(self.fc3_2(x3))
        x3 = self.dropout(x3)
        
        # Concatenate
        x = torch.cat([x1, x2, x3], dim=-1)
        
        # Apply attention
        x = self.attention(x)
        
        # Fully connected layers
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        
        # Output Q-values
        q_values = self.output(x)
        
        return q_values


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state1, state2, state3, action, reward, 
             next_state1, next_state2, next_state3, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state1, state2, state3, action, reward, 
                                      next_state1, next_state2, next_state3, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        
        state1, state2, state3, action, reward, next_state1, next_state2, next_state3, done = zip(*batch)
        
        return (torch.FloatTensor(state1), torch.FloatTensor(state2), torch.FloatTensor(state3),
                torch.LongTensor(action), torch.FloatTensor(reward),
                torch.FloatTensor(next_state1), torch.FloatTensor(next_state2), torch.FloatTensor(next_state3),
                torch.FloatTensor(done))
    
    def __len__(self):
        return len(self.buffer)