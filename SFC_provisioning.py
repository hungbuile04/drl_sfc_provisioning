"""
SFC Provisioning Algorithm (Algorithm 1 from paper)
"""
import numpy as np
from omegaconf import DictConfig
from agent import StateBuilder


class SFCProvisioner:
    def __init__(self, env, agent, cfg: DictConfig):
        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.state_builder = StateBuilder(cfg)
    
    def set_dc_priority(self):
        """Set priority for DCs based on resources and pending requests"""
        if not self.env.sfc_requests:
            return list(range(len(self.env.dcs)))
        
        min_delay_sfc = min(self.env.sfc_requests, 
                           key=lambda x: x.get_remaining_time(self.env.current_time))
        
        path = self.env.get_shortest_path(min_delay_sfc.source, min_delay_sfc.destination)
        
        dc_priorities = []
        for dc in self.env.dcs:
            if dc.id in path:
                priority = path.index(dc.id)
            else:
                priority = len(path) + dc.id
            dc_priorities.append((dc.id, priority))
        
        dc_priorities.sort(key=lambda x: x[1])
        return [dc_id for dc_id, _ in dc_priorities]
    
    def get_action_type_and_vnf(self, action):
        """Decode action into action type and VNF type"""
        num_vnf_types = len(self.cfg.vnf_types)
        
        if action < num_vnf_types:
            return 'uninstall', self.cfg.vnf_types[action]
        elif action < 2 * num_vnf_types:
            vnf_idx = action - num_vnf_types
            return 'allocate', self.cfg.vnf_types[vnf_idx]
        else:
            return 'wait', None
    
    def calculate_vnf_priority(self, vnf, sfc_request, dc_id):
        """Calculate priority for VNF allocation"""
        priority = 0
        
        # P1: Based on remaining time
        remaining_time = sfc_request.get_remaining_time(self.env.current_time)
        p1 = -remaining_time
        
        # P2: SFC-based priority
        p2 = 0
        if dc_id in sfc_request.allocated_dcs:
            p2 += 10
        else:
            p2 -= len(set(sfc_request.allocated_dcs))
        
        # P3: Urgency
        p3 = 0
        if remaining_time < self.cfg.priority.threshold:
            p3 = self.cfg.priority.urgency_constant / (remaining_time + self.cfg.priority.epsilon)
        
        priority = p1 + p2 + p3
        return priority
    
    def select_vnf_for_allocation(self, vnf_type, dc_id):
        """Select highest priority VNF for allocation"""
        vnf_priorities = []
        
        for sfc in self.env.sfc_requests:
            next_vnf = sfc.get_next_vnf()
            if next_vnf == vnf_type:
                priority = self.calculate_vnf_priority(next_vnf, sfc, dc_id)
                vnf_priorities.append((sfc, priority))
        
        if not vnf_priorities:
            return None
        
        vnf_priorities.sort(key=lambda x: x[1], reverse=True)
        return vnf_priorities[0][0]
    
    def perform_action(self, action, dc_id):
        """Perform the selected action"""
        action_type, vnf_type = self.get_action_type_and_vnf(action)
        reward = self.cfg.rewards.default
        
        dc = self.env.dcs[dc_id]
        
        if action_type == 'wait':
            pass
        
        elif action_type == 'uninstall':
            if dc.installed_vnfs[vnf_type] > dc.allocated_vnfs[vnf_type]:
                has_waiting = any(sfc.get_next_vnf() == vnf_type for sfc in self.env.sfc_requests)
                
                if not has_waiting:
                    dc.uninstall_vnf(vnf_type, self.cfg.vnf_resources)
                else:
                    reward = self.cfg.rewards.uninstall_required
            else:
                reward = self.cfg.rewards.invalid_action
        
        elif action_type == 'allocate':
            if not dc.can_install_vnf(vnf_type, self.cfg.vnf_resources) and dc.installed_vnfs[vnf_type] == 0:
                if dc.install_vnf(vnf_type, self.cfg.vnf_resources):
                    pass
                else:
                    reward = self.cfg.rewards.invalid_action
                    return reward
            
            if dc.installed_vnfs[vnf_type] <= dc.allocated_vnfs[vnf_type]:
                reward = self.cfg.rewards.invalid_action
                return reward
            
            selected_sfc = self.select_vnf_for_allocation(vnf_type, dc_id)
            
            if selected_sfc:
                if dc.allocate_vnf(vnf_type):
                    selected_sfc.allocate_vnf(vnf_type, dc_id)
                    
                    if selected_sfc.is_completed():
                        if self.env.check_sfc_completion(selected_sfc):
                            reward = self.cfg.rewards.sfc_satisfied
                            self.env.satisfied_sfcs.append(selected_sfc)
                            self.env.sfc_requests.remove(selected_sfc)
                            
                            for vnf, dc_idx in zip(selected_sfc.allocated_vnfs, 
                                                   selected_sfc.allocated_dcs):
                                self.env.dcs[dc_idx].deallocate_vnf(vnf)
                        else:
                            reward = self.cfg.rewards.sfc_dropped
                            self.env.dropped_sfcs.append(selected_sfc)
                            self.env.sfc_requests.remove(selected_sfc)
                            
                            for vnf, dc_idx in zip(selected_sfc.allocated_vnfs, 
                                                   selected_sfc.allocated_dcs):
                                self.env.dcs[dc_idx].deallocate_vnf(vnf)
            else:
                reward = self.cfg.rewards.invalid_action
        
        return reward
    
    def run_episode(self, training=True):
        """Run one episode of SFC provisioning"""
        self.env.reset()
        total_reward = 0
        step_count = 0
        
        while self.env.sfc_requests and step_count < 1000:
            dc_list = self.set_dc_priority()
            
            for _ in range(self.cfg.training.actions_per_step):
                if not self.env.sfc_requests:
                    break
                
                dc_id = dc_list[0]
                
                # Build states
                state1 = self.state_builder.build_state1(self.env.dcs[dc_id])
                state2 = self.state_builder.build_state2(self.env, dc_id)
                state3 = self.state_builder.build_state3(self.env)
                
                # Select action
                action = self.agent.select_action(state1, state2, state3, training=training)
                
                # Perform action
                reward = self.perform_action(action, dc_id)
                total_reward += reward
                
                # Build next states
                next_state1 = self.state_builder.build_state1(self.env.dcs[dc_id])
                next_state2 = self.state_builder.build_state2(self.env, dc_id)
                next_state3 = self.state_builder.build_state3(self.env)
                
                done = len(self.env.sfc_requests) == 0
                
                # Store transition
                if training:
                    self.agent.store_transition(state1, state2, state3, action, reward,
                                               next_state1, next_state2, next_state3, done)
                
                dc_list = self.set_dc_priority()
            
            self.env.step()
            step_count += 1
        
        return total_reward
    
    def get_metrics(self):
        """Get performance metrics"""
        total_requests = len(self.env.satisfied_sfcs) + len(self.env.dropped_sfcs)
        
        if total_requests == 0:
            return {
                'acceptance_ratio': 0,
                'avg_e2e_delay': 0,
                'cpu_utilization': 0,
                'storage_utilization': 0,
                'satisfied_count': 0,
                'dropped_count': 0
            }
        
        acceptance_ratio = len(self.env.satisfied_sfcs) / total_requests
        
        # Calculate average E2E delay
        total_delay = 0
        for sfc in self.env.satisfied_sfcs:
            prop_delay = 0
            for i in range(len(sfc.allocated_dcs) - 1):
                dc_i = sfc.allocated_dcs[i]
                dc_j = sfc.allocated_dcs[i + 1]
                prop_delay += self.env.distance_matrix[dc_i][dc_j] / self.cfg.network.speed_of_light
            
            proc_delay = sum([self.cfg.vnf_resources[vnf]['proc_time'] for vnf in sfc.allocated_vnfs])
            total_delay += prop_delay + proc_delay
        
        avg_e2e_delay = total_delay / len(self.env.satisfied_sfcs) if self.env.satisfied_sfcs else 0
        
        # Resource utilization
        total_cpu_used = sum(dc.cpu_total - dc.cpu_available for dc in self.env.dcs)
        total_cpu = sum(dc.cpu_total for dc in self.env.dcs)
        cpu_utilization = total_cpu_used / total_cpu if total_cpu > 0 else 0
        
        total_storage_used = sum(dc.storage_total - dc.storage_available for dc in self.env.dcs)
        total_storage = sum(dc.storage_total for dc in self.env.dcs)
        storage_utilization = total_storage_used / total_storage if total_storage > 0 else 0
        
        return {
            'acceptance_ratio': acceptance_ratio,
            'avg_e2e_delay': avg_e2e_delay,
            'cpu_utilization': cpu_utilization,
            'storage_utilization': storage_utilization,
            'satisfied_count': len(self.env.satisfied_sfcs),
            'dropped_count': len(self.env.dropped_sfcs)
        }