"""
Network Environment for SFC Provisioning
"""
import numpy as np
import random
from omegaconf import DictConfig


class DataCenter:
    def __init__(self, dc_id, cpu, ram, storage):
        self.id = dc_id
        self.cpu_total = cpu
        self.ram_total = ram
        self.storage_total = storage
        self.cpu_available = cpu
        self.ram_available = ram
        self.storage_available = storage
        self.installed_vnfs = {}
        self.allocated_vnfs = {}
        
    def init_vnf_dicts(self, vnf_types):
        """Initialize VNF dictionaries"""
        self.installed_vnfs = {vnf: 0 for vnf in vnf_types}
        self.allocated_vnfs = {vnf: 0 for vnf in vnf_types}
        
    def can_install_vnf(self, vnf_type, vnf_resources):
        res = vnf_resources[vnf_type]
        return (self.cpu_available >= res['cpu'] and 
                self.ram_available >= res['ram'] and 
                self.storage_available >= res['storage'])
    
    def install_vnf(self, vnf_type, vnf_resources):
        if self.can_install_vnf(vnf_type, vnf_resources):
            res = vnf_resources[vnf_type]
            self.cpu_available -= res['cpu']
            self.ram_available -= res['ram']
            self.storage_available -= res['storage']
            self.installed_vnfs[vnf_type] += 1
            return True
        return False
    
    def uninstall_vnf(self, vnf_type, vnf_resources):
        if self.installed_vnfs[vnf_type] > self.allocated_vnfs[vnf_type]:
            res = vnf_resources[vnf_type]
            self.cpu_available += res['cpu']
            self.ram_available += res['ram']
            self.storage_available += res['storage']
            self.installed_vnfs[vnf_type] -= 1
            return True
        return False
    
    def allocate_vnf(self, vnf_type):
        if self.installed_vnfs[vnf_type] > self.allocated_vnfs[vnf_type]:
            self.allocated_vnfs[vnf_type] += 1
            return True
        return False
    
    def deallocate_vnf(self, vnf_type):
        if self.allocated_vnfs[vnf_type] > 0:
            self.allocated_vnfs[vnf_type] -= 1
            return True
        return False


class SFCRequest:
    def __init__(self, sfc_type, source, destination, request_id, sfc_config):
        self.sfc_type = sfc_type
        self.source = source
        self.destination = destination
        self.request_id = request_id
        self.chain = sfc_config['chain'].copy()
        self.bandwidth = sfc_config['bandwidth']
        self.delay_limit = sfc_config['delay']
        self.time_created = 0
        self.allocated_vnfs = []
        self.allocated_dcs = []
        self.current_vnf_index = 0
        
    def get_next_vnf(self):
        if self.current_vnf_index < len(self.chain):
            return self.chain[self.current_vnf_index]
        return None
    
    def allocate_vnf(self, vnf_type, dc_id):
        if self.get_next_vnf() == vnf_type:
            self.allocated_vnfs.append(vnf_type)
            self.allocated_dcs.append(dc_id)
            self.current_vnf_index += 1
            return True
        return False
    
    def is_completed(self):
        return self.current_vnf_index >= len(self.chain)
    
    def get_remaining_time(self, current_time):
        return self.delay_limit - (current_time - self.time_created)


class NetworkEnvironment:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.num_dcs = cfg.network.num_dcs
        self.dcs = []
        self.distance_matrix = None
        self.bandwidth_matrix = None
        self.sfc_requests = []
        self.current_time = 0
        self.satisfied_sfcs = []
        self.dropped_sfcs = []
        self.request_counter = 0
        
        self._initialize_network()
    
    def _initialize_network(self):
        # Create data centers
        for i in range(self.num_dcs):
            cpu = random.randint(*self.cfg.network.dc_cpu_range)
            dc = DataCenter(i, cpu, self.cfg.network.dc_ram, self.cfg.network.dc_storage)
            dc.init_vnf_dicts(self.cfg.vnf_types)
            self.dcs.append(dc)
        
        # Create distance matrix
        self.distance_matrix = np.random.uniform(10, 500, (self.num_dcs, self.num_dcs))
        np.fill_diagonal(self.distance_matrix, 0)
        
        # Create bandwidth matrix
        self.bandwidth_matrix = np.full((self.num_dcs, self.num_dcs), 
                                       self.cfg.network.link_bandwidth)
        np.fill_diagonal(self.bandwidth_matrix, 0)
    
    def generate_sfc_requests(self):
        """Generate new SFC requests"""
        new_requests = []
        for sfc_type in self.cfg.sfc_types:
            sfc_config = self.cfg.sfc_characteristics[sfc_type]
            bundle_min, bundle_max = sfc_config['bundle_size']
            num_requests = random.randint(bundle_min, bundle_max)
            
            for _ in range(num_requests):
                source = random.randint(0, self.num_dcs - 1)
                destination = random.randint(0, self.num_dcs - 1)
                while destination == source:
                    destination = random.randint(0, self.num_dcs - 1)
                
                request = SFCRequest(sfc_type, source, destination, 
                                   self.request_counter, sfc_config)
                request.time_created = self.current_time
                new_requests.append(request)
                self.request_counter += 1
        
        self.sfc_requests.extend(new_requests)
        return new_requests
    
    def get_shortest_path(self, source, destination):
        """Simple shortest path based on distance"""
        distances = [float('inf')] * self.num_dcs
        distances[source] = 0
        visited = [False] * self.num_dcs
        parent = [-1] * self.num_dcs
        
        for _ in range(self.num_dcs):
            min_dist = float('inf')
            u = -1
            for i in range(self.num_dcs):
                if not visited[i] and distances[i] < min_dist:
                    min_dist = distances[i]
                    u = i
            
            if u == -1:
                break
            
            visited[u] = True
            
            for v in range(self.num_dcs):
                if not visited[v] and self.distance_matrix[u][v] > 0:
                    new_dist = distances[u] + self.distance_matrix[u][v]
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        parent[v] = u
        
        # Reconstruct path
        path = []
        current = destination
        while current != -1:
            path.append(current)
            current = parent[current]
        path.reverse()
        
        return path if path[0] == source else []
    
    def check_sfc_completion(self, sfc_request):
        """Check if SFC can be completed within delay limit"""
        if not sfc_request.is_completed():
            return False
        
        # Calculate propagation delay
        prop_delay = 0
        for i in range(len(sfc_request.allocated_dcs) - 1):
            dc_i = sfc_request.allocated_dcs[i]
            dc_j = sfc_request.allocated_dcs[i + 1]
            prop_delay += self.distance_matrix[dc_i][dc_j] / self.cfg.network.speed_of_light
        
        # Calculate processing delay
        proc_delay = 0
        for vnf in sfc_request.allocated_vnfs:
            proc_delay += self.cfg.vnf_resources[vnf]['proc_time']
        
        total_delay = prop_delay + proc_delay
        elapsed_time = self.current_time - sfc_request.time_created
        
        return (total_delay + elapsed_time) <= sfc_request.delay_limit
    
    def reset(self):
        """Reset environment for new episode"""
        self._initialize_network()
        self.sfc_requests = []
        self.current_time = 0
        self.satisfied_sfcs = []
        self.dropped_sfcs = []
        self.request_counter = 0
        self.generate_sfc_requests()
    
    def step(self):
        """Advance time by one step"""
        self.current_time += 1
        
        # Check for timed-out requests
        to_remove = []
        for sfc in self.sfc_requests:
            if sfc.get_remaining_time(self.current_time) <= 0:
                self.dropped_sfcs.append(sfc)
                to_remove.append(sfc)
        
        for sfc in to_remove:
            self.sfc_requests.remove(sfc)
        
        # Generate new requests periodically
        if self.current_time % self.cfg.training.request_generation_interval == 0:
            self.generate_sfc_requests()