"""
SFC Request Management and VNF Instances
"""
import numpy as np
import config


class VNFInstance:
    """VNF instance installed in a DC"""
    
    def __init__(self, vnf_type, dc_id):
        self.vnf_type = vnf_type
        self.dc_id = dc_id
        self.remaining_proc_time = 0.0  # ms
        self.assigned_sfc_id = None
        self.waiting_time = 0.0  # ms
    
    def is_idle(self):
        """Check if VNF is idle"""
        return self.remaining_proc_time <= 0 and self.assigned_sfc_id is None
    
    def assign(self, sfc_id, proc_time_ms, waiting_time_ms=0.0):
        """Assign VNF to an SFC request"""
        self.assigned_sfc_id = sfc_id
        self.remaining_proc_time = max(config.TIME_STEP, proc_time_ms)
        self.waiting_time = waiting_time_ms
    
    def tick(self):
        """Update VNF state after one time step"""
        if self.remaining_proc_time > 0:
            self.remaining_proc_time = max(0, self.remaining_proc_time - config.TIME_STEP)
            
            if self.remaining_proc_time <= 0:
                self.assigned_sfc_id = None
                self.waiting_time = 0.0
                self.remaining_proc_time = 0.0
    
    def get_total_delay(self):
        """Return total delay (waiting + processing)"""
        return self.waiting_time + self.remaining_proc_time


class SFCRequest:
    """SFC request with VNF chain"""
    
    def __init__(self, req_id, sfc_type, source, destination, arrival_time):
        self.id = req_id
        self.type = sfc_type
        self.specs = config.SFC_SPECS[sfc_type]
        self.chain = self.specs['chain'][:]
        self.current_vnf_index = 0
        
        self.source = source
        self.destination = destination
        self.arrival_time = arrival_time
        
        self.max_delay = self.specs['delay']  # ms
        self.elapsed_time = 0.0  # ms
        
        self.is_dropped = False
        self.is_completed = False
        self.all_vnfs_processed = False
        
        # Track placed VNFs: [(vnf_name, dc_id, prop_delay, proc_delay)]
        self.placed_vnfs = []
        
        # Total delays
        self.total_propagation_delay = 0.0  # ms
        self.total_processing_delay = 0.0   # ms
        
        # Track VNF instances processing this request
        self.processing_vnf_instances = []
    
    def get_next_vnf(self):
        """Get next VNF in chain that needs to be placed"""
        if self.current_vnf_index < len(self.chain):
            return self.chain[self.current_vnf_index]
        return None
    
    def advance_chain(self, dc_id, prop_delay=0.0, proc_delay=0.0, vnf_instance=None):
        """Advance in chain after successfully placing VNF"""
        if self.is_completed:
            return
        
        vnf_name = self.chain[self.current_vnf_index]
        self.placed_vnfs.append((vnf_name, dc_id, prop_delay, proc_delay))
        
        # Accumulate delays
        self.total_propagation_delay += prop_delay
        self.total_processing_delay += proc_delay
        
        # Track VNF instance
        if vnf_instance:
            self.processing_vnf_instances.append(vnf_instance)
        
        self.current_vnf_index += 1
    
    def check_completion(self):
        """
        Check if all VNFs are processed
        Complete when:
        1. All VNFs placed
        2. All VNFs finished processing (idle)
        """
        if self.is_completed or self.is_dropped:
            return
        
        if self.current_vnf_index >= len(self.chain):
            all_idle = all(vnf.is_idle() for vnf in self.processing_vnf_instances)
            
            if all_idle:
                self.is_completed = True
                self.all_vnfs_processed = True
    
    def get_total_e2e_delay(self):
        """Calculate total E2E delay"""
        return self.total_propagation_delay + self.total_processing_delay
    
    def get_remaining_time(self):
        """Time remaining before drop"""
        return max(0, self.max_delay - self.elapsed_time)
    
    def update_time(self):
        """Update elapsed time"""
        if self.is_completed or self.is_dropped:
            return
        
        self.elapsed_time += config.TIME_STEP
        
        # Check completion first
        self.check_completion()
        
        # Check drop condition
        if not self.is_completed and self.elapsed_time > self.max_delay:
            self.is_dropped = True
    
    def get_last_placed_dc(self):
        """Get DC ID of last placed VNF"""
        if self.placed_vnfs:
            return self.placed_vnfs[-1][1]
        return None


class SFC_Manager:
    """Manage all SFC requests"""
    
    def __init__(self):
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.req_counter = 0
    
    def reset_history(self):
        """Reset to initial state"""
        self.active_requests = []
        self.completed_history = []
        self.dropped_history = []
        self.req_counter = 0
    
    def generate_requests(self, time_step, num_dcs):
        """Generate SFC request bundles"""
        generated_count = 0
        
        for sfc_type in config.SFC_TYPES:
            if np.random.rand() < 0.3:  # 30% probability
                bundle_min, bundle_max = config.SFC_SPECS[sfc_type]['bundle']
                count = np.random.randint(bundle_min, bundle_max + 1)
                
                for _ in range(count):
                    src = np.random.randint(0, num_dcs)
                    dst = np.random.randint(0, num_dcs)
                    
                    while dst == src:
                        dst = np.random.randint(0, num_dcs)
                    
                    req = SFCRequest(self.req_counter, sfc_type, src, dst, time_step)
                    self.active_requests.append(req)
                    self.req_counter += 1
                    generated_count += 1
        
        return generated_count
    
    def clean_requests(self):
        """Move completed/dropped requests to history"""
        still_active = []
        
        for req in self.active_requests:
            if req.is_completed:
                self.completed_history.append(req)
            elif req.is_dropped:
                self.dropped_history.append(req)
            else:
                still_active.append(req)
        
        self.active_requests = still_active
    
    def get_statistics(self):
        """Calculate statistics"""
        total = self.req_counter
        accepted = len(self.completed_history)
        dropped = len(self.dropped_history)
        
        acc_ratio = (accepted / total * 100) if total > 0 else 0.0
        drop_ratio = (dropped / total * 100) if total > 0 else 0.0
        
        avg_e2e = 0.0
        if self.completed_history:
            avg_e2e = np.mean([r.get_total_e2e_delay() for r in self.completed_history])
        
        return {
            'acceptance_ratio': acc_ratio,
            'drop_ratio': drop_ratio,
            'total_generated': total,
            'total_accepted': accepted,
            'total_dropped': dropped,
            'avg_e2e_delay': avg_e2e
        }