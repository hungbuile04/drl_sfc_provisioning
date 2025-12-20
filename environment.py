"""
NFV Environment - Gymnasium Environment for SFC Provisioning
"""
import gymnasium as gym
import numpy as np

import config
from sfc_provisioning import VNFInstance, SFC_Manager


class DataCenter:
    """Data Center with resources"""
    
    def __init__(self, dc_id):
        self.id = dc_id
        self.cpu = config.DC_CPU_CYCLES
        self.ram = config.DC_RAM
        self.storage = config.DC_STORAGE
        self.installed_vnfs = []
    
    def has_resources(self, vnf_type):
        """Check if DC has enough resources"""
        specs = config.VNF_SPECS[vnf_type]
        return (self.cpu >= specs['cpu'] and self.ram >= specs['ram'] and self.storage >= specs['storage'])
    
    def consume_resources(self, vnf_type):
        """Consume resources to install VNF"""
        if self.has_resources(vnf_type):
            specs = config.VNF_SPECS[vnf_type]
            self.cpu -= specs['cpu']
            self.ram -= specs['ram']
            self.storage -= specs['storage']
            return True
        return False
    
    def release_resources(self, vnf_type):
        """Release resources when uninstalling VNF"""
        specs = config.VNF_SPECS[vnf_type]
        self.cpu += specs['cpu']
        self.ram += specs['ram']
        self.storage += specs['storage']
    
    def get_idle_vnf(self, vnf_type):
        """Find an idle VNF instance of given type"""
        for vnf in self.installed_vnfs:
            if vnf.vnf_type == vnf_type and vnf.is_idle():
                return vnf
        return None
    
    def count_vnf_type(self, vnf_type):
        """Count VNF instances of a type"""
        return sum(1 for v in self.installed_vnfs if v.vnf_type == vnf_type)
    
    def count_idle_vnf_type(self, vnf_type):
        """Count idle VNF instances of a type"""
        return sum(1 for v in self.installed_vnfs if v.vnf_type == vnf_type and v.is_idle())


class TopologyManager:
    """Network topology and propagation delay calculation"""
    
    def __init__(self, num_dcs):
        self.num_dcs = num_dcs
        self.distance_matrix = self._generate_distance_matrix()
        self.bw_matrix = np.full((num_dcs, num_dcs), config.LINK_BW_CAPACITY, dtype=float)
        np.fill_diagonal(self.bw_matrix, 0)
    
    def _generate_distance_matrix(self):
        """Generate random distance matrix (km)"""
        W = np.zeros((self.num_dcs, self.num_dcs))
        for i in range(self.num_dcs):
            for j in range(i + 1, self.num_dcs):
                distance = np.random.uniform(100, 1000)
                W[i, j] = distance
                W[j, i] = distance
        return W
    
    def get_propagation_delay(self, dc_i, dc_j):
        """Calculate propagation delay between two DCs (ms)"""
        if dc_i == dc_j:
            return 0.0
        distance_km = self.distance_matrix[dc_i, dc_j]
        delay_seconds = distance_km / config.SPEED_OF_LIGHT
        return delay_seconds * 1000
    
    def get_shortest_path_dcs(self, source, destination):
        """Find shortest path using Dijkstra"""
        if source == destination:
            return [source]
        
        dist = [float('inf')] * self.num_dcs
        prev = [None] * self.num_dcs
        visited = [False] * self.num_dcs
        dist[source] = 0
        
        for _ in range(self.num_dcs):
            min_dist = float('inf')
            u = -1
            for i in range(self.num_dcs):
                if not visited[i] and dist[i] < min_dist:
                    min_dist = dist[i]
                    u = i
            
            if u == -1:
                break
            
            visited[u] = True
            
            for v in range(self.num_dcs):
                if not visited[v] and self.distance_matrix[u, v] > 0:
                    alt = dist[u] + self.distance_matrix[u, v]
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
        
        # Reconstruct path
        path = []
        current = destination
        while current is not None:
            path.insert(0, current)
            current = prev[current]
        
        if path[0] != source:
            return [source, destination]
        
        return path


class ActionController:
    """Execute actions from DRL agent"""
    
    def __init__(self, sfc_manager, topology_manager):
        self.manager = sfc_manager
        self.topology = topology_manager
        self.accepted_count = 0
        self.dropped_count = 0
    
    def execute_action(self, action, curr_dc):
        """Execute action and return (reward, is_completed)"""
        if action == 0:
            return config.REWARD_WAIT, False
        
        vnf_idx = (action - 1) % config.NUM_VNF_TYPES
        vnf_name = config.VNF_TYPES[vnf_idx]
        is_alloc = action > config.NUM_VNF_TYPES
        
        if is_alloc:
            return self._handle_allocation(curr_dc, vnf_name)
        else:
            return self._handle_uninstall(curr_dc, vnf_name)
    
    def _handle_allocation(self, dc, vnf_name):
        """Handle VNF allocation action"""
        # Find best request needing this VNF
        candidates = []
        for req in self.manager.active_requests:
            if not req.is_completed and not req.is_dropped and req.get_next_vnf() == vnf_name:
                priority = self._calculate_vnf_priority(req, dc.id)
                candidates.append((priority, req))
        
        if not candidates:
            return config.REWARD_INVALID, False
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_req = candidates[0][1]
        
        # Get or create VNF instance
        vnf_instance = dc.get_idle_vnf(vnf_name)
        
        if vnf_instance is None:
            if not dc.has_resources(vnf_name):
                return config.REWARD_INVALID, False
            
            dc.consume_resources(vnf_name)
            vnf_instance = VNFInstance(vnf_name, dc.id)
            dc.installed_vnfs.append(vnf_instance)
        
        # Calculate delays
        prop_delay = 0.0
        last_dc = best_req.get_last_placed_dc()
        if last_dc is not None and last_dc != dc.id:
            prop_delay = self.topology.get_propagation_delay(last_dc, dc.id)
        
        proc_time = config.VNF_SPECS[vnf_name]['proc_time']
        waiting_time = 0.0
        
        # Assign VNF
        vnf_instance.assign(best_req.id, proc_time, waiting_time)
        
        # Advance chain
        total_proc_delay = proc_time + waiting_time
        best_req.advance_chain(dc.id, prop_delay, total_proc_delay, vnf_instance)
        
        # Check completion
        best_req.check_completion()
        
        if best_req.is_completed:
            self.accepted_count += 1
            return config.REWARD_SATISFIED, True
        
        return config.REWARD_SATISFIED / 2.0, False
    
    def _handle_uninstall(self, dc, vnf_name):
        """Handle VNF uninstall action"""
        is_needed = any(r.get_next_vnf() == vnf_name for r in self.manager.active_requests)
        
        if is_needed:
            return config.REWARD_UNINSTALL_NEEDED, False
        
        vnf_to_remove = None
        for vnf in dc.installed_vnfs:
            if vnf.vnf_type == vnf_name and vnf.is_idle():
                vnf_to_remove = vnf
                break
        
        if vnf_to_remove:
            dc.installed_vnfs.remove(vnf_to_remove)
            dc.release_resources(vnf_name)
            return 0.0, False
        
        return config.REWARD_INVALID, False
    
    def _calculate_vnf_priority(self, req, dc_id):
        """Calculate VNF priority for request selection"""
        # P1: Time priority
        p1 = req.elapsed_time - req.max_delay
        
        # P2: Affinity
        p2 = 0.0
        last_dc = req.get_last_placed_dc()
        if last_dc is not None:
            if last_dc == dc_id:
                p2 = config.PRIORITY_P2_SAME_DC
            else:
                p2 = config.PRIORITY_P2_DIFF_DC
        
        # P3: Urgency
        remaining_time = req.get_remaining_time()
        p3 = 0.0
        if remaining_time < config.URGENCY_THRESHOLD:
            p3 = config.URGENCY_CONSTANT_C / (remaining_time + config.EPSILON_SMALL)
        
        return p1 + p2 + p3


class PriorityManager:
    """Manage DC priority for iteration"""
    
    def __init__(self, topology_manager):
        self.topology = topology_manager
    
    def get_dc_priority_order(self, dcs, active_requests):
        """Calculate DC priority order"""
        if not active_requests:
            return list(range(len(dcs)))
        
        # Find request with smallest E2E delay constraint
        min_delay_req = min(active_requests, key=lambda r: r.max_delay)
        source_dc = min_delay_req.source
        dest_dc = min_delay_req.destination
        
        # Find shortest path
        path_dcs = self.topology.get_shortest_path_dcs(source_dc, dest_dc)
        
        # Calculate priority scores
        priority_scores = {}
        
        for dc_id in range(len(dcs)):
            if dc_id == source_dc:
                priority_scores[dc_id] = 1000.0
            elif dc_id in path_dcs:
                path_index = path_dcs.index(dc_id)
                priority_scores[dc_id] = 500.0 - path_index * 10
            else:
                priority_scores[dc_id] = 0.0
        
        sorted_dcs = sorted(priority_scores.keys(), key=lambda x: priority_scores[x], reverse=True)
        
        return sorted_dcs


class Simulator:
    """Simulation time and traffic generation"""
    
    def __init__(self, sfc_manager, dcs):
        self.manager = sfc_manager
        self.dcs = dcs
        self.sim_time = 0
        self.has_generated_initial = False
    
    def reset(self):
        """Reset simulator"""
        self.sim_time = 0
        self.has_generated_initial = False
    
    def advance_time(self):
        """Advance time by one step"""
        self.sim_time += config.TIME_STEP
        penalty = 0.0
        
        # Update VNFs
        for dc in self.dcs:
            for vnf in dc.installed_vnfs:
                vnf.tick()
        
        # Update requests
        newly_dropped = []
        for req in self.manager.active_requests[:]:
            was_dropped = req.is_dropped
            req.update_time()
            if not was_dropped and req.is_dropped:
                newly_dropped.append(req)
        
        penalty = len(newly_dropped) * config.REWARD_DROPPED
        
        # Clean requests
        self.manager.clean_requests()
        
        # Generate new traffic
        if self.sim_time < config.TRAFFIC_STOP_TIME:
            if not self.has_generated_initial:
                self.manager.generate_requests(self.sim_time, len(self.dcs))
                self.has_generated_initial = True
            elif self.sim_time % config.TRAFFIC_GEN_INTERVAL == 0:
                self.manager.generate_requests(self.sim_time, len(self.dcs))
        
        return penalty
    
    def is_done(self):
        """Check if episode is done"""
        if self.sim_time >= config.MAX_SIM_TIME_PER_EPISODE:
            return True
        
        if self.sim_time > config.TRAFFIC_STOP_TIME:
            if len(self.manager.active_requests) == 0:
                return True
        
        return False
    
    def should_generate_initial_traffic(self):
        """Check if should generate initial traffic"""
        return not self.has_generated_initial


class Observer:
    """Create state representation"""
    
    @staticmethod
    def get_full_state(curr_dc, sfc_manager):
        """Generate state tuple (s1, s2, s3)"""
        # Input 1: DC state
        s1 = np.array([curr_dc.cpu, curr_dc.storage], dtype=np.float32)
        
        installed_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        idle_counts = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
        
        vnf_map = {v: i for i, v in enumerate(config.VNF_TYPES)}
        
        for vnf in curr_dc.installed_vnfs:
            idx = vnf_map[vnf.vnf_type]
            installed_counts[idx] += 1
            if vnf.is_idle():
                idle_counts[idx] += 1
        
        s1 = np.concatenate([s1, installed_counts, idle_counts])
        
        # Group requests by type
        reqs_by_type = {t: [] for t in config.SFC_TYPES}
        for req in sfc_manager.active_requests:
            if not req.is_completed and not req.is_dropped:
                reqs_by_type[req.type].append(req)
        
        s2_matrix = np.zeros((config.NUM_SFC_TYPES, 1 + 2 * config.NUM_VNF_TYPES), dtype=np.float32)
        s3_matrix = np.zeros((config.NUM_SFC_TYPES, 4 + config.NUM_VNF_TYPES), dtype=np.float32)
        
        for i, sfc_type in enumerate(config.SFC_TYPES):
            reqs = reqs_by_type[sfc_type]
            
            # Input 3: Global state
            count = len(reqs)
            s3_matrix[i, 0] = count
            
            if count > 0:
                avg_remaining = np.mean([r.get_remaining_time() for r in reqs])
                s3_matrix[i, 1] = avg_remaining
            
            s3_matrix[i, 2] = config.SFC_SPECS[sfc_type]['bw']
            
            pending_vnfs = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            total_pending = 0
            
            # Input 2: DC-specific state
            dc_relevant_count = 0
            allocated_in_dc = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            remaining_in_chain = np.zeros(config.NUM_VNF_TYPES, dtype=np.float32)
            
            for req in reqs:
                next_vnf = req.get_next_vnf()
                if next_vnf:
                    pending_vnfs[vnf_map[next_vnf]] += 1
                    total_pending += 1
                
                is_involved = False
                
                for vnf_name, dc_id, _, _ in req.placed_vnfs:
                    if dc_id == curr_dc.id:
                        allocated_in_dc[vnf_map[vnf_name]] += 1
                        is_involved = True
                
                if next_vnf and curr_dc.has_resources(next_vnf):
                    is_involved = True
                
                if is_involved:
                    dc_relevant_count += 1
                    for k in range(req.current_vnf_index, len(req.chain)):
                        remaining_in_chain[vnf_map[req.chain[k]]] += 1
            
            s3_matrix[i, 3] = total_pending
            s3_matrix[i, 4:] = pending_vnfs
            
            s2_matrix[i, 0] = dc_relevant_count
            s2_matrix[i, 1:1+config.NUM_VNF_TYPES] = allocated_in_dc
            s2_matrix[i, 1+config.NUM_VNF_TYPES:] = remaining_in_chain
        
        return (s1, s2_matrix.flatten(), s3_matrix.flatten())


class Env(gym.Env):
    """Custom Gymnasium Environment for SFC Provisioning"""
    
    def __init__(self):
        super().__init__()
        
        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE_SIZE)
        
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=np.inf, shape=(2 * config.NUM_VNF_TYPES + 2,), dtype=np.float32),
            gym.spaces.Box(low=0, high=np.inf, shape=(config.NUM_SFC_TYPES * (1 + 2 * config.NUM_VNF_TYPES),), dtype=np.float32),
            gym.spaces.Box(low=0, high=np.inf, shape=(config.NUM_SFC_TYPES * (4 + config.NUM_VNF_TYPES),), dtype=np.float32)
        ))
        
        self.sfc_manager = SFC_Manager()
        self.dcs = []
        self.topology = None
        self.simulator = None
        self.controller = None
        self.priority_manager = None
        
        self.count_step = 0
        self.current_dc_idx = 0
        self.dc_priority_order = []
        self.actions_this_step = 0
    
    def reset(self, num_dcs=None, seed=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        n = num_dcs if num_dcs else np.random.randint(2, config.MAX_NUM_DCS + 1)
        self.dcs = [DataCenter(i) for i in range(n)]
        
        self.topology = TopologyManager(n)
        
        self.sfc_manager = SFC_Manager()
        self.sfc_manager.reset_history()
        
        self.controller = ActionController(self.sfc_manager, self.topology)
        self.simulator = Simulator(self.sfc_manager, self.dcs)
        self.priority_manager = PriorityManager(self.topology)
        
        self.simulator.reset()
        
        if self.simulator.should_generate_initial_traffic():
            self.sfc_manager.generate_requests(0, len(self.dcs))
            self.simulator.has_generated_initial = True
        
        self._update_dc_priority_order()
        self.current_dc_idx = 0
        self.actions_this_step = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Get observation from current DC"""
        if not self.dc_priority_order:
            self._update_dc_priority_order()
        
        curr_dc = self.dcs[self.dc_priority_order[self.current_dc_idx]]
        return Observer.get_full_state(curr_dc, self.sfc_manager)
    
    def _get_valid_actions_mask(self):
        """Get valid actions mask for current DC"""
        curr_dc = self.dcs[self.dc_priority_order[self.current_dc_idx]]
        mask = np.zeros(config.ACTION_SPACE_SIZE, dtype=bool)
        
        mask[0] = True  # WAIT always valid
        
        # Uninstall actions
        for i, vnf_type in enumerate(config.VNF_TYPES):
            if curr_dc.count_idle_vnf_type(vnf_type) > 0:
                mask[i + 1] = True
        
        # Allocation actions
        needed_vnfs = set()
        for req in self.sfc_manager.active_requests:
            if not req.is_completed and not req.is_dropped:
                next_vnf = req.get_next_vnf()
                if next_vnf:
                    needed_vnfs.add(next_vnf)
        
        for i, vnf_type in enumerate(config.VNF_TYPES):
            if vnf_type in needed_vnfs:
                has_idle = curr_dc.get_idle_vnf(vnf_type) is not None
                has_resources = curr_dc.has_resources(vnf_type)
                if has_idle or has_resources:
                    mask[config.NUM_VNF_TYPES + 1 + i] = True
        
        return mask
    
    def _update_dc_priority_order(self):
        """Update DC priority order"""
        self.dc_priority_order = self.priority_manager.get_dc_priority_order(
            self.dcs, self.sfc_manager.active_requests
        )
    
    def step(self, action):
        """Execute one step"""
        curr_dc_id = self.dc_priority_order[self.current_dc_idx]
        curr_dc = self.dcs[curr_dc_id]
        
        # Execute action
        reward, sfc_completed = self.controller.execute_action(action, curr_dc)
        
        # Move to next DC
        self.current_dc_idx = (self.current_dc_idx + 1) % len(self.dcs)
        
        # Count actions
        self.actions_this_step += 1
        
        # Advance time after A actions
        if self.actions_this_step >= config.ACTIONS_PER_TIME_STEP:
            drop_penalty = self.simulator.advance_time()
            reward += drop_penalty
            
            self.actions_this_step = 0
            self._update_dc_priority_order()
            self.current_dc_idx = 0
        
        # Check done
        done = self.simulator.is_done()
        
        # Get info
        stats = self.sfc_manager.get_statistics()
        info = {
            'acceptance_ratio': stats['acceptance_ratio'],
            'action_mask': self._get_valid_actions_mask(),
            'total_generated': stats['total_generated'],
            'avg_e2e_delay': stats['avg_e2e_delay']
        }
        
        return self._get_obs(), reward, done, False, info
    
class BaselineAgent:
    """Baseline heuristic agent with multiple simple policies.

    Supported policies:
      - 'random'     : choose a random valid action
      - 'greedy'     : allocate for the most-needed VNF (by pending count / urgency) if possible,
                       otherwise uninstall an idle VNF not needed; fallback WAIT
      - 'first_fit'  : allocate the first VNF type (by VNF_TYPES order) that is needed and valid,
                       otherwise uninstall first idle VNF; fallback WAIT
      - 'best_fit'   : prefer allocations that reuse idle instances in current DC; otherwise
                       choose needed VNF with smallest CPU footprint; fallback uninstall / WAIT

    The agent implements get_action(state, epsilon, valid_actions_mask=...) so it can be
    used interchangeably with the DRL agent in utils.py.
    """
    def __init__(self, policy='greedy'):
        self.policy = policy

    def get_action(self, state, epsilon, valid_actions_mask=None):
        # Exploration
        if valid_actions_mask is None:
            # if mask missing, fallback to WAIT
            return 0

        mask = np.array(valid_actions_mask, dtype=bool)
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return 0

        if np.random.rand() < epsilon:
            return int(np.random.choice(valid_indices))

        # state is tuple (s1, s2_flat, s3_flat) as produced by Observer.get_full_state
        try:
            s1, s2_flat, s3_flat = state
        except Exception:
            # not a tuple -> random valid
            return int(np.random.choice(valid_indices))

        num_vnfs = config.NUM_VNF_TYPES
        # s1 layout: [cpu, storage, installed_counts..., idle_counts...]
        idle_counts = s1[2 + num_vnfs: 2 + 2 * num_vnfs] if len(s1) >= 2 + 2 * num_vnfs else np.zeros(num_vnfs)

        # s3 is flattened blocks per SFC type, each block has (4 + NUM_VNF_TYPES)
        block_size = 4 + num_vnfs
        s3 = np.array(s3_flat, dtype=float)
        pending_per_vnf = np.zeros(num_vnfs, dtype=float)
        avg_remaining_per_sfc = []
        for i in range(config.NUM_SFC_TYPES):
            base = i * block_size
            # total_pending = s3[base + 3]
            pending_vnfs = s3[base + 4: base + 4 + num_vnfs]
            pending_per_vnf += pending_vnfs
            avg_remaining_per_sfc.append(s3[base + 1])

        # Helper to map vnf idx -> action indexes
        def alloc_action_idx(v_idx):
            return config.NUM_VNF_TYPES + 1 + v_idx

        def uninstall_action_idx(v_idx):
            return 1 + v_idx

        # Build list of needed vnf indices (pending > 0)
        needed_indices = [i for i in range(num_vnfs) if pending_per_vnf[i] > 0]

        # Policy implementations
        if self.policy == 'random':
            return int(np.random.choice(valid_indices))

        if self.policy == 'first_fit':
            # try allocations in VNF_TYPES order
            for v in range(num_vnfs):
                a = alloc_action_idx(v)
                if a < len(mask) and mask[a]:
                    # prefer allocating needed VNFs, else first alloc
                    if v in needed_indices or all(not (alloc_action_idx(k) < len(mask) and mask[alloc_action_idx(k)]) for k in range(num_vnfs)):
                        return int(a)
            # uninstall first idle if possible
            for v in range(num_vnfs):
                u = uninstall_action_idx(v)
                if u < len(mask) and mask[u]:
                    return int(u)
            return 0

        if self.policy == 'best_fit':
            # prefer types with idle instances in this DC (reuse)
            reuse_candidates = [i for i in needed_indices if idle_counts[i] > 0 and alloc_action_idx(i) < len(mask) and mask[alloc_action_idx(i)]]
            if reuse_candidates:
                return int(alloc_action_idx(reuse_candidates[0]))
            # otherwise pick needed VNF with smallest CPU footprint
            if needed_indices:
                cpu_reqs = [(i, config.VNF_SPECS[config.VNF_TYPES[i]]['cpu']) for i in needed_indices]
                cpu_reqs.sort(key=lambda x: x[1])
                for i, _ in cpu_reqs:
                    a = alloc_action_idx(i)
                    if a < len(mask) and mask[a]:
                        return int(a)
            # uninstall idle if available
            for v in range(num_vnfs):
                u = uninstall_action_idx(v)
                if u < len(mask) and mask[u]:
                    return int(u)
            return 0

        # default: greedy
        # choose most urgent needed VNF: highest pending, tie-breaker lowest avg_remaining across SFCs
        if needed_indices:
            # priority: pending count (desc), then avg_remaining (asc)
            priorities = []
            for v in needed_indices:
                pending = pending_per_vnf[v]
                # compute avg remaining only for sfc types that include this vnf (approx: use global avg)
                avg_rem = np.nanmean([r for r in avg_remaining_per_sfc]) if avg_remaining_per_sfc else 0.0
                priorities.append((pending, -avg_rem, v))
            priorities.sort(key=lambda x: (x[0], x[1]), reverse=True)
            for _, _, v in priorities:
                a = alloc_action_idx(v)
                if a < len(mask) and mask[a]:
                    return int(a)

        # if cannot allocate, try uninstall an idle VNF not needed (first-fit uninstall)
        for v in range(num_vnfs):
            u = uninstall_action_idx(v)
            if u < len(mask) and mask[u]:
                return int(u)

        return 0
