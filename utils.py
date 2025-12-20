"""
Utility functions for training, plotting, and evaluation
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config


def run_single_episode(env, agent, epsilon, training_mode=True):
    """
    Run a single episode
    
    Args:
        env: Environment
        agent: DQN Agent
        epsilon: Exploration rate
        training_mode: If True, store transitions in memory
        
    Returns:
        If training_mode: (total_reward, acceptance_ratio, episode_memory)
        Else: (total_reward, acceptance_ratio)
    """
    state, _ = env.reset()
    action_mask = env._get_valid_actions_mask()
    
    total_reward = 0.0
    done = False
    episode_memory = []
    
    while not done:
        # Dynamic epsilon decay
        epsilon = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * \
                  np.exp(-env.count_step * 3 / config.DECAY_STEP)
        
        # Select action
        action = agent.get_action(state, epsilon, valid_actions_mask=action_mask)
        
        # Take step
        next_state, reward, done, _, info = env.step(action)
        next_action_mask = info.get('action_mask', None)
        
        # Store transition
        if training_mode:
            episode_memory.append((state, action, reward, next_state, done))
        
        # Update
        state = next_state
        total_reward += reward
        action_mask = next_action_mask
        
        # Progress indicator for testing
        env.count_step += 1
        if not training_mode and env.count_step % 500 == 0:
            print(".", end="", flush=True)
        
        # Train network
        if training_mode and env.count_step % config.TARGET_NETWORK_UPDATE == 0:
            print()
            print(f"  Training network...", end=" ", flush=True)
            loss = agent.train()
            agent.update_target_model()
            print(f"Loss={loss:.4f}")
    
    acc_ratio = info.get('acceptance_ratio', 0.0)
    
    if training_mode:
        return total_reward, acc_ratio, episode_memory
    else:
        return total_reward, acc_ratio


def plot_training_results(all_rewards, all_ars, save_path='fig/training_progress.png'):
    """Plot training progress"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        episodes = list(range(1, len(all_ars) + 1))
        
        # Plot 1: Acceptance Ratio
        ax1.plot(episodes, all_ars, alpha=0.3, label='Per Episode', color='blue')
        
        window = 20
        if len(all_ars) >= window:
            moving_avg = np.convolve(all_ars, np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(all_ars) + 1), moving_avg,
                    linewidth=2, color='red', label=f'Moving Avg ({window} eps)')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Acceptance Ratio (%)')
        ax1.set_title('Training Progress: Acceptance Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Total Reward
        ax2.plot(episodes, all_rewards, alpha=0.3, label='Per Episode', color='green')
        
        if len(all_rewards) >= window:
            moving_avg_rew = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
            ax2.plot(range(window, len(all_rewards) + 1), moving_avg_rew,
                    linewidth=2, color='red', label=f'Moving Avg ({window} eps)')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.set_title('Training Progress: Total Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[Graph] Training progress saved to: {save_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"\n[Error] Could not create training plot: {e}")


def plot_exp1_results(sfc_types, acc_ratios, e2e_delays, save_path='fig/result_exp1.png'):
    """Plot Experiment 1 results"""
    if not sfc_types:
        return
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(sfc_types))
        width = 0.35
        
        # Bar: Acceptance Ratio
        ax1.bar(x, acc_ratios, width, label='Acceptance Ratio (%)', color='b', alpha=0.6)
        ax1.set_xlabel('SFC Types')
        ax1.set_ylabel('Acceptance Ratio (%)', color='b')
        ax1.set_ylim(0, 110)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sfc_types, rotation=15)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # Line: E2E Delay
        ax2 = ax1.twinx()
        ax2.plot(x, e2e_delays, color='r', marker='o', linewidth=2, label='E2E Delay (ms)')
        ax2.set_ylabel('Avg E2E Delay (ms)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        max_delay = max(e2e_delays) if e2e_delays else 100
        ax2.set_ylim(0, max_delay * 1.2)
        
        plt.title('Experiment 1: Performance per SFC Type (Fixed 4 DCs)')
        fig.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[Graph] Saved {save_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"\n[Error] Could not plot Exp1: {e}")


def plot_exp2_results(dc_counts, delays, resources, save_path='fig/result_exp2.png'):
    """Plot Experiment 2 results"""
    if not dc_counts:
        return
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Graph 1: E2E Delay
        ax1.plot(dc_counts, delays, 'g-o', linewidth=2)
        ax1.set_title('E2E Delay vs Number of DCs')
        ax1.set_xlabel('Number of DCs')
        ax1.set_ylabel('Avg E2E Delay (ms)')
        ax1.set_xticks(dc_counts)
        ax1.grid(True)
        
        # Graph 2: Resource Consumption
        ax2.bar(dc_counts, resources, color='orange', alpha=0.7, width=0.8)
        ax2.set_title('Avg Resource Consumption vs Number of DCs')
        ax2.set_xlabel('Number of DCs')
        ax2.set_ylabel('Avg CPU Usage (%)')
        ax2.set_xticks(dc_counts)
        ax2.grid(True, axis='y')
        
        plt.suptitle('Experiment 2: Reconfigurability & Robustness')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[Graph] Saved {save_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"\n[Error] Could not plot Exp2: {e}")


def run_experiment_performance(env, agent, episodes=10):
    """Run Experiment 1: Performance Analysis per SFC Type"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1: Performance Analysis per SFC Type (4 DCs)")
    print(f"{'='*80}")
    
    total_completed = []
    total_dropped = []
    
    for ep in range(episodes):
        print(f"\n[Episode {ep+1}/{episodes}] Running", end=" ", flush=True)
        env.reset(num_dcs=4)
        run_single_episode(env, agent, epsilon=config.TEST_EPSILON, training_mode=False)
        
        total_completed.extend(env.sfc_manager.completed_history)
        total_dropped.extend(env.sfc_manager.dropped_history)
        print(" ✓")
    
    print("\nProcessing results...")
    
    # Analyze per SFC type
    sfc_types = config.SFC_TYPES
    acc_ratios = []
    e2e_delays = []
    
    for sfc_type in sfc_types:
        completed = [r for r in total_completed if r.type == sfc_type]
        dropped = [r for r in total_dropped if r.type == sfc_type]
        total = len(completed) + len(dropped)
        
        ar = (len(completed) / total * 100) if total > 0 else 0.0
        avg_delay = np.mean([r.get_total_e2e_delay() for r in completed]) if completed else 0.0
        
        acc_ratios.append(ar)
        e2e_delays.append(avg_delay)
        
        print(f"  {sfc_type:15s}: AR={ar:6.2f}%  |  E2E Delay={avg_delay:6.2f} ms")
    
    plot_exp1_results(sfc_types, acc_ratios, e2e_delays)


def run_experiment_scalability(env, agent, episodes=10):
    """Run Experiment 2: Scalability"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 2: Reconfigurability & Scalability")
    print(f"{'='*80}")
    
    dc_counts = config.TEST_FIG3_DCS
    exp2_delays = []
    exp2_resources = []
    
    for n_dc in dc_counts:
        print(f"\n[Config: {n_dc} DCs]")
        current_completed = []
        cpu_usages = []
        
        for ep in range(episodes):
            print(f"  Episode {ep+1}/{episodes} running", end=" ", flush=True)
            env.reset(num_dcs=n_dc)
            
            state, _ = env._get_obs(), {}
            done = False
            step_count = 0
            
            while not done:
                mask = env._get_valid_actions_mask()
                action = agent.get_action(state, epsilon=config.TEST_EPSILON, valid_actions_mask=mask)
                state, _, done, _, _ = env.step(action)
                
                # Track CPU usage
                total_cap = n_dc * config.DC_CPU_CYCLES
                used_cap = sum(config.DC_CPU_CYCLES - dc.cpu for dc in env.dcs)
                usage_pct = (used_cap / total_cap * 100) if total_cap > 0 else 0
                cpu_usages.append(usage_pct)
                
                step_count += 1
                if step_count % 500 == 0:
                    print(".", end="", flush=True)
            
            current_completed.extend(env.sfc_manager.completed_history)
            print(" ✓")
        
        avg_delay = np.mean([r.get_total_e2e_delay() for r in current_completed]) if current_completed else 0.0
        avg_cpu = np.mean(cpu_usages) if cpu_usages else 0.0
        
        exp2_delays.append(avg_delay)
        exp2_resources.append(avg_cpu)
        
        print(f"  → Avg E2E Delay: {avg_delay:.2f} ms  |  Avg CPU Usage: {avg_cpu:.2f}%")
    
    plot_exp2_results(dc_counts, exp2_delays, exp2_resources)

def compare_with_baselines(env, drl_agent, episodes=50):
    """
    Compare DRL agent with baseline heuristics
    
    Returns:
        dict: {algorithm_name: {'rewards': [...], 'ars': [...]}}
    """
    from environment import BaselineAgent
    
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE HEURISTICS")
    print("="*80)
    
    algorithms = {
        'DRL (Ours)': drl_agent,
        'Random': BaselineAgent('random'),
        'Greedy': BaselineAgent('greedy'),
        'First-Fit': BaselineAgent('first_fit'),
        'Best-Fit': BaselineAgent('best_fit')
    }
    
    results = {}
    
    for algo_name, agent in algorithms.items():
        print(f"\n[Testing {algo_name}]")
        rewards = []
        ars = []
        
        for ep in range(episodes):
            reward, ar = run_single_episode(env, agent, epsilon=0.0, training_mode=False)
            rewards.append(reward)
            ars.append(ar)
            
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1}/{episodes}: AR={ar:.2f}%", end="\r")
        
        avg_reward = np.mean(rewards)
        avg_ar = np.mean(ars)
        std_ar = np.std(ars)
        
        results[algo_name] = {
            'rewards': rewards,
            'ars': ars,
            'avg_reward': avg_reward,
            'avg_ar': avg_ar,
            'std_ar': std_ar
        }
        
        print(f"  {algo_name:15s}: Avg AR={avg_ar:5.2f}% (±{std_ar:.2f}%), Avg Reward={avg_reward:7.2f}")
    
    # Plot comparison
    plot_baseline_comparison(results, save_path='fig/baseline_comparison.png')
    
    return results


def plot_baseline_comparison(results, save_path='fig/baseline_comparison.png'):
    """Plot comparison with baseline algorithms"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        algorithms = list(results.keys())
        avg_ars = [results[algo]['avg_ar'] for algo in algorithms]
        std_ars = [results[algo]['std_ar'] for algo in algorithms]
        avg_rewards = [results[algo]['avg_reward'] for algo in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.6
        
        # Plot 1: Acceptance Rate Comparison
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
        bars1 = ax1.bar(x, avg_ars, width, yerr=std_ars, capsize=5, 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax1.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Acceptance Rate (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Acceptance Rate Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=15, ha='right')
        ax1.set_ylim(0, max(avg_ars) * 1.15)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val, std) in enumerate(zip(bars1, avg_ars, std_ars)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 2: Average Reward Comparison
        bars2 = ax2.bar(x, avg_rewards, width, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=1.5)
        
        ax2.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
        ax2.set_title('Average Reward Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms, rotation=15, ha='right')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, avg_rewards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[Graph] Baseline comparison saved to: {save_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"\n[Error] Could not create baseline comparison plot: {e}")


def plot_baseline_comparison_detailed(results, save_path='fig/baseline_comparison_detailed.png'):
    """Plot detailed comparison with distribution"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        algorithms = list(results.keys())
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
        
        for idx, (algo, ax) in enumerate(zip(algorithms, axes.flatten())):
            if idx >= len(algorithms):
                ax.axis('off')
                continue
            
            ars = results[algo]['ars']
            avg_ar = results[algo]['avg_ar']
            
            # Histogram
            ax.hist(ars, bins=20, alpha=0.7, color=colors[idx], edgecolor='black')
            ax.axvline(avg_ar, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean={avg_ar:.2f}%')
            
            ax.set_xlabel('Acceptance Rate (%)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{algo}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        if len(algorithms) < 6:
            axes.flatten()[len(algorithms)].axis('off')
        
        plt.suptitle('Acceptance Rate Distribution per Algorithm', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Graph] Detailed comparison saved to: {save_path}")
        plt.close(fig)
        
    except Exception as e:
        print(f"[Error] Could not create detailed comparison plot: {e}")