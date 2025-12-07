"""
Utility functions for SFC Provisioning
"""
import numpy as np
import matplotlib.pyplot as plt
from config import *

def calculate_sfc_statistics(satisfied_sfcs, dropped_sfcs):
    """Calculate statistics by SFC type"""
    stats = {}
    
    for sfc_type in SFC_TYPES:
        satisfied = [sfc for sfc in satisfied_sfcs if sfc.sfc_type == sfc_type]
        dropped = [sfc for sfc in dropped_sfcs if sfc.sfc_type == sfc_type]
        total = len(satisfied) + len(dropped)
        
        stats[sfc_type] = {
            'satisfied': len(satisfied),
            'dropped': len(dropped),
            'total': total,
            'acceptance_ratio': len(satisfied) / total if total > 0 else 0
        }
    
    return stats

def plot_sfc_type_comparison(stats):
    """Plot acceptance ratio by SFC type"""
    sfc_types = list(stats.keys())
    acceptance = [stats[st]['acceptance_ratio'] for st in sfc_types]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sfc_types, acceptance, color=['#1f77b4', '#ff7f0e', '#2ca02c', 
                                                   '#d62728', '#9467bd', '#8c564b'])
    plt.xlabel('SFC Type')
    plt.ylabel('Acceptance Ratio')
    plt.title('Acceptance Ratio by SFC Type')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def calculate_resource_consumption(env):
    """Calculate detailed resource consumption"""
    total_cpu = sum(dc.cpu_total for dc in env.dcs)
    total_ram = sum(dc.ram_total for dc in env.dcs)
    total_storage = sum(dc.storage_total for dc in env.dcs)
    
    used_cpu = sum(dc.cpu_total - dc.cpu_available for dc in env.dcs)
    used_ram = sum(dc.ram_total - dc.ram_available for dc in env.dcs)
    used_storage = sum(dc.storage_total - dc.storage_available for dc in env.dcs)
    
    return {
        'cpu_utilization': used_cpu / total_cpu if total_cpu > 0 else 0,
        'ram_utilization': used_ram / total_ram if total_ram > 0 else 0,
        'storage_utilization': used_storage / total_storage if total_storage > 0 else 0,
        'total_cpu': total_cpu,
        'total_ram': total_ram,
        'total_storage': total_storage,
        'used_cpu': used_cpu,
        'used_ram': used_ram,
        'used_storage': used_storage
    }

def calculate_e2e_delay_by_type(satisfied_sfcs, env):
    """Calculate E2E delay for each SFC type"""
    delays = {sfc_type: [] for sfc_type in SFC_TYPES}
    
    for sfc in satisfied_sfcs:
        # Propagation delay
        prop_delay = 0
        for i in range(len(sfc.allocated_dcs) - 1):
            dc_i = sfc.allocated_dcs[i]
            dc_j = sfc.allocated_dcs[i + 1]
            prop_delay += env.distance_matrix[dc_i][dc_j] / (SPEED_OF_LIGHT / 1000)
        
        # Processing delay
        proc_delay = sum([VNF_RESOURCES[vnf]['proc_time'] for vnf in sfc.allocated_vnfs])
        
        total_delay = prop_delay + proc_delay
        delays[sfc.sfc_type].append(total_delay)
    
    # Calculate average for each type
    avg_delays = {}
    for sfc_type, delay_list in delays.items():
        avg_delays[sfc_type] = np.mean(delay_list) if delay_list else 0
    
    return avg_delays

def print_detailed_metrics(metrics, sfc_stats, resource_stats, delay_stats):
    """Print detailed performance metrics"""
    print("\n" + "="*60)
    print("DETAILED PERFORMANCE METRICS")
    print("="*60)
    
    print("\n--- Overall Metrics ---")
    print(f"Acceptance Ratio: {metrics['acceptance_ratio']:.2%}")
    print(f"Average E2E Delay: {metrics['avg_e2e_delay']:.2f} ms")
    print(f"Satisfied SFCs: {metrics['satisfied_count']}")
    print(f"Dropped SFCs: {metrics['dropped_count']}")
    
    print("\n--- Acceptance Ratio by SFC Type ---")
    for sfc_type, stats in sfc_stats.items():
        print(f"{sfc_type:8s}: {stats['acceptance_ratio']:6.2%} "
              f"({stats['satisfied']}/{stats['total']} satisfied)")
    
    print("\n--- Resource Utilization ---")
    print(f"CPU:     {resource_stats['cpu_utilization']:6.2%} "
          f"({resource_stats['used_cpu']:.1f}/{resource_stats['total_cpu']:.1f} GHz)")
    print(f"RAM:     {resource_stats['ram_utilization']:6.2%} "
          f"({resource_stats['used_ram']:.1f}/{resource_stats['total_ram']:.1f} GB)")
    print(f"Storage: {resource_stats['storage_utilization']:6.2%} "
          f"({resource_stats['used_storage']:.1f}/{resource_stats['total_storage']:.1f} GB)")
    
    print("\n--- Average E2E Delay by SFC Type ---")
    for sfc_type, delay in delay_stats.items():
        delay_limit = SFC_CHARACTERISTICS[sfc_type]['delay']
        print(f"{sfc_type:8s}: {delay:6.2f} ms (limit: {delay_limit} ms)")
    
    print("="*60 + "\n")

def save_results_to_file(filename, metrics, sfc_stats, resource_stats, delay_stats):
    """Save results to text file"""
    with open(filename, 'w') as f:
        f.write("SFC Provisioning Results\n")
        f.write("="*60 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Acceptance Ratio: {metrics['acceptance_ratio']:.2%}\n")
        f.write(f"  Avg E2E Delay: {metrics['avg_e2e_delay']:.2f} ms\n")
        f.write(f"  Satisfied: {metrics['satisfied_count']}\n")
        f.write(f"  Dropped: {metrics['dropped_count']}\n\n")
        
        f.write("Acceptance by SFC Type:\n")
        for sfc_type, stats in sfc_stats.items():
            f.write(f"  {sfc_type}: {stats['acceptance_ratio']:.2%}\n")
        f.write("\n")
        
        f.write("Resource Utilization:\n")
        f.write(f"  CPU: {resource_stats['cpu_utilization']:.2%}\n")
        f.write(f"  RAM: {resource_stats['ram_utilization']:.2%}\n")
        f.write(f"  Storage: {resource_stats['storage_utilization']:.2%}\n")

def compare_models(drl_metrics, baseline_metrics):
    """Compare DRL model with baseline"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'DRL':>12} {'Baseline':>12} {'Improvement':>12}")
    print("-"*60)
    
    acc_improvement = (drl_metrics['acceptance_ratio'] - baseline_metrics['acceptance_ratio']) / baseline_metrics['acceptance_ratio'] * 100
    print(f"{'Acceptance Ratio':<30} {drl_metrics['acceptance_ratio']:>11.2%} "
          f"{baseline_metrics['acceptance_ratio']:>11.2%} {acc_improvement:>11.1f}%")
    
    delay_improvement = (baseline_metrics['avg_e2e_delay'] - drl_metrics['avg_e2e_delay']) / baseline_metrics['avg_e2e_delay'] * 100
    print(f"{'Avg E2E Delay (ms)':<30} {drl_metrics['avg_e2e_delay']:>11.2f} "
          f"{baseline_metrics['avg_e2e_delay']:>11.2f} {delay_improvement:>11.1f}%")
    
    cpu_improvement = (baseline_metrics['cpu_utilization'] - drl_metrics['cpu_utilization']) / baseline_metrics['cpu_utilization'] * 100
    print(f"{'CPU Utilization':<30} {drl_metrics['cpu_utilization']:>11.2%} "
          f"{baseline_metrics['cpu_utilization']:>11.2%} {cpu_improvement:>11.1f}%")
    
    print("="*60 + "\n")

def plot_comparison_chart(drl_metrics, baseline_metrics):
    """Create comparison chart between DRL and baseline"""
    metrics_names = ['Acceptance\nRatio', 'E2E Delay\n(normalized)', 'CPU\nUtilization']
    
    # Normalize E2E delay (inverse, lower is better)
    drl_values = [
        drl_metrics['acceptance_ratio'],
        1 - (drl_metrics['avg_e2e_delay'] / max(drl_metrics['avg_e2e_delay'], baseline_metrics['avg_e2e_delay'])),
        1 - drl_metrics['cpu_utilization']  # Lower utilization is better for efficiency
    ]
    
    baseline_values = [
        baseline_metrics['acceptance_ratio'],
        1 - (baseline_metrics['avg_e2e_delay'] / max(drl_metrics['avg_e2e_delay'], baseline_metrics['avg_e2e_delay'])),
        1 - baseline_metrics['cpu_utilization']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, drl_values, width, label='DRL', color='#2ca02c')
    bars2 = ax.bar(x + width/2, baseline_values, width, label='Baseline', color='#d62728')
    
    ax.set_ylabel('Performance (normalized)')
    ax.set_title('DRL vs Baseline Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return plt