"""
Main script with Hydra configuration
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from pathlib import Path

from environment import NetworkEnvironment
from agent import DRLAgent
from SFC_provisioning import SFCProvisioner


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main function with Hydra config"""
    print("="*60)
    print("SFC Provisioning with DRL")
    print("="*60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Set device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"Using device: {device}\n")
    
    # Create directories
    Path(cfg.output.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create environment and agent
    env = NetworkEnvironment(cfg)
    agent = DRLAgent(cfg, device=device)
    provisioner = SFCProvisioner(env, agent, cfg)
    
    # Training
    print(f"Starting training...")
    print(f"  Updates: {cfg.training.num_updates}")
    print(f"  Episodes per update: {cfg.training.episodes_per_update}")
    print(f"  Network: {cfg.network.num_dcs} DCs\n")
    
    best_acceptance = 0
    
    for update in range(cfg.training.num_updates):
        update_rewards = []
        update_acceptance = []
        
        for episode in range(cfg.training.episodes_per_update):
            total_reward = provisioner.run_episode(training=True)
            metrics = provisioner.get_metrics()
            
            update_rewards.append(total_reward)
            update_acceptance.append(metrics['acceptance_ratio'])
            
            # Train agent
            if len(agent.memory) >= agent.batch_size:
                for _ in range(10):
                    agent.train_step()
        
        # Update target network
        agent.update_target_network()
        agent.decay_epsilon()
        
        # Calculate metrics
        avg_reward = np.mean(update_rewards)
        avg_acceptance = np.mean(update_acceptance)
        
        # Log
        if (update + 1) % cfg.logging.print_interval == 0:
            print(f"Update {update + 1}/{cfg.training.num_updates}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Acceptance Ratio: {avg_acceptance:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
        
        # Save best model
        if avg_acceptance > best_acceptance:
            best_acceptance = avg_acceptance
            model_path = Path(cfg.output.model_dir) / f"best_model_{cfg.network.num_dcs}dc.pth"
            agent.save_model(str(model_path))
            if (update + 1) % cfg.logging.save_interval == 0:
                print(f"  ✓ Saved best model (Acc: {best_acceptance:.2%})")
        
        if (update + 1) % cfg.logging.print_interval == 0:
            print()
    
    # Save final model
    final_model_path = Path(cfg.output.model_dir) / f"final_model_{cfg.network.num_dcs}dc.pth"
    agent.save_model(str(final_model_path))
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Acceptance Ratio: {best_acceptance:.2%}")
    print(f"Final model saved to: {final_model_path}")
    print(f"{'='*60}\n")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def test(cfg: DictConfig):
    """Test trained model"""
    print("="*60)
    print("Testing SFC Provisioning Model")
    print("="*60)
    
    # Set device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    
    # Load model
    model_path = Path(cfg.output.model_dir) / f"best_model_{cfg.network.num_dcs}dc.pth"
    
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first using: python main.py")
        return
    
    print(f"\nLoading model: {model_path}")
    print(f"Testing with {cfg.network.num_dcs} DCs\n")
    
    # Create environment and agent
    env = NetworkEnvironment(cfg)
    agent = DRLAgent(cfg, device=device)
    agent.load_model(str(model_path))
    agent.epsilon = 0  # No exploration
    
    provisioner = SFCProvisioner(env, agent, cfg)
    
    # Test
    num_episodes = 10
    results = []
    
    print("Running test episodes...")
    for ep in range(num_episodes):
        provisioner.run_episode(training=False)
        metrics = provisioner.get_metrics()
        results.append(metrics)
        
        print(f"Episode {ep+1:2d}: Acc={metrics['acceptance_ratio']:6.2%}, "
              f"E2E={metrics['avg_e2e_delay']:6.2f}ms, "
              f"Satisfied={metrics['satisfied_count']:3d}, "
              f"Dropped={metrics['dropped_count']:3d}")
    
    # Calculate averages
    print(f"\n{'='*60}")
    print("Average Test Metrics")
    print(f"{'='*60}")
    
    avg_metrics = {}
    for key in results[0].keys():
        avg_metrics[key] = np.mean([r[key] for r in results])
    
    print(f"Acceptance Ratio:     {avg_metrics['acceptance_ratio']:.2%}")
    print(f"Avg E2E Delay:        {avg_metrics['avg_e2e_delay']:.2f} ms")
    print(f"CPU Utilization:      {avg_metrics['cpu_utilization']:.2%}")
    print(f"Storage Utilization:  {avg_metrics['storage_utilization']:.2%}")
    print(f"Avg Satisfied:        {avg_metrics['satisfied_count']:.1f}")
    print(f"Avg Dropped:          {avg_metrics['dropped_count']:.1f}")
    print(f"{'='*60}\n")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def reconfigurability(cfg: DictConfig):
    """Test reconfigurability across different network sizes"""
    print("="*60)
    print("Testing Reconfigurability")
    print("="*60)
    
    # Train on default config
    print(f"\n1. Training on {cfg.network.num_dcs} DCs...")
    train_cfg = cfg.copy()
    main(train_cfg)
    
    # Test on different configurations
    test_configs = [2, 4, 6, 8]
    print(f"\n2. Testing on different network sizes: {test_configs}")
    print("="*60)
    
    results = {}
    
    for num_dcs in test_configs:
        print(f"\nTesting with {num_dcs} DCs...")
        
        test_cfg = cfg.copy()
        test_cfg.network.num_dcs = num_dcs
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load original model
        model_path = Path(cfg.output.model_dir) / f"best_model_{cfg.network.num_dcs}dc.pth"
        
        env = NetworkEnvironment(test_cfg)
        agent = DRLAgent(test_cfg, device=device)
        agent.load_model(str(model_path))
        agent.epsilon = 0
        
        provisioner = SFCProvisioner(env, agent, test_cfg)
        
        # Test episodes
        test_results = []
        for _ in range(5):
            provisioner.run_episode(training=False)
            metrics = provisioner.get_metrics()
            test_results.append(metrics)
        
        # Average
        avg_acc = np.mean([r['acceptance_ratio'] for r in test_results])
        avg_delay = np.mean([r['avg_e2e_delay'] for r in test_results])
        avg_cpu = np.mean([r['cpu_utilization'] for r in test_results])
        
        results[num_dcs] = {
            'acceptance_ratio': avg_acc,
            'avg_e2e_delay': avg_delay,
            'cpu_utilization': avg_cpu
        }
        
        print(f"  Acceptance: {avg_acc:.2%}, E2E: {avg_delay:.2f}ms, CPU: {avg_cpu:.2%}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Reconfigurability Test Summary")
    print(f"{'='*60}")
    print(f"{'DCs':<8} {'Acceptance':<15} {'E2E Delay':<15} {'CPU Usage':<15}")
    print("-"*60)
    for num_dcs, metrics in results.items():
        print(f"{num_dcs:<8} {metrics['acceptance_ratio']:<14.2%} "
              f"{metrics['avg_e2e_delay']:<14.2f} {metrics['cpu_utilization']:<14.2%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    # Simple CLI
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