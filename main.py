"""
Main training/evaluation script for DRL-based SFC Provisioning
Usage:
    python main.py train    # Train model
    python main.py eval     # Evaluate model
    python main.py demo     # Run demo tests
"""
import sys
import os
import numpy as np
from pathlib import Path

import config
from agent import Agent
from environment import Env
from utils import compare_with_baselines, plot_training_results, run_single_episode, run_experiment_performance, run_experiment_scalability


def train():
    """Train DQN agent"""
    print("\n" + "="*80)
    print("STARTING DRL TRAINING FOR SFC PROVISIONING")
    print("="*80)
    
    # Create directories
    Path('models').mkdir(exist_ok=True)
    Path('fig').mkdir(exist_ok=True)
    
    # Initialize
    env = Env()
    agent = Agent()
    
    # Training state
    epsilon = config.EPSILON_START
    all_rewards = []
    all_ars = []
    best_ar = 0.0
    current_update_rewards = []
    current_update_ars = []
    
    print(f"\nConfiguration:")
    print(f"  - Total Updates: {config.TRAIN_UPDATES}")
    print(f"  - Episodes per Update: {config.EPISODES_PER_UPDATE}")
    print(f"  - Actions per Time Step: {config.ACTIONS_PER_TIME_STEP}")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Memory Size: {config.MEMORY_SIZE}")
    print("="*80)
    
    # Main training loop
    for update_idx in range(1, config.TRAIN_UPDATES + 1):
        print(f"\n[UPDATE {update_idx}/{config.TRAIN_UPDATES}]")
        
        # Run episodes
        for ep_idx in range(config.EPISODES_PER_UPDATE):
            reward, ar, memory_trace = run_single_episode(env, agent, epsilon, training_mode=True)
            
            episode_epsilon = config.EPSILON_MIN + (config.EPSILON_START - config.EPSILON_MIN) * \
                            np.exp(-env.count_step * 3 / config.DECAY_STEP)
            
            # Store transitions
            agent.global_replay_memory.extend(memory_trace)
            current_update_rewards.append(reward)
            current_update_ars.append(ar)
            
            # Update epsilon
            if epsilon > config.EPSILON_MIN:
                epsilon *= config.EPSILON_DECAY
            
            # Progress
            print(f"  Ep {ep_idx+1:3d}: Reward={reward:7.1f}  |  AR={ar:5.1f}%  |  ε={episode_epsilon:.4f} | "
                  f"total step={env.count_step}", end="\r", flush=True)
        
        print()  # New line
        
        # Calculate averages
        avg_reward = np.mean(current_update_rewards)
        avg_ar = np.mean(current_update_ars)
        
        all_rewards.extend(current_update_rewards)
        all_ars.extend(current_update_ars)
        
        # Reset buffers
        current_update_rewards = []
        current_update_ars = []
        
        print(f"  → Avg Reward: {avg_reward:.1f}  |  Avg AR: {avg_ar:.2f}%")
        
        # Save best model
        if avg_ar > best_ar:
            best_ar = avg_ar
            agent.model.save_weights(f'models/best_{config.WEIGHTS_FILE}')
            print(f"  ★ New best model saved! (AR={best_ar:.2f}%)")
        
        # Save checkpoint every 50 updates
        if update_idx % 50 == 0:
            agent.model.save_weights(f'models/checkpoint_{update_idx}_{config.WEIGHTS_FILE}')
            print(f"  💾 Checkpoint saved at update {update_idx}")
    
    # Save final model
    agent.model.save_weights(f'models/{config.WEIGHTS_FILE}')
    
    # Plot results
    plot_training_results(all_rewards, all_ars, 'fig/training_progress.png')
    
    # Final statistics
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Total Episodes: {len(all_ars)}")
    print(f"Final Avg AR (last 100 eps): {np.mean(all_ars[-100:]):.2f}%")
    print(f"Best AR achieved: {best_ar:.2f}%")
    print(f"Final Epsilon: {epsilon:.4f}")
    print("="*80)


def evaluate():
    """Evaluate trained agent"""
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    # Initialize
    env = Env()
    agent = Agent()
    
    # Load weights
    weights_path = f'models/best_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
        weights_path = f'models/{config.WEIGHTS_FILE}'
    
    if os.path.exists(weights_path):
        print(f"\nLoading weights from: {weights_path}")
        try:
            dummy_state, _ = env.reset()
            agent.get_action(dummy_state, 0.0)
            agent.model.load_weights(weights_path)
            print("✓ Weights loaded successfully")
        except Exception as e:
            print(f"✗ Error loading weights: {e}")
            return
    else:
        print(f"\n✗ No weights found at: {weights_path}")
        print("Please run train first.")
        return
    
    # Run basic evaluation
    print(f"\nRunning {config.TEST_EPISODES} test episodes...")
    
    total_rewards = []
    total_ars = []
    
    for ep in range(config.TEST_EPISODES):
        reward, ar = run_single_episode(env, agent, epsilon=config.TEST_EPSILON, training_mode=False)
        total_rewards.append(reward)
        total_ars.append(ar)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}: AR={ar:.2f}%")
    
    print("\n" + "="*80)
    print("BASIC EVALUATION COMPLETED")
    print("="*80)
    print(f"Avg Reward: {np.mean(total_rewards):.2f}")
    print(f"Avg Acceptance Rate: {np.mean(total_ars):.2f}%")
    print("="*80)
    
    # Run detailed experiments
    print(f"\nTest Configuration:")
    print(f"  - Episodes per experiment: {config.TEST_EPISODES}")
    print(f"  - Epsilon (exploration): {config.TEST_EPSILON}")
    print(f"  - DC configurations: {config.TEST_FIG3_DCS}")
    
    # Experiment 1: Performance per SFC type
    run_experiment_performance(env, agent, episodes=config.TEST_EPISODES)
    
    # Experiment 2: Scalability
    run_experiment_scalability(env, agent, episodes=config.TEST_EPISODES)
    
    print("\n" + "="*80)
    print("ALL EVALUATIONS COMPLETED")
    print("="*80)
    print("Check the 'fig/' directory for generated plots.")

def baseline():
    """Evaluate trained agent"""
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    # Initialize
    env = Env()
    agent = Agent()
    
    # Load weights
    weights_path = f'models/best_{config.WEIGHTS_FILE}'
    if not os.path.exists(weights_path):
        weights_path = f'models/{config.WEIGHTS_FILE}'
    
    if os.path.exists(weights_path):
        print(f"\nLoading weights from: {weights_path}")
        try:
            dummy_state, _ = env.reset()
            agent.get_action(dummy_state, 0.0)
            agent.model.load_weights(weights_path)
            print("✓ Weights loaded successfully")
        except Exception as e:
            print(f"✗ Error loading weights: {e}")
            return
    else:
        print(f"\n✗ No weights found at: {weights_path}")
        print("Please run train first.")
        return
    
    # Experiment 3: Baseline Comparison
    print("\n" + "="*80)
    print("Running Baseline Comparison (this may take a few minutes)...")
    print("="*80)
    compare_with_baselines(env, agent, episodes=50)
    
    print("\n" + "="*80)
    print("ALL EVALUATIONS COMPLETED")
    print("="*80)


def demo():
    """Run demo tests"""
    print("\n" + "="*60)
    print("DRL SFC PROVISIONING - DEMO & VALIDATION")
    print("="*60)
    
    print("\n[Test 1: Configuration Validation]")
    print(f"VNF Types: {config.VNF_TYPES}")
    print(f"SFC Types: {config.SFC_TYPES}")
    print(f"Action Space: {config.ACTION_SPACE_SIZE}")
    assert config.ACTION_SPACE_SIZE == 13, "Action space should be 13"
    print("✓ Config validated")
    
    print("\n[Test 2: Environment Setup]")
    env = Env()
    state, _ = env.reset(num_dcs=4)
    print(f"State shape: {[s.shape for s in state]}")
    print("✓ Environment setup OK")
    
    print("\n[Test 3: Quick Episode Test]")
    agent = Agent()
    reward, ar = run_single_episode(env, agent, epsilon=0.5, training_mode=False)
    print(f"Episode reward: {reward:.1f}, AR: {ar:.1f}%")
    print("✓ Episode completed")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == 'train':
            train()
        elif mode == 'eval':
            evaluate()
        elif mode == 'demo':
            demo()
        elif mode == 'baseline':
            baseline()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python main.py [train|eval|demo|baseline]")
    else:
        print("Usage: python main.py [train|eval|demo|baseline]")


if __name__ == '__main__':
    main()