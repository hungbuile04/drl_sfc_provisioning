import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from pathlib import Path

from environment import NetworkEnvironment
from agent import DRLAgent
from SFC_provisioning import SFCProvisioner


def run_train(cfg: DictConfig):
    print("="*60)
    print("TRAINING SFC Provisioning with DRL")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(
        "cuda" if (cfg.device == "auto" and torch.cuda.is_available()) 
        else cfg.device
    )
    print(f"Using device: {device}")

    Path(cfg.output.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.log_dir).mkdir(parents=True, exist_ok=True)

    env = NetworkEnvironment(cfg)
    agent = DRLAgent(cfg, device=device)
    provisioner = SFCProvisioner(env, agent, cfg)

    best_accept = 0

    for update in range(cfg.training.num_updates):
        rewards, accs = [], []

        for _ in range(cfg.training.episodes_per_update):
            r = provisioner.run_episode(training=True)
            m = provisioner.get_metrics()

            rewards.append(r)
            accs.append(m["acceptance_ratio"])

            if len(agent.memory) >= agent.batch_size:
                for _ in range(10):
                    agent.train_step()

        agent.update_target_network()
        agent.decay_epsilon()

        avg_r = np.mean(rewards)
        avg_a = np.mean(accs)

        if (update+1) % cfg.logging.print_interval == 0:
            print(f"[Update {update+1}] Reward={avg_r:.2f}, Accept={avg_a:.2%}, eps={agent.epsilon:.4f}")

        if avg_a > best_accept:
            best_accept = avg_a
            path = Path(cfg.output.model_dir) / f"best_model_{cfg.network.num_dcs}dc.pth"
            agent.save_model(str(path))

    print(f"\nTraining Done! Best Accept={best_accept:.2%}")

def run_test(cfg: DictConfig):
    print("="*60)
    print("TESTING MODEL")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(cfg.output.model_dir) / f"best_model_{cfg.network.num_dcs}dc.pth"

    if not model_path.exists():
        print("Model not found. Please train first.")
        return

    env = NetworkEnvironment(cfg)
    agent = DRLAgent(cfg, device=device)
    agent.load_model(str(model_path))
    agent.epsilon = 0

    provisioner = SFCProvisioner(env, agent, cfg)

    for ep in range(5):
        provisioner.run_episode(training=False)
        m = provisioner.get_metrics()
        print(f"Ep {ep+1}: Accept={m['acceptance_ratio']:.2%}, Delay={m['avg_e2e_delay']:.2f}ms")


def run_reconfig(cfg: DictConfig):
    print("="*60)
    print("RECONFIGURABILITY TEST")
    print("="*60)

    run_train(cfg)

    sizes = [2, 4, 6, 8]
    for s in sizes:
        test_cfg = cfg.copy()
        test_cfg.network.num_dcs = s
        print(f"\nTesting with {s} DCs:")
        run_test(test_cfg)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def entry(cfg: DictConfig):
    mode = cfg.get("mode", "train")

    if mode == "train":
        run_train(cfg)
    elif mode == "test":
        run_test(cfg)
    elif mode == "reconfig":
        run_reconfig(cfg)
    else:
        print(f"Unknown mode={mode}")


if __name__ == "__main__":
    entry()
