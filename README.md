# DRL-based SFC Provisioning

Implementation của bài báo "Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning".

## Cấu trúc Project

```
sfc-drl/
├── conf/
│   └── config.yaml          # Hydra configuration
├── environment.py           # Network environment
├── dqn_model.py             # DQN với attention layer
├── agent.py                 # DRL agent
├── sfc_provisioning.py      # SFC provisioning algorithm
├── main.py                  # Main script
├── pyproject.toml           # UV dependencies
└── README.md
```

## Setup

### 1. Cài đặt UV

```bash
# MacOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone và Setup Project

```bash
# Clone project
git clone https://github.com/hungbuile04/drl_sfc_provisioning.git

# Sync dependencies
uv sync
```

### Training

```bash
# Train với config mặc định
uv run python main.py

# Train với custom config
uv run python main.py network.num_dcs=6 training.num_updates=200

# Train với nhiều thay đổi
uv run python main.py \
  network.num_dcs=8 \
  training.num_updates=150 \
  training.learning_rate=0.0005
```

### Testing

```bash
# Test model đã train
uv run python main.py test

# Test với config khác
uv run python main.py test network.num_dcs=6
```

### Reconfigurability Test

```bash
# Test model trên nhiều network sizes
uv run python main.py reconfig
```

## Configuration

File `conf/config.yaml` chứa toàn bộ cấu hình.

## Citation

```
Onsu, M. A., Lohan, P., Kantarci, B., Janulewicz, E., & Slobodrian, S. (2024). 
Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning. 
IEEE Networking Letters.
```
