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

## Citation

```
Onsu, M. A., Lohan, P., Kantarci, B., Janulewicz, E., & Slobodrian, S. (2024). 
Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning. 
IEEE Networking Letters.
```
