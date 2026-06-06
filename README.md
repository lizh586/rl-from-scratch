# RL from Scratch

从零手写 DQN / PPO / SAC / REINFORCE，不依赖 RL 框架，只用 PyTorch + Gymnasium。CartPole / LunarLander / Pendulum 全部 solved。

## Benchmark

| 算法 | 环境 | 收敛指标 | 结果 | 备注 |
|------|------|---------|------|------|
| DQN | CartPole-v1 | 10-ep avg >= 495 | 728 episodes | from scratch |
| DQN | LunarLander-v2 | 100-ep avg >= 200 | 938 episodes | 前 380ep 无信号，突破后快速拉升 |
| PPO | CartPole-v1 | 10-ep avg >= 495 | 62 iterations | 7 轮调参后稳定 |
| PPO | LunarLander-v2 | 100-ep avg >= 200 | 405 iterations | 曲线平滑攀升，无负区 |
| SAC | Pendulum-v1 | eval return | ~ -120 | lr 3e-4 + learning_starts 5000 + clip_grad |
| REINFORCE | CartPole-v1 | 10-ep avg >= 495 | ~800 episodes | MC policy gradient |
| REINFORCE + Baseline | CartPole-v1 | 10-ep avg >= 495 | ~400 episodes | value network 降方差 |

## 收敛曲线

### DQN on LunarLander-v2
![DQN LunarLander](images/dqn_lunarlander_reward.png)

### PPO on LunarLander-v2
![PPO LunarLander](images/ppo_lunarlander_reward.png)

## 安装 & 运行

```bash
git clone https://github.com/lizh586/rl-from-scratch.git
cd rl-from-scratch
pip install -r requirements.txt
```

每个算法独立运行：

```bash
# DQN on CartPole
python dqn/dqn_handcraft.py

# PPO on CartPole
python ppo/ppo_handcraft.py

# SAC on Pendulum
python sac/sac_handcraft.py

# REINFORCE on CartPole
python reinforce/reinforce_cartpole.py
```

## 目录结构

```
rl-from-scratch/
├── README.md
├── requirements.txt
├── images/
│   ├── dqn_lunarlander_reward.png
│   └── ppo_lunarlander_reward.png
├── dqn/
│   ├── dqn_handcraft.py            # CartPole
│   └── dqn_handcraft_lunar.py      # LunarLander
├── ppo/
│   ├── ppo_handcraft.py            # CartPole
│   └── ppo_handcraft_lunar.py      # LunarLander
├── sac/
│   └── sac_handcraft.py            # Pendulum
└── reinforce/
    ├── reinforce_cartpole.py
    └── reinforce_baseline_cartpole.py
```

## 踩坑记录

- **SAC Q 网络 lr 过大**：qf1_loss 频繁 spike（500-900），eval return 大幅振荡。lr 从 1e-3 降到 3e-4 + gradient clipping 后稳定。
- **DQN epsilon 衰减过快**：每 episode 衰减导致 epsilon 过早趋近 0.01，探索不足。改为每 step 衰减。
- **learning_starts 不够**：buffer 没填满就开始训练，数据多样性不足。延迟到 5000 steps 后启动训练。

## 技术博客

详细讲解见：[从零手写 DQN / PPO / SAC：三个强化学习算法的完整时间](https://example.com)（待发布）
