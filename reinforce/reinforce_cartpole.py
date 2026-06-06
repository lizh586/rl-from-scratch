import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128, logits_dim=2):
        super().__init__()
        self.net = nn.Sequential (
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, logits_dim)
        )
    def forward(self, x):
        return self.net(x)

def sample_action(state_tensor, policy_net):
    logits = policy_net(state_tensor)       # 输出 logits
    dist = Categorical(logits=logits)       # 内部做了 softmax
    action = dist.sample()                  # 根据概率采样
    log_prob = dist.log_prob(action)        # log pai(a|s)
    return log_prob, action

def main():
    gamma = 0.9

    env = gym.make("CartPole-v1")
    policy_net = PolicyNetwork()
    optimizer = optim.Adam(policy_net.parameters(), lr = 1e-3)
    for episode in range(1000):
        actions = []        # 存每一步的 action
        rewards = []        # 存每一步的 reward
        returns = []        # 存G_t
        log_probs = []      # 存log pi(a|s)
        # 1. 跑完整个 episode
        s, _  = env.reset() 
        while True:
            log_prob, action = sample_action(torch.tensor(s, dtype=torch.float32),policy_net)      # 从策略采样
            s_next, r, done, truncated, _ = env.step(action.item())
            actions.append(action)
            rewards.append(r)
            s = s_next
            log_probs.append(log_prob)
            if done or truncated:
                break
        

        # 2. 从后往前算 G_t
        G = 0
        for t in reversed(rewards):
            G = t + gamma * G
            returns.insert(0,G)

        # 3. 算 loss
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        log_probs_tensor = torch.stack(log_probs)   # stack 把标量列表变成 tensor

        loss = -(returns_tensor * log_probs_tensor).sum()

        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5. 看结果
        total_reward = sum(rewards)
        if episode % 20 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    
if __name__ == "__main__":
    main()
