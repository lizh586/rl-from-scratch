import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128, logits_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),  
            nn.Linear(hidden_dim, logits_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    def forward(self, x):
        return self.net(x)
    

def sample_action(state_tensor, policy_net):
    logits = policy_net(state_tensor)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return log_prob, action

def main():
    gamma = 0.9

    env = gym.make("CartPole-v1")
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()
    optimizer_p = optim.Adam(policy_net.parameters(), lr=1e-3)
    optimizer_v = optim.Adam(value_net.parameters(), lr=1e-3)
    for episode in range(1000):
        states=[]
        actions=[]
        rewards=[]
        returns=[]
        log_probs=[]
        s, _ = env.reset()
        while True:
            log_prob, action = sample_action(torch.tensor(s, dtype=torch.float32), policy_net)
            s_next, r, done, truncated, _ = env.step(action.item())
            states.append(s)
            actions.append(action)
            rewards.append(r)
            log_probs.append(log_prob)
            s = s_next
            if done or truncated:
                break
        
        G = 0
        for t in reversed(rewards):
            G = t + gamma*G
            returns.insert(0,G)
        V_s = value_net(torch.tensor(states, dtype=torch.float32)).squeeze()
        loss_v = F.mse_loss(V_s, torch.tensor(returns, dtype=torch.float32))
        optimizer_v.zero_grad()
        loss_v.backward()
        optimizer_v.step()

        returns_tensor = torch.tensor(returns, dtype=torch.float32) - V_s.detach()
        log_probs_tensor = torch.stack(log_probs)


        loss_p = - (returns_tensor * log_probs_tensor).sum()

        optimizer_p.zero_grad()
        loss_p.backward()
        optimizer_p.step()

        total_reward = sum(rewards)
        if episode % 20 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    
if __name__ == "__main__":
    main()
