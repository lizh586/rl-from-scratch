import torch 
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import torch.nn.functional as F
import numpy as np
from collections import deque

from torch.distributions import Categorical


# 环境 : CartPole-v1:
#           observation_space = Box(4,)         4 维连续
#           action_space      = Discrete(2)     2 个离散动作
class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(     # logits
            nn.Linear(4,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,2)
        )
        self.critic = nn.Sequential(    # V(s)
            nn.Linear(4,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )

    def get_value(self,x):
        return self.critic(x)
    
    def get_action_and_value(self,x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()       # 熵
        value = self.critic(x)
        return action, log_prob, entropy, value

    def get_log_prob(self, x, actions):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.log_prob(actions)



def main():
    gamma = 0.99
    clip_eps = 0.2
    roll_out_step = 2048
    recent_rewards = deque(maxlen=10)

    env = gym.make("CartPole-v1")
    agent = Agent()
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    num_iterations = 500 # 重复 rollout + update 次数
    next_state, _ = env.reset()
    for iteration in range(num_iterations):
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        episode_rewards = []    # 跟踪每个 episode 的总回报
        episode_return = 0
        
        
        # Rollout
        for step in range(roll_out_step):
            
            # 1. tensor 化 next_state
            tensor_next_state = torch.tensor(next_state, dtype=torch.float32)
            
            # 2. agent.get_action_and_value(...) → action, log_prob, entropy, value
            action, log_prob, entropy, value = agent.get_action_and_value(tensor_next_state)
            
            # 3. env.step → s_next, r, done, truncated
            s_next, r, done, truncated, _ = env.step(action.item())
            episode_return += r
            # 4. 存入 6 个列表o
            states.append(next_state)
            actions.append(action)
            log_probs.append(log_prob.detach())
            rewards.append(r)
            dones.append(done or truncated)
            values.append(value)

            next_state = s_next
            
            # 5. 若 done 或 truncated，reset 环境
            if done or truncated:
                episode_rewards.append(episode_return)
                episode_return = 0
                next_state, _ = env.reset()
            

        # GAE generalized advantage estimation
        gae = 0
        advantages =[]
        lam = 0.95
        with torch.no_grad():
            next_value = agent.get_value(
                torch.tensor(next_state, dtype=torch.float32)
            ).item()

        for step in reversed(range(roll_out_step)):
            if step == roll_out_step - 1:
                next_val = next_value
            else:
                next_val = values[step+1].item()
            # δ = r + γ * V(s') * (1-done) - V(s)
            delta = rewards[step] + gamma * next_val * (1-dones[step]) - values[step].item()
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae) # 头插
        
        returns = [adv + v.item() for adv, v in zip(advantages, values)]
            
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        log_probs = torch.tensor(log_probs)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize advantages (trick)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update： K epochs x mini-batches
        K = 2
        batch_size = 32
        for epoch in range(K):
            # 随机打乱索引
            indices = np.random.permutation(roll_out_step)
            for start in range(0, roll_out_step, batch_size):
                idx = indices[start:start+batch_size]

                # 前向计算新的 log_prob 和 value
                logits = agent.actor(states[idx])
                probs = Categorical(logits=logits)
                new_log_prob = agent.get_log_prob(states[idx],actions[idx])
                entropy = probs.entropy().mean()
                new_value = agent.get_value(states[idx])
                

                # ratio = exp(new_log_prob - old_log_prob)
                ratio = torch.exp(new_log_prob - log_probs[idx])

                # clipped surrogate loss
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_loss = F.mse_loss(new_value.squeeze(), returns[idx])

                # entropy bonus
                entropy_loss = entropy.mean()

                loss = policy_loss + 0.5*value_loss - 0.01*entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
        # 判断是否了解了 (连续 10 个 > 475)
        recent_rewards.extend(episode_rewards)
        if len(recent_rewards) == 10 and sum(recent_rewards) / 10 > 475:
            print(f"Solved in {iteration} iteration!")
            break

                
        if iteration % 20 == 0 :
            if episode_rewards:
                print(f"Iter {iteration}, mean reward: {np.mean(episode_rewards):.1f}")
            else:
                print(f"Iter {iteration}, no episode finished, max_step=(roll_out_step)")
            episode_rewards = []




   


    


    


if __name__ == "__main__":
    main()