import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym
import torch.nn.functional as F
import matplotlib.pyplot as plt

class QNetwork(nn.Module):
    def __init__(self, state_dim = 4, hidden_dim = 128, action_dim = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self,x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_states, dones):
        # 存一条经验
        self.buffer.append((state, action, reward, next_states, dones))
    
    def sample(self, batch_size: int):
        # 随机抽 batch_size 条经验
        # 返回的每条经验转成 toech.tensor
        # 返回: states, actions, rewards, next_states, dones (tensor)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return(
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),

        )
        

    def __len__(self):
        return len(self.buffer)

def main():
    env = gym.make("LunarLander-v2")
    buffer = ReplayBuffer(capacity=10000)
    q_net = QNetwork(state_dim=8, action_dim=4)
    target_net = QNetwork(state_dim=8, action_dim=4)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr = 1e-3)
    recent_rewards = deque(maxlen=100)

    # 参数
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 10000    # 线性衰减步数
    target_update_freq = 100
    learning_starts = 5000
    train_freq = 4

    episode = 0
    global_step = 0
    episode_returns = []

    while True: # 直到解了才停
        
        episode += 1
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            epsilon = max(epsilon_min, epsilon - (1.0 - epsilon_min) / epsilon_decay)
            # TODO 1: epsilon-greedy 选 action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = q_net(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()

            # TODO 2: env.step(action) -> next_state, reward, terminated, truncated
            next_state, reward, terminated, truncated,_ =env.step(action)
            episode_reward += reward
            # TODO 3: done = terminated or truncated
            done = terminated or truncated

            # TODO 4: push 到 buffer
            buffer.push(state, action, reward, next_state, done)
            
            # TODO 5：if buffer 足够大, 每 train_freq 步学一次
            if len(buffer) > learning_starts:
                if global_step % train_freq == 0:
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                    with torch.no_grad():
                        target_q, _ = target_net(next_states).max(dim = 1)
                        td_target = rewards + gamma * target_q * (1 - dones)
                    
                    current_q = q_net(states).gather(1, actions).squeeze()
                    loss = F.mse_loss(td_target, current_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if global_step % target_update_freq == 0:
                        target_net.load_state_dict(q_net.state_dict())
                    


            state = next_state
            global_step += 1

        
        # 打印每个 episode 的 reward
       
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")
        
        # 判断是否了解了 (连续 100 个 > 200)
        recent_rewards.append(episode_reward)
        episode_returns.append(episode_reward)
        if len(recent_rewards) == 100 and sum(recent_rewards) / 100 > 200:
            print(f"Solved in {episode} episodes!")
            break

    # 保存训练曲线
    plt.plot(episode_returns)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN on LunarLander-v2")
    plt.savefig("C:/Users/Rize/Desktop/Lee/playground/blog/images/dqn_lunarlander_reward.png")
    plt.close()
    print("Curve saved to blog/images/dqn_lunarlander_reward.png")


if __name__ == "__main__":
    main()

