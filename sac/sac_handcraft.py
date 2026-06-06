import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, buffer_size=1000, obs_dim=3, act_dim=1):
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.reward = np.zeros((buffer_size, 1), dtype=np.float32)
        self.done = np.zeros((buffer_size, 1), dtype=np.float32)
        self.pos = 0
        self.full = False
        self.buffer_size = buffer_size
        

    def push(self, obs, next_obs, actions, rewards, dones):
        n = len(obs)        
        for i in range(n):
            idx = self.pos
            self.obs[idx] = obs[i]
            self.next_obs[idx] = next_obs[i]
            self.action[idx] = actions[i]
            self.reward[idx] = rewards[i]
            self.done[idx] = dones[i]
            self.pos = (self.pos+1) % self.buffer_size
        if self.pos == 0:
            self.full = True
        

    def sample(self, batch_size):
        limit = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, limit, size=batch_size)
        obs = torch.tensor(self.obs[indices])
        actions = torch.tensor(self.action[indices])
        rewards = torch.tensor(self.reward[indices])
        dones = torch.tensor(self.done[indices])
        next_obs = torch.tensor(self.next_obs[indices])
        
        return obs, actions, rewards, dones, next_obs

LOG_STD_MIN = -5
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self, obs_dim=3, act_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, act_dim)
        self.fc_logstd = nn.Linear(128, act_dim)
        self.register_buffer("action_scale", torch.tensor(2.0))
        self.register_buffer("action_bias", torch.tensor(0.0))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean,std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob
    



class QNetwork(nn.Module):
    def __init__(self, obs_dim=3, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))
    

if __name__ =="__main__":
    env = gym.make("Pendulum-v1")

    obs_dim = env.observation_space.shape[0]    # 3
    act_dim = env.action_space.shape[0]         # 1
    actor = Actor(obs_dim, act_dim)
    qf1, qf2 = QNetwork(obs_dim, act_dim), QNetwork(obs_dim, act_dim)
    qf1_target, qf2_target = QNetwork(obs_dim, act_dim), QNetwork(obs_dim, act_dim)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=3e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr = 3e-4)

    buffer = ReplayBuffer(buffer_size=100000)
    alpha = 0.2
    gamma = 0.99
    tau = 0.005
    learning_starts = 5000
    policy_frequency = 2
    total_step = 50000
    batch_size = 128

    obs, _ = env.reset()

    for step in range(50000):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            action, _ = actor.get_action(torch.FloatTensor(obs).unsqueeze(0))
            action = action.squeeze(0).detach().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # push to buffer (注意加 batch 维)
        buffer.push(obs[None,:], next_obs[None,:],
              np.array([action], dtype=np.float32),
              np.array([[reward]], dtype=np.float32),
              np.array([[done]], dtype=np.float32))

        obs = next_obs

        if step >= learning_starts:
            # 1.samlple batch 
            batch_obs, batch_act, batch_rew, batch_done, batch_next_obs = buffer.sample(batch_size)

            # 2. TD target: r + γ(1-d)[min(qf1_target, qf2_target) - α·log_prob]
            with torch.no_grad():
                next_action, next_log_prob = actor.get_action(batch_next_obs)
                qf1_next = qf1_target(batch_next_obs, next_action)
                qf2_next = qf2_target(batch_next_obs, next_action)
                min_q_next = torch.min(qf1_next, qf2_next) - alpha * next_log_prob
                td_target = batch_rew + gamma * (1 - batch_done) * min_q_next

            # 3. qf1_loss + qf2_loss → backward
            qf1_pred = qf1(batch_obs, batch_act)
            qf2_pred = qf2(batch_obs, batch_act)
            qf1_loss = F.mse_loss(qf1_pred, td_target)
            qf2_loss = F.mse_loss(qf2_pred, td_target)

            q_optimizer.zero_grad()
            (qf1_loss + qf2_loss).backward()
            torch.nn.utils.clip_grad_norm_(qf1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(qf2.parameters(), max_norm=1.0)
            q_optimizer.step()

            # 4. (delayed) actor_loss → backward
            if step % policy_frequency == 0:
                pi, log_pi = actor.get_action(batch_obs)
                qf1_pi = qf1(batch_obs, pi)
                qf2_pi = qf2(batch_obs, pi)
                min_q_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = (alpha * log_pi - min_q_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            # 5. soft update: qf1_target, qf2_target
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)    
            
        if step % 100 == 0 and step >= learning_starts:
            print(f"step={step}, qf1_loss={qf1_loss.item():.4f}, actor_loss={actor_loss.item():.4f}")
        if step % 5000 == 0:
            eval_obs, _ = env.reset()
            eval_return = 0
            for _ in range (200):
                with torch.no_grad():
                    eval_action, _ = actor.get_action(torch.FloatTensor(eval_obs).unsqueeze(0))
                eval_action = eval_action.squeeze(0).detach().numpy()
                eval_obs, eval_reward, eval_terminated, eval_truncated, _ = env.step(eval_action)
                eval_return += eval_reward
                if eval_terminated or eval_truncated:
                    break
            print(f"step={step}, eval_return={eval_return:.1f}")
        
        if done:
            obs, _ = env.reset()
