import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"


# 四种融合模式：[RGB, IR, SmallObject, Robust]
ACTION_WEIGHTS = torch.tensor([
    [0.70, 0.10, 0.10, 0.10],  # RGB 优先
    [0.10, 0.70, 0.10, 0.10],  # 红外优先
    [0.25, 0.25, 0.25, 0.25],  # 均衡融合
    [0.10, 0.10, 0.10, 0.70],  # 鲁棒优先
], dtype=torch.float32)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# 状态包含：3 个场景特征、4 个专家置信度、4 个专家目标数量、1 个尺度统计
state_dim = 12
action_dim = 4
num_samples = 5000
states = torch.randn(num_samples, state_dim)

# 模拟不同场景对应不同最佳融合动作
reward_table = torch.zeros(num_samples, action_dim)

for i in range(num_samples):
    light = states[i, 0]
    weather = states[i, 1]
    density = states[i, 2]

    if light > 0.5:
        best_action = 0       # 光照较好，RGB 优先
    elif weather < -0.5:
        best_action = 1       # 恶劣天气，红外优先
    elif density > 0.5:
        best_action = 2       # 目标密集，均衡融合
    else:
        best_action = 3       # 其他场景，鲁棒专家优先

    reward_table[i] = -0.2
    reward_table[i, best_action] = 1.0


online_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

optimizer = optim.Adam(online_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()

gamma = 0.90
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 3000
batch_size = 64
target_update_interval = 200
global_step = 0

for episode in range(10000):
    index = random.randrange(num_samples)
    state = states[index].numpy()
    state_tensor = torch.tensor(
        state, dtype=torch.float32, device=device
    ).unsqueeze(0)

    epsilon = epsilon_end + (
        epsilon_start - epsilon_end
    ) * np.exp(-global_step / epsilon_decay)

    if random.random() < epsilon:
        action = random.randrange(action_dim)
    else:
        with torch.no_grad():
            action = online_net(state_tensor).argmax(dim=1).item()

    reward = reward_table[index, action].item()
    next_index = (index + 1) % num_samples
    next_state = states[next_index].numpy()
    done = False

    buffer.push(state, action, reward, next_state, done)
    global_step += 1

    if len(buffer) < batch_size:
        continue

    batch_state, batch_action, batch_reward, \
        batch_next_state, batch_done = buffer.sample(batch_size)

    batch_state = batch_state.to(device)
    batch_action = batch_action.to(device)
    batch_reward = batch_reward.to(device)
    batch_next_state = batch_next_state.to(device)
    batch_done = batch_done.to(device)

    current_q = online_net(batch_state).gather(
        1, batch_action.unsqueeze(1)
    ).squeeze(1)

    # Double DQN：在线网络选择动作，目标网络评价动作
    next_action = online_net(batch_next_state).argmax(dim=1)
    next_q = target_net(batch_next_state).gather(
        1, next_action.unsqueeze(1)
    ).squeeze(1)

    target_q = batch_reward + gamma * (
        1.0 - batch_done
    ) * next_q.detach()

    loss = nn.SmoothL1Loss()(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        online_net.parameters(), max_norm=10.0
    )
    optimizer.step()

    if global_step % target_update_interval == 0:
        target_net.load_state_dict(online_net.state_dict())

    if episode % 1000 == 0:
        print(
            f"episode={episode}, "
            f"loss={loss.item():.4f}, "
            f"epsilon={epsilon:.3f}"
        )