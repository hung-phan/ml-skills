---
name: reinforcement-learning
description: RL algorithms from tabular Q-learning to deep RL (DQN, PPO, SAC, A2C), gymnasium environments, experience replay, reward shaping, multi-agent RL, GAE, stable-baselines3, and CleanRL patterns. Use when training agents in interactive environments, designing reward functions, or selecting an RL algorithm for a control or game task.
---

## Why This Exists

**Problem**: Some tasks have no fixed "correct answer" dataset — the agent must learn through trial and error by interacting with an environment, discovering which sequences of actions maximize long-term reward (game playing, robotics, resource allocation, trading).

**Key insight**: Define a reward signal, let the agent explore, and use temporal difference learning (bootstrap current estimates from future estimates) or policy gradients (directly optimize the probability of high-reward action sequences) to learn optimal behavior without explicit supervision.

**Reach for this when**: You have a sequential decision problem where actions affect future states, no labeled dataset of optimal actions exists, and you can simulate the environment cheaply (games, simulators). If you have expert demonstrations, prefer imitation learning first; if the reward is hard to define, prefer reward modeling or RLHF.


# Reinforcement Learning

## Core Concepts

```
Agent → Action → Environment → (Next State, Reward) → Agent
```

**MDP**: (S, A, P, R, γ) — States, Actions, Transition Probabilities, Reward Function, Discount Factor.

## Algorithm Taxonomy

### Model-Free vs Model-Based

| Approach | Learns | Examples | Use When |
|----------|--------|----------|----------|
| Model-Free (Value) | Q(s,a) directly | DQN, Double DQN | Discrete actions, sample-efficient not critical |
| Model-Free (Policy) | π(a|s) directly | REINFORCE, PPO, SAC | Continuous actions, high-dim action spaces |
| Model-Based | Transition model T(s'|s,a) | Dreamer, MuZero, MBPO | Sample efficiency critical, env is expensive |

### Algorithm Selection Guide

```
Is action space discrete?
├── Yes → Is environment simple (< 10K states)?
│   ├── Yes → Tabular Q-Learning
│   └── No → DQN / Rainbow DQN
└── No (continuous) → Do you need sample efficiency?
    ├── Yes → SAC (off-policy, entropy-regularized)
    └── No → PPO (on-policy, stable, parallelizable)

Multi-agent? → MAPPO, QMIX, or independent PPO with shared rewards
Sparse rewards? → Reward shaping + Hindsight Experience Replay (HER)
```

## Q-Learning (Tabular)

```python
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, epsilon = 0.1, 0.99, 0.1

for episode in range(10_000):
    state, _ = env.reset()
    done = False
    while not done:
        # ε-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Bellman update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state
```

## DQN (Deep Q-Network)

Key innovations: experience replay buffer, target network (soft/hard update), ε-greedy exploration.

```python
import torch
import torch.nn as nn
from collections import deque
import random
import gymnasium as gym
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
        )
    
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

# Training loop (CleanRL style)
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

q_net = QNetwork(obs_dim, act_dim)
target_net = QNetwork(obs_dim, act_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()
gamma = 0.99
batch_size = 64
target_update_freq = 500
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

state, _ = env.reset()
for step in range(50_000):
    # ε-greedy
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            action = q_net(torch.FloatTensor(state)).argmax().item()
    
    next_state, reward, terminated, truncated, _ = env.step(action)
    buffer.push(state, action, reward, next_state, float(terminated))
    state = next_state if not (terminated or truncated) else env.reset()[0]
    
    if len(buffer.buffer) >= batch_size:
        s, a, r, ns, d = buffer.sample(batch_size)
        current_q = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_next_q = target_net(ns).max(1)[0]
            target_q = r + gamma * max_next_q * (1 - d)
        
        loss = nn.functional.mse_loss(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if step % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

**Double DQN**: Use online net to SELECT action, target net to EVALUATE — reduces overestimation:
```python
next_actions = q_net(ns).argmax(1, keepdim=True)
max_next_q = target_net(ns).gather(1, next_actions).squeeze()
```

## Policy Gradient (REINFORCE)

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, act_dim),
        )
    
    def forward(self, x):
        return Categorical(logits=self.net(x))

env = gym.make("CartPole-v1")
policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
gamma = 0.99

for episode in range(1000):
    log_probs, rewards = [], []
    state, _ = env.reset()
    done = False
    
    while not done:
        dist = policy(torch.FloatTensor(state))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated
    
    # Compute discounted returns (rewards-to-go)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # baseline
    
    loss = -sum(lp * G for lp, G in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## PPO (Proximal Policy Optimization)

The workhorse of modern RL. Clips policy updates to prevent destructive large steps.

```python
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
    
    def get_action_and_value(self, x):
        mean = self.actor_mean(x)
        std = self.actor_logstd.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(x).squeeze()
    
    def get_value(self, x):
        return self.critic(x).squeeze()

# PPO update (single epoch shown)
def ppo_update(agent, optimizer, states, actions, old_log_probs, returns, advantages,
               clip_eps=0.2, epochs=4, batch_size=64):
    for _ in range(epochs):
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), batch_size):
            idx = indices[start:start + batch_size]
            s = states[idx]
            a = actions[idx]
            
            mean = agent.actor_mean(s)
            std = agent.actor_logstd.exp()
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(a).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            values = agent.get_value(s)
            
            # Clipped surrogate objective
            ratio = (new_log_probs - old_log_probs[idx]).exp()
            adv = advantages[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, returns[idx])
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
```

## SAC (Soft Actor-Critic)

Off-policy, entropy-regularized. Best for continuous control when sample efficiency matters.

```python
# Key difference from PPO: maximizes expected reward + entropy
# J(π) = E[Σ r(s,a) + α * H(π(·|s))]

# stable-baselines3 usage:
from stable_baselines3 import SAC
import gymnasium as gym

env = gym.make("Pendulum-v1")
model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000,
            batch_size=256, tau=0.005, gamma=0.99, verbose=1)
model.learn(total_timesteps=100_000)
```

## Stable-Baselines3 Patterns

```python
from stable_baselines3 import PPO, SAC, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym

# Vectorized environments (parallel rollouts)
env = make_vec_env("HalfCheetah-v4", n_envs=4, vec_env_cls=SubprocVecEnv)

# PPO with custom hyperparams
model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,          # rollout length per env
    batch_size=64,         # minibatch size for updates
    n_epochs=10,           # PPO epochs per rollout
    gamma=0.99,
    gae_lambda=0.95,       # GAE λ for advantage estimation
    clip_range=0.2,
    ent_coef=0.01,         # entropy bonus
    verbose=1,
)

# Callbacks
eval_env = gym.make("HalfCheetah-v4")
callbacks = [
    EvalCallback(eval_env, eval_freq=10_000, best_model_save_path="./best/"),
    CheckpointCallback(save_freq=50_000, save_path="./checkpoints/"),
]
model.learn(total_timesteps=1_000_000, callback=callbacks)

# Save / load
model.save("ppo_cheetah")
loaded = PPO.load("ppo_cheetah", env=env)

# Custom network architecture
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),  # separate actor/critic
    activation_fn=torch.nn.ReLU,
)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
```

## Gymnasium Environment Setup

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    """Minimal custom environment template."""
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # or spaces.Box for continuous
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(-0.5, 0.5, size=(4,)).astype(np.float32)
        return self.state, {}  # obs, info
    
    def step(self, action):
        self.state = np.clip(self.state + 0.1 * (action - 1), -1, 1)
        reward = -np.sum(self.state ** 2)
        terminated = bool(np.any(np.abs(self.state) > 0.95))
        truncated = False
        return self.state, reward, terminated, truncated, {}

# Register for gym.make()
gym.register(id="Custom-v0", entry_point="my_module:CustomEnv", max_episode_steps=200)
```

## GAE (Generalized Advantage Estimation)

```python
def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
    returns = advantages + values
    return advantages, returns
```

## When to Use What

| Scenario | Algorithm | Why |
|----------|-----------|-----|
| Discrete actions, single env | DQN / Rainbow | Sample efficient, off-policy |
| Continuous control, need stability | PPO | On-policy, forgiving hyperparams |
| Continuous control, need sample efficiency | SAC | Off-policy, auto-tuned entropy |
| Simple/fast baseline | A2C | Quick to run, decent results |
| Robotics with sparse goals | SAC + HER | Off-policy + goal relabeling |
| Multi-agent cooperative | MAPPO | Scales, centralized critic |
| Atari / image observations | PPO + CNN policy | Proven, scalable |
| You want to understand the code | CleanRL | Single-file, no abstractions |
| You want quick experiments | stable-baselines3 | Batteries included, good defaults |
| Custom environment | gymnasium.Env subclass | Standard interface, all algos work |

## Hyperparameter Quick Reference

### PPO
- `learning_rate`: 3e-4 (default), anneal linearly for stability
- `n_steps`: 2048 (per env), lower = more bias, higher = more variance
- `batch_size`: 64 (minibatch), must divide n_steps * n_envs
- `n_epochs`: 4-10
- `clip_range`: 0.1-0.3 (0.2 standard)
- `gae_lambda`: 0.95
- `ent_coef`: 0.0-0.01 (higher = more exploration)

### SAC
- `learning_rate`: 3e-4
- `buffer_size`: 1M
- `batch_size`: 256
- `tau`: 0.005 (soft target update)
- `alpha`: auto-tuned (target entropy = -dim(A))

### DQN
- `learning_rate`: 1e-4
- `buffer_size`: 100K-1M
- `batch_size`: 32-128
- `target_update_interval`: 1000-10000 steps
- `exploration_fraction`: 0.1 (of total steps)
- `epsilon_final`: 0.01-0.05

---

## References

- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) — PPO algorithm
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io) — Reliable RL implementations
- [Gymnasium](https://gymnasium.farama.org) — Standard RL environment interface
