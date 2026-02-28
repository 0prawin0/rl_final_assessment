import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from mdp_formulation import QuickMartEnv

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# ACTOR-CRITIC NETWORK (PPO)
# =====================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

        # Diagonal Gaussian policy: shared log_std for all actions
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.shared(state)

        mean = self.actor(x)
        value = self.critic(x).squeeze(-1)

        log_std = torch.clamp(self.log_std, -5, 2)
        std = torch.exp(log_std)

        return mean, std, value


# =====================================================
# GAE (Generalized Advantage Estimation)
# =====================================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: list[float] length T
    values:  list[float] length T (V(s_t))
    dones:   list[bool]  length T
    """
    advantages = []
    gae = 0.0

    # bootstrap value for terminal
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]

    returns = [adv + v for adv, v in zip(advantages, values)]
    return advantages, returns


# =====================================================
# PPO TRAINING
# =====================================================
def train():
    env = QuickMartEnv()
    state = env.reset()

    state_dim = int(np.prod(state.shape))
    action_dim = env.num_stores * env.num_skus

    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    clip_eps = 0.2
    gamma = 0.99
    lam = 0.95
    ppo_epochs = 5

    episodes = 50
    reward_history = []

    for episode in tqdm(range(episodes)):
        # -------- Rollout storage (ALL DETACHED) --------
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        values = []
        dones = []

        state = env.reset()
        done = False

        while not done:
            state_flat = torch.tensor(
                state.reshape(-1),
                dtype=torch.float32,
                device=device
            )

            # stability preprocessing
            state_flat = torch.nan_to_num(state_flat, nan=0.0, posinf=1e6, neginf=-1e6)
            state_flat = torch.clamp(state_flat, -1e6, 1e6)
            state_flat = state_flat / 500.0

            with torch.no_grad():
                mean, std, value = model(state_flat)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()

            action_clipped = torch.clamp(action, -0.15, 0.15)

            next_state, reward, done = env.step(
                action_clipped.cpu().numpy().reshape(env.num_stores, env.num_skus)
            )

            # scale reward
            reward = reward / 1e6

            # store DETACHED rollout data
            states.append(state_flat.detach())
            actions.append(action.detach())
            old_log_probs.append(log_prob.detach())
            rewards.append(float(reward))
            values.append(float(value.item()))
            dones.append(bool(done))

            state = next_state

        # -------- Compute GAE / Returns --------
        advs, rets = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)

        advantages = torch.tensor(advs, dtype=torch.float32, device=device)
        returns = torch.tensor(rets, dtype=torch.float32, device=device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # batch tensors
        states_t = torch.stack(states)                  # already detached
        actions_t = torch.stack(actions)                # already detached
        old_log_probs_t = torch.stack(old_log_probs)    # already detached

        # -------- PPO Updates --------
        for _ in range(ppo_epochs):
            mean, std, value_pred = model(states_t)
            dist = torch.distributions.Normal(mean, std)

            new_log_probs = dist.log_prob(actions_t).sum(dim=1)

            ratio = torch.exp(new_log_probs - old_log_probs_t)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(value_pred, returns)

            entropy = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        episode_reward = sum(rewards)
        reward_history.append(episode_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Reward: {episode_reward:.4f}")

    torch.save(model.state_dict(), "ppo_phase3.pth")
    print("Saved: ppo_phase3.pth")

    plt.plot(reward_history)
    plt.title("Phase 3 PPO Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (scaled)")
    plt.show()


if __name__ == "__main__":
    train()