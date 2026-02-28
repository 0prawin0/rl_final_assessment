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
# DECENTRALIZED ACTOR (per-store)
# input: local obs for store i (all SKUs) -> action for store i (all SKUs)
# =====================================================
class StoreActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mean = self.net(obs)
        log_std = torch.clamp(self.log_std, -5, 2)
        std = torch.exp(log_std)
        return mean, std


# =====================================================
# CENTRALIZED CRITIC (global)
# input: global state (all stores + all SKUs) -> scalar V(s)
# =====================================================
class CentralCritic(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, global_state):
        return self.v(global_state).squeeze(-1)


# =====================================================
# GAE for CTDE (shared team reward, centralized V(s))
# =====================================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: list[float] length T
    values:  list[float] length T (V(s_t))
    dones:   list[bool] length T
    returns advantages(list), returns(list)
    """
    advantages = []
    gae = 0.0
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
# MAPPO TRAINING (CTDE)
# =====================================================
def train():
    env = QuickMartEnv()
    state = env.reset()

    n_stores = env.num_stores
    n_skus = env.num_skus

    local_obs_dim = 24 * n_skus              # store observation: (425,24) -> 10200
    global_state_dim = 24 * n_skus * n_stores  # 102000
    act_dim = n_skus                          # actions per store (425)

    actors = [StoreActor(local_obs_dim, act_dim).to(device) for _ in range(n_stores)]
    critic = CentralCritic(global_state_dim).to(device)

    actor_opts = [optim.Adam(a.parameters(), lr=3e-5) for a in actors]
    critic_opt = optim.Adam(critic.parameters(), lr=3e-5)

    # PPO / MAPPO hyperparams
    clip_eps = 0.2
    gamma = 0.99
    lam = 0.95
    ppo_epochs = 4
    entropy_coef = 0.01
    critic_coef = 0.5
    max_grad_norm = 1.0

    episodes = 30

    # Practical: cap rollout horizon so CPU training finishes in reasonable time
    rollout_horizon = 60  # you can raise this (e.g., 200) if you want longer rollouts

    reward_curve = []

    for ep in tqdm(range(episodes)):
        state = env.reset()
        done = False
        t = 0

        # ------- rollout buffers (all detached storage) -------
        obs_buf = []          # list[T] of tensor (n_stores, local_obs_dim)
        act_buf = []          # list[T] of tensor (n_stores, act_dim)
        logp_buf = []         # list[T] of tensor (n_stores,)
        rew_buf = []          # list[T] of float (team reward)
        done_buf = []         # list[T] of bool
        val_buf = []          # list[T] of float V(s)

        ep_reward = 0.0

        while (not done) and (t < rollout_horizon):
            # local obs per store
            local_obs = []
            for i in range(n_stores):
                o_i = state[i].reshape(-1).astype(np.float32)  # (local_obs_dim,)
                local_obs.append(o_i)
            local_obs = np.stack(local_obs, axis=0)  # (n_stores, local_obs_dim)

            local_obs_t = torch.tensor(local_obs, dtype=torch.float32, device=device)
            local_obs_t = torch.nan_to_num(local_obs_t, nan=0.0, posinf=1e6, neginf=-1e6)
            local_obs_t = torch.clamp(local_obs_t, -1e6, 1e6) / 500.0

            # global state for critic
            global_state = state.reshape(-1).astype(np.float32)
            global_state_t = torch.tensor(global_state, dtype=torch.float32, device=device)
            global_state_t = torch.nan_to_num(global_state_t, nan=0.0, posinf=1e6, neginf=-1e6)
            global_state_t = torch.clamp(global_state_t, -1e6, 1e6) / 500.0

            with torch.no_grad():
                v_s = critic(global_state_t).item()

            actions = []
            logps = []

            # decentralized execution: each store actor uses ONLY its local obs
            with torch.no_grad():
                for i in range(n_stores):
                    mean, std = actors[i](local_obs_t[i])
                    dist = torch.distributions.Normal(mean, std)
                    a_i = dist.sample()
                    a_i = torch.clamp(a_i, -0.15, 0.15)
                    lp_i = dist.log_prob(a_i).sum()
                    actions.append(a_i)
                    logps.append(lp_i)

            actions_t = torch.stack(actions, dim=0)     # (n_stores, act_dim)
            logps_t = torch.stack(logps, dim=0)         # (n_stores,)

            # env step expects (n_stores, n_skus)
            next_state, reward, done = env.step(actions_t.cpu().numpy())

            # scale reward (helps stability)
            reward_scaled = float(reward) / 1e6
            ep_reward += reward_scaled

            # store rollout (DETACHED)
            obs_buf.append(local_obs_t.detach())
            act_buf.append(actions_t.detach())
            logp_buf.append(logps_t.detach())
            rew_buf.append(reward_scaled)
            done_buf.append(bool(done))
            val_buf.append(float(v_s))

            state = next_state
            t += 1

        # ------- compute GAE / returns (shared) -------
        advs, rets = compute_gae(rew_buf, val_buf, done_buf, gamma=gamma, lam=lam)

        advantages = torch.tensor(advs, dtype=torch.float32, device=device)
        returns = torch.tensor(rets, dtype=torch.float32, device=device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # stack buffers: T x n_stores x dim
        obs_t = torch.stack(obs_buf, dim=0)        # (T, n_stores, local_obs_dim)
        act_t = torch.stack(act_buf, dim=0)        # (T, n_stores, act_dim)
        old_logp_t = torch.stack(logp_buf, dim=0)  # (T, n_stores)

        # also create stacked global states for critic update
        # (rebuild from obs to avoid storing huge state tensor twice)
        # We'll reconstruct approximate global state by flattening all local obs
        # (since local obs were the per-store SKU features already).
        global_states_t = obs_t.reshape(obs_t.shape[0], -1)  # (T, global_state_dim)

        # ------- MAPPO / PPO updates -------
        T = obs_t.shape[0]

        for _ in range(ppo_epochs):
            # ---- critic update ----
            v_pred = critic(global_states_t.reshape(T, -1))
            critic_loss = nn.MSELoss()(v_pred, returns)

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_opt.step()

            # ---- actor updates (each store actor independently, but using shared advantage) ----
            for i in range(n_stores):
                mean, std = actors[i](obs_t[:, i, :])  # (T, act_dim)
                dist = torch.distributions.Normal(mean, std)

                new_logp = dist.log_prob(act_t[:, i, :]).sum(dim=1)  # (T,)
                old_logp = old_logp_t[:, i]                          # (T,)

                ratio = torch.exp(new_logp - old_logp)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                entropy = dist.entropy().mean()

                loss = actor_loss - entropy_coef * entropy

                actor_opts[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actors[i].parameters(), max_grad_norm)
                actor_opts[i].step()

        reward_curve.append(ep_reward)

        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1} | Steps: {T} | Team Reward (scaled): {ep_reward:.4f} | CriticLoss: {critic_loss.item():.4f}")

    # save
    for i, a in enumerate(actors):
        torch.save(a.state_dict(), f"mappo_actor_store_{i}.pth")
    torch.save(critic.state_dict(), "mappo_central_critic.pth")

    print("Saved actors: mappo_actor_store_*.pth")
    print("Saved critic: mappo_central_critic.pth")

    # plot
    plt.plot(reward_curve)
    plt.title("Phase 6 MAPPO (CTDE) Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Team Reward (scaled)")
    plt.show()


if __name__ == "__main__":
    train()