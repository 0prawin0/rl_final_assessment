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
# POLICY NETWORK 
# =====================================================
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
)

        # Learnable log_std (clamped later)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.model(state)

        # Clamp std for stability
        log_std = torch.clamp(self.log_std, -5, 2)
        std = torch.exp(log_std)

        return mean, std


# =====================================================
# VALUE NETWORK (Baseline)
# =====================================================
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        # Value
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.model(state)


# =====================================================
# COMPUTE RETURNS
# =====================================================
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    # Normalize returns (VERY IMPORTANT)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


# =====================================================
# TRAINING LOOP
# =====================================================
def train():

    env = QuickMartEnv()
    state = env.reset()

    state_dim = np.prod(state.shape)
    action_dim = env.num_stores * env.num_skus

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)

    policy_optimizer = optim.Adam(policy.parameters(), lr=3e-5)
    value_optimizer = optim.Adam(value_net.parameters(), lr=3e-5)

    episodes = 50
    gamma = 0.99

    all_rewards = []

    for episode in tqdm(range(episodes)):

        state = env.reset()

        log_probs = []
        rewards = []
        values = []

        done = False

        while not done:

            state_flat = torch.tensor(
                state.reshape(-1),
                dtype=torch.float32,
                device=device
            )

            state_flat = torch.nan_to_num(state_flat, nan=0.0, posinf=1e6, neginf=-1e6)
            state_flat = torch.clamp(state_flat, -1e6, 1e6)
            state_flat = state_flat / 500.0 

            mean, std = policy(state_flat)
            dist = torch.distributions.Normal(mean, std)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            action_clipped = torch.clamp(action, -0.15, 0.15)

            action_np = action_clipped.detach().cpu().numpy()
            action_np = action_np.reshape(env.num_stores, env.num_skus)

            next_state, reward, done = env.step(action_np)

            # Scale reward (prevents explosion)
            reward = reward / 1e6

            value = value_net(state_flat)

            log_probs.append(log_prob.detach())
            rewards.append(reward)
            values.append(value)

            state = next_state

        # Compute normalized returns
        returns = compute_returns(rewards, gamma)

        values = torch.cat(values).squeeze()
        advantages = returns - values.detach()

        # Policy Loss
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)

        policy_loss = torch.stack(policy_loss).sum()

        # Value Loss
        value_loss = nn.MSELoss()(values, returns)

        # Update Policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        policy_optimizer.step()

        # Update Value Network
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        value_optimizer.step()

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")

        # Save checkpoint every 50 episodes
        if (episode + 1) % 50 == 0:
            torch.save(policy.state_dict(), f"policy_ep{episode+1}.pth")
            torch.save(value_net.state_dict(), f"value_ep{episode+1}.pth")

    print("Training Complete.")

    # Save final model
    torch.save(policy.state_dict(), "policy_final.pth")
    torch.save(value_net.state_dict(), "value_final.pth")

    # Plot reward curve
    plt.plot(all_rewards)
    plt.title("Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()


if __name__ == "__main__":
    train()