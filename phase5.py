import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from mdp_formulation import QuickMartEnv

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# STORE AGENT
# =====================================================
class StoreAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.shared(state)

        mean = self.actor(x)
        value = self.critic(x).squeeze(-1)

        log_std = torch.clamp(self.log_std, -5, 2)
        std = torch.exp(log_std)

        return mean, std, value


# =====================================================
# TRAIN MARL
# =====================================================
def train():

    env = QuickMartEnv()
    state = env.reset()

    sku_count = env.num_skus
    store_count = env.num_stores

    state_dim = 24 * sku_count
    action_dim = sku_count

    agents = [StoreAgent(state_dim, action_dim).to(device) for _ in range(store_count)]
    optimizers = [optim.Adam(agent.parameters(), lr=3e-5) for agent in agents]

    episodes = 30

    for episode in tqdm(range(episodes)):

        state = env.reset()
        done = False

        while not done:

            actions = []

            for i in range(store_count):

                store_state = state[i].reshape(-1)
                store_state = torch.tensor(store_state, dtype=torch.float32, device=device)
                store_state = torch.nan_to_num(store_state) / 500.0

                mean, std, _ = agents[i](store_state)

                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                action = torch.clamp(action, -0.15, 0.15)

                actions.append(action.detach().cpu().numpy())

            actions = np.array(actions)

            next_state, reward, done = env.step(actions)

            # simple cooperative reward
            reward = reward / 1e6

            for opt in optimizers:
                opt.zero_grad()

            loss = -torch.tensor(reward, requires_grad=True)

            loss.backward()

            for opt in optimizers:
                opt.step()

            state = next_state

    print("MARL Training Complete")

    for i, agent in enumerate(agents):
        torch.save(agent.state_dict(), f"store_agent_{i}.pth")

if __name__ == "__main__":
    train()