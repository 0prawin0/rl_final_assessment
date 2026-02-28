import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# GRPO ADVANTAGE
# =====================================================
def compute_grpo_advantages(rewards):

    rewards = torch.tensor(rewards, dtype=torch.float32)

    group_mean = rewards.mean()

    advantages = rewards - group_mean

    return advantages


# =====================================================
# VERIFIABLE REWARD CHECKS
# =====================================================
def verify_rewards(prices, costs, inventory):

    margin_ok = (prices >= costs * 1.05).float()
    inventory_ok = (inventory >= 0).float()

    return margin_ok.mean() + inventory_ok.mean()


# =====================================================
# GRPO UPDATE
# =====================================================
def grpo_update(policy, optimizer, states, rewards):

    advantages = compute_grpo_advantages(rewards)

    loss = -advantages.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# =====================================================
# TEST
# =====================================================
def test():

    rewards = np.random.rand(10)

    adv = compute_grpo_advantages(rewards)

    print("Rewards:", rewards)
    print("GRPO Advantages:", adv.numpy())


if __name__ == "__main__":
    test()