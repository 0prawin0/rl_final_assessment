import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# REWARD MODEL (Bradley-Terry)
# =====================================================
class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


# =====================================================
# Bradley-Terry Loss
# P(A > B) = sigmoid(rA - rB)
# =====================================================
def bt_loss(rA, rB):
    return -torch.log(torch.sigmoid(rA - rB)).mean()


# =====================================================
# Dummy preference data generator
# (Phase 5 will replace this with real trajectories)
# =====================================================
def generate_preferences(num_samples=1000, input_dim=102000):

    A = torch.randn(num_samples, input_dim)
    B = torch.randn(num_samples, input_dim)

    # Assume A preferred if sum larger (proxy logic)
    labels = (A.sum(dim=1) > B.sum(dim=1)).float()

    return A, B, labels


# =====================================================
# TRAIN REWARD MODEL
# =====================================================
def train():

    input_dim = 102000  # 10 stores * 425 SKUs * 24 state
    model = RewardModel(input_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20

    for epoch in tqdm(range(epochs)):

        A, B, _ = generate_preferences(256, input_dim)

        A = A.to(device)
        B = B.to(device)

        rA = model(A)
        rB = model(B)

        loss = bt_loss(rA, rB)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "reward_model_phase4.pth")
    print("Saved reward_model_phase4.pth")


if __name__ == "__main__":
    train()