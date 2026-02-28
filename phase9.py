import numpy as np

# =====================================================
# HARD CONSTRAINTS
# =====================================================
def enforce_hard_constraints(prices, costs, inventory):

    # Margin ≥ 5%
    prices = np.maximum(prices, costs * 1.05)

    # Inventory ≥ 0
    inventory = np.maximum(inventory, 0)

    return prices, inventory


# =====================================================
# SOFT CONSTRAINT PENALTY
# =====================================================
def soft_penalty(stockouts, price_variance):

    penalty = 0

    # Stockout target <2%
    if stockouts > 0.02:
        penalty += (stockouts - 0.02) * 10

    # Cross-store variance target <10%
    if price_variance > 0.10:
        penalty += (price_variance - 0.10) * 5

    return penalty


# =====================================================
# REWARD HACKING PREVENTION
# =====================================================
def prevent_reward_hacking(price_changes):

    smooth_penalty = np.mean(np.abs(price_changes))

    return smooth_penalty


# =====================================================
# TEST
# =====================================================
def test():

    prices = np.random.uniform(90, 110, (10,425))
    costs = prices * 0.9
    inventory = np.random.randint(-5,100,(10,425))

    prices, inventory = enforce_hard_constraints(prices, costs, inventory)

    print("Min Margin Check:", np.min(prices - costs*1.05))
    print("Min Inventory:", np.min(inventory))


if __name__ == "__main__":
    test()