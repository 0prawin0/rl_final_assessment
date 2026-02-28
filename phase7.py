import torch
import numpy as np

# =====================================================
# STRATEGIC LLM-LIKE PLANNER
# (Simulated - acts as high-level policy)
# =====================================================
class StrategicPlanner:

    def analyze_sales(self, state):
        return np.mean(state)

    def check_competitor_prices(self, state):
        return np.min(state)

    def forecast_demand(self, state):
        return np.mean(state) * 1.05

    def recommend_price(self, rl_price, demand_forecast):
        if demand_forecast > 0.5:
            return rl_price * 1.05
        else:
            return rl_price * 0.95

    def execute_price_change(self, price):
        return np.clip(price, -0.15, 0.15)


# =====================================================
# LLM-AUGMENTED ACTION ADAPTER
# =====================================================
class LLMAgentWrapper:

    def __init__(self):
        self.planner = StrategicPlanner()

    def adjust_actions(self, actions, state):

        adjusted = []

        for i in range(actions.shape[0]):

            store_state = state[i]

            demand = self.planner.forecast_demand(store_state)
            new_action = self.planner.recommend_price(actions[i], demand)
            final = self.planner.execute_price_change(new_action)

            adjusted.append(final)

        return np.array(adjusted)


# =====================================================
# TEST
# =====================================================
def test():

    wrapper = LLMAgentWrapper()

    actions = np.random.uniform(-0.15, 0.15, (10, 425))
    state = np.random.rand(10, 425, 24)

    adjusted = wrapper.adjust_actions(actions, state)

    print("Original mean:", np.mean(actions))
    print("Adjusted mean:", np.mean(adjusted))


if __name__ == "__main__":
    test()