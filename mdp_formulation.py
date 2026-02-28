import numpy as np

class QuickMartEnv:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.num_stores = 10
        self.num_skus = 425
        self.day = 0
        self.max_days = 365

        # 1.1 Regional Setup 
        # Urban: High foot traffic, Suburban: Family shoppers, Rural: Supply challenges
        self.regions = ["Urban"] * 4 + ["Suburban"] * 4 + ["Rural"] * 2
        
        # 1.2 Product Catalog [cite: 35]
        self.categories = (["Electronics"] * 50 + ["Groceries"] * 200 + 
                           ["Apparel"] * 100 + ["Home"] * 75)
        self.elasticities = {"Electronics": -2.5, "Groceries": -0.8, 
                             "Apparel": -1.5, "Home": -1.2}

        # State Initialization [cite: 59]
        self.prices = np.random.uniform(100, 500, (self.num_stores, self.num_skus))
        self.costs = self.prices * 0.7
        self.inventory = np.full((self.num_stores, self.num_skus), 100)
        self.base_demand = np.random.uniform(5, 20, (self.num_stores, self.num_skus))
        self.demand_history = np.zeros((self.num_stores, self.num_skus, 7))
        self.competitor_prices = np.random.uniform(90, 520, (self.num_stores, self.num_skus, 3))
        self.weather = np.zeros((self.num_stores, 2))
        self.customer_segments = np.random.dirichlet(np.ones(5), size=(self.num_stores, self.num_skus))

    def _get_season_one_hot(self):
        season = (self.day // 90) % 4
        one_hot = np.zeros(4)
        one_hot[season] = 1
        return one_hot

    def get_state(self):
        """Returns state: shape (10, 425, 24) as per source 60"""
        season_oh = self._get_season_one_hot()
        all_states = []
        for s in range(self.num_stores):
            store_skus = []
            for sku in range(self.num_skus):
                feat = [self.prices[s, sku], self.costs[s, sku], self.inventory[s, sku]]
                feat.extend(self.demand_history[s, sku])
                feat.extend(self.competitor_prices[s, sku])
                feat.extend(season_oh)
                feat.extend(self.weather[s])
                feat.extend(self.customer_segments[s, sku])
                store_skus.append(feat)
            all_states.append(store_skus)
        return np.array(all_states)

    def get_demand(self, s, sku):
        """Demand Model: D = D_base * (P/P_ref)^epsilon * S * (1+n) [cite: 63]"""
        eps = self.elasticities[self.categories[sku]]
        ref_p = self.costs[s, sku] * 1.2
        # Regional multiplier 
        reg_mult = 1.2 if self.regions[s] == "Urban" else 1.0
        season_mult = 1 + 0.1 * np.sin(2 * np.pi * self.day / 365)
        noise = np.random.normal(0, 0.05)
        
        demand = self.base_demand[s, sku] * (self.prices[s, sku] / ref_p)**eps * season_mult * reg_mult * (1 + noise)
        return max(0, int(demand))

    def step(self, actions):
        """Phase 1.3: Transition with Business Constraints [cite: 42, 75]"""
        total_reward = 0
        self.day += 1
        
        # Update weather randomly each day [cite: 59]
        self.weather = np.random.normal(25, 5, (self.num_stores, 2))

        for s in range(self.num_stores):
            for sku in range(self.num_skus):
                # 1. Action: Price Change (Constraint: +/- 15%) [cite: 42]
                change = np.clip(actions[s, sku], -0.15, 0.15)
                self.prices[s, sku] *= (1 + change)

                # 2. Hard Constraint: Min 5% Margin [cite: 42]
                self.prices[s, sku] = max(self.prices[s, sku], self.costs[s, sku] * 1.05)

                # 3. Process Demand [cite: 63]
                demand = self.get_demand(s, sku)
                sold = min(demand, self.inventory[s, sku])
                self.inventory[s, sku] -= sold
                
                # 4. Inventory Replenishment (Simple Logic for Phase 1) [cite: 24, 44]
                if self.inventory[s, sku] < 20:
                    self.inventory[s, sku] += 100 # Restock trigger

                # 5. Reward Calculation 
                revenue = sold * self.prices[s, sku]
                holding_cost = 0.001 * self.inventory[s, sku]
                stockout_penalty = 10.0 if demand > sold else 0
                
                total_reward += (revenue - holding_cost - stockout_penalty)

                # Update history
                self.demand_history[s, sku] = np.roll(self.demand_history[s, sku], -1)
                self.demand_history[s, sku][-1] = sold

        done = self.day >= self.max_days
        return self.get_state(), total_reward, done

    def reset(self):
        self.__init__()
        return self.get_state()
    
def main():
    print("Initializing QuickMart Environment...")
    env = QuickMartEnv(seed=42)

    # Reset environment
    state = env.reset()
    print("Initial State Shape:", state.shape)  # Expect (10, 425, 24)

    assert state.shape == (10, 425, 24), "State shape is incorrect!"

    total_rewards = []

    print("\nRunning Simulation...\n")

    for step in range(5):  # Run 5 test days
        print(f"--- Day {step + 1} ---")

        # Random valid actions between -15% and +15%
        actions = np.random.uniform(-0.15, 0.15, (env.num_stores, env.num_skus))

        next_state, reward, done = env.step(actions)

        total_rewards.append(reward)

        print("Reward:", round(reward, 2))
        print("Inventory Min:", np.min(env.inventory))
        print("Price Min Margin Check:",
              np.min(env.prices - env.costs * 1.05))

        # Hard constraint checks
        assert np.all(env.inventory >= 0), "Inventory went negative!"
        assert np.all(env.prices >= env.costs * 1.05), "Margin constraint violated!"

        if done:
            print("Simulation reached max days.")
            break

    print("\nSimulation Completed.")
    print("Average Reward:", round(np.mean(total_rewards), 2))


if __name__ == "__main__":
    main()