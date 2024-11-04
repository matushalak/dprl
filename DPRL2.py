# import matplotlib.pyplot as plt
# from numpy import zeros, argmax, meshgrid, mean
# from numpy.random import uniform
# from pandas import DataFrame

# # Define constants
# MAX_TIME = 500
# MAX_INVENTORY = 100
# PRICES = [50, 100, 200]
# SALE_PROB = {50: 0.8, 100: 0.5, 200: 0.1}

# # Define reward function
# def reward(a):
#     return a * SALE_PROB[a]

# # Define allowed actions based on previous price
# def get_action_set(last_price):
#     if last_price == 50:
#         return PRICES  # All prices are allowed
#     elif last_price == 100:
#         return [50, 100]  # Only 50 or 100 allowed
#     elif last_price == 200:
#         return [50, 200]  # Only 50 or 200 allowed

# # Define and solve the problem as a finite-horizon DP with constrained state space
# def constrained_inventory_dp(T_dim, X_dim, A, P):
#     V = {}  # Value function dictionary with (time, inventory, last_price) as keys
#     POLICY = {}  # Policy dictionary to store the optimal action at each state

#     # Terminal condition for all (inventory, last_price) combinations at MAX_TIME
#     for X in range(X_dim + 1):
#         for last_price in A:
#             V[MAX_TIME, X, last_price] = 0  # No value at the end of the horizon
#             POLICY[MAX_TIME, X, last_price] = 0  # No action at terminal time

#     # Backward induction
#     for t in range(MAX_TIME - 1, -1, -1):
#         for X in range(X_dim + 1):
#             for last_price in A:
#                 if X == 0:
#                     V[t, X, last_price] = 0  # No inventory, value is zero
#                     POLICY[t, X, last_price] = 0  # No action can be taken
#                 else:
#                     action_values = [
#                         reward(a) + SALE_PROB[a] * V.get((t + 1, X - 1, a), 0) +
#                         (1 - SALE_PROB[a]) * V.get((t + 1, X, a), 0)
#                         for a in get_action_set(last_price)
#                     ]
#                     best_action_index = argmax(action_values)
#                     best_action = get_action_set(last_price)[best_action_index]
#                     V[t, X, last_price] = action_values[best_action_index]
#                     POLICY[t, X, last_price] = best_action

#     # Convert V and POLICY dictionaries to DataFrames
#     V_list = [(t, X, p, V[t, X, p]) for t in range(MAX_TIME + 1) for X in range(X_dim + 1) for p in A]
#     POLICY_list = [(t, X, p, POLICY[t, X, p]) for t in range(MAX_TIME + 1) for X in range(X_dim + 1) for p in A]

#     V_frame = DataFrame(V_list, columns=['Time', 'Inventory', 'Last Price', 'Value'])
#     POLICY_frame = DataFrame(POLICY_list, columns=['Time', 'Inventory', 'Last Price', 'Optimal Action'])

#     return V, POLICY, V_frame, POLICY_frame

# # Plot the optimal policy for different inventory levels and last prices
# # def plot_policy_constrained(Td, Xd, policy):
# #     for last_price in PRICES:
# #         plt.figure(figsize=(10, 6))
# #         X, Y = meshgrid(range(Td), range(Xd + 1))
# #         Z = [[policy.get((t, x, last_price), 0) for t in range(Td)] for x in range(Xd + 1)]
# #         plt.contourf(X, Y, Z, levels=len(PRICES))
# #         plt.colorbar(label='Optimal Price')
# #         plt.xlabel('Time')
# #         plt.ylabel('Inventory')
# #         plt.title(f'Optimal Policy for Last Price = {last_price}')
# #         plt.show()


# # import matplotlib.pyplot as plt

# def plot_policy_with_subplots(Td, Xd, policy):
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     last_prices = [50, 100, 200]
    
#     for i, last_price in enumerate(last_prices):
#         ax = axes[i]
#         X, Y = meshgrid(range(Td), range(Xd + 1))
#         Z = [[policy.get((t, x, last_price), 0) for t in range(Td)] for x in range(Xd + 1)]
#         c = ax.contourf(X, Y, Z, levels=len(PRICES), cmap='viridis')
#         fig.colorbar(c, ax=ax, label='Optimal Price')
#         ax.set_title(f'Optimal Policy for Last Price = {last_price}')
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Inventory')
    
#     plt.tight_layout()
#     plt.show()

# # Simulate the process 1000 times under the constrained optimal policy
# def simulate_constrained(Td, Xd, A, P, policy, n_sim=1000):
#     max_rewards = []
#     for _ in range(n_sim):
#         inventory = Xd - 1
#         last_price = 50  # Starting price
#         revenue = 0
#         for t in range(Td - 1):
#             action = policy.get((t, inventory, last_price), 0)
#             if action != 0:
#                 prob = SALE_PROB[action]
#                 if uniform() < prob:
#                     inventory -= 1
#                     revenue += action
#                 last_price = action
#         max_rewards.append(revenue)
#     plt.hist(max_rewards, bins=30)
#     plt.xlabel('Revenue obtained')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of Rewards from Simulations')
#     plt.show()
#     mean_rev = mean(max_rewards)
#     print(f'Mean expected revenue: {mean_rev}')
#     return mean_rev

# # Main function to run DP and simulation
# if __name__ == '__main__':
#     # Define dimensions
#     T_dim = MAX_TIME + 1
#     X_dim = MAX_INVENTORY

#     # Solve DP with constraints
#     V_constrained, policy_constrained, V_df, policy_df = constrained_inventory_dp(T_dim, X_dim, PRICES, SALE_PROB)
#     # plot_policy_constrained(T_dim, X_dim, policy_constrained)
#     plot_policy_with_subplots(T_dim, X_dim, policy_constrained)

#     # Simulate the constrained policy
#     avg_reward_constrained = simulate_constrained(T_dim, X_dim, PRICES, SALE_PROB, policy_constrained)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

class SeasonalProductDP:
    def __init__(self, T, initial_inventory, prices, probabilities):
        self.T = T
        self.initial_inventory = initial_inventory
        self.prices = prices
        self.probabilities = probabilities
        self.V = None
        self.policy = None

    def calculate_value_function(self):
        num_prices = len(self.prices)
        V = np.zeros((self.T + 1, self.initial_inventory + 1, num_prices))  # initialize value function at time t with inventory i and price index p --> V[t, i, p]
        policy = np.zeros((self.T, self.initial_inventory + 1, num_prices), dtype=int)  # policy[t, i, p] stores the optimal action

        for t in reversed(range(self.T)): # backward induction approach (reverse iterate through T)
            for i in range(self.initial_inventory + 1):
                for p in range(num_prices):
                    best_value = float('-inf')
                    best_action = p  # Start with the current price as the best action
                    for a in range(p, num_prices):  # Only allow prices that are less than or equal to the current price level
                        price = self.prices[a]
                        prob = self.probabilities[a]
                        if i > 0:
                            reward = price * prob
                            next_inventory = i - 1
                            expected_value = reward + prob * V[t + 1, next_inventory, a] + (1 - prob) * V[t + 1, i, a]
                        else:
                            expected_value = 0
                        
                        if expected_value > best_value:
                            best_value = expected_value
                            best_action = a
                    
                    V[t, i, p] = best_value
                    policy[t, i, p] = best_action
        
        self.V = V
        self.policy = policy
        return V, policy

    def report_expected_maximal_reward(self):
        if self.V is None:
            raise ValueError("Value function not calculated yet. Run calculate_value_function() first.")
        expected_maximal_reward = max(self.V[0, self.initial_inventory, :])
        print(f"Expected Maximal Reward: {expected_maximal_reward:.2f}")
        return expected_maximal_reward

    def plot_optimal_policy(self):
        plt.figure(figsize=(12, 8))
        cmap = mcolors.ListedColormap(['blue', 'green', 'red'])  # Define a colormap with 3 colors representing actions
        bounds = [-0.5, 0.5, 1.5, 2.5]  # Define bounds for the colors
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        optimal_policy_matrix = np.zeros((self.T, self.initial_inventory + 1))
        for t in range(self.T):
            for i in range(self.initial_inventory + 1):
                optimal_policy_matrix[t, i] = self.prices[self.policy[t, i, 0]]  # Use the optimal price for the initial price index
        
        plt.imshow(optimal_policy_matrix.T, aspect='auto', cmap=cmap, norm=norm, origin='lower')
        colorbar = plt.colorbar(ticks=self.prices)
        colorbar.set_label('Optimal Action (Price)')
        plt.xlabel('Time Period')
        plt.ylabel('Inventory Level')
        plt.title('Optimal Policy as a Function of Time and Inventory Level')
        plt.show()

    def plot_cumulative_reward_over_time(self):
        inventory = self.initial_inventory
        cumulative_reward = 0
        cumulative_rewards = [cumulative_reward]
        
        for t in range(self.T):
            if inventory == 0:
                break
            action = self.policy[t, inventory, 0]
            price = self.prices[action]
            sale = np.random.rand() < self.probabilities[action]
            if sale:
                cumulative_reward += price
                inventory -= 1
            cumulative_rewards.append(cumulative_reward)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(cumulative_rewards)), cumulative_rewards, marker='o')
        plt.xlabel('Time Period')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward Over Time (Single Simulation)')
        plt.show()

    def plot_value_function(self):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.V[:self.T, :self.initial_inventory + 1, 0], aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Value')
        plt.xlabel('Inventory Level')
        plt.ylabel('Time Period')
        plt.title('Value Function Heatmap')
        plt.show()

    def plot_reward_histogram(self, rewards):
        plt.figure(figsize=(10, 6))
        plt.hist(rewards, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(rewards), color='r', linestyle='dashed', linewidth=1.5, label=f'\nAverage Reward: {np.mean(rewards):.2f}\n')
        plt.xlabel('Total Reward')
        plt.ylabel('Frequency')
        plt.title('Histogram of Rewards over 1000 Simulations')
        plt.legend()
        plt.show()

    def plot_inventory_over_time(self):
        inventory = self.initial_inventory
        inventory_levels = [inventory]
        for t in range(self.T):
            if inventory == 0:
                break
            action = self.policy[t, inventory, 0]
            sale = np.random.rand() < self.probabilities[action]
            if sale:
                inventory -= 1
            inventory_levels.append(inventory)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(inventory_levels)), inventory_levels, marker='o')
        plt.xlabel('Time Period')
        plt.ylabel('Inventory Level')
        plt.title('Inventory Level Over Time (Single Simulation)')
        plt.show()

    def run_simulations(self, n_simulations=1000):
        rewards = []
        for _ in range(n_simulations):
            inventory = self.initial_inventory
            total_reward = 0
            for t in range(self.T):
                if inventory == 0:
                    break
                action = self.policy[t, inventory, 0]
                price = self.prices[action]
                sale = np.random.rand() < self.probabilities[action]
                if sale:
                    total_reward += price
                    inventory -= 1
            rewards.append(total_reward)
        return rewards

    def main(self):
        self.calculate_value_function()
        rewards = self.run_simulations()
        
        self.plot_optimal_policy()
        self.plot_reward_histogram(rewards)
        self.plot_inventory_over_time()
        self.plot_cumulative_reward_over_time()
        self.plot_value_function()

# Example usage
T = 500
initial_inventory = 100
prices = [200, 100, 50]
probabilities = [0.1, 0.5, 0.8]

dp = SeasonalProductDP(T, initial_inventory, prices, probabilities)
dp.main()
