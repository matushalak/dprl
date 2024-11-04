# import matplotlib.pyplot as plt
# from numpy import zeros, argmax, ndarray, meshgrid, mean
# from numpy.random import uniform
# from pandas import DataFrame

# def basic_inventory_dp(T_dim:int, X_dim:int, A:list, P:list) -> tuple:
#     DP = zeros((X_dim, T_dim))  # initialize with zeros
#     POLICY = zeros((X_dim, T_dim - 1))

#     # Expected immediate reward
#     ER = [p * a for p, a in zip(P, A)]

#     # Value function using Bellman equation
#     for t in range(T_dim - 2, -1, -1):
#         for x in range(1, X_dim):
#             possible_actions = [
#                 ER[ai] + P[ai] * DP[x - 1, t + 1] + (1 - P[ai]) * DP[x, t + 1]
#                 for ai, a in enumerate(A)
#             ]
#             best_action = argmax(possible_actions)
#             DP[x, t] = possible_actions[best_action]
#             POLICY[x, t] = A[best_action]

#     maxrev = DP[100, 0]  # Optimal expected revenue
#     DP_frame = DataFrame(DP, columns=[f't={t}' for t in range(T_dim)])
#     POLICY_frame = DataFrame(POLICY, columns=[f't={t}' for t in range(T_dim - 1)])

#     return maxrev, DP, DP_frame, POLICY, POLICY_frame

# def plot_policy(Td:int, Xd:int, pol:ndarray):
#     X, Y = meshgrid(range(Td - 1), range(Xd))
#     plt.contourf(X, Y, pol, levels=1)
#     plt.colorbar(label='Optimal policy (t, x)')
#     plt.xlabel('Time period (t)')
#     plt.ylabel('Remaining inventory (x)')
#     plt.tight_layout()
#     plt.show()  # Directly display the plot

# def simulate(Td:int, Xd:int, A:list, P:list, policy:ndarray, n_sim:int=1000):
#     maxrewards = []
#     for _ in range(n_sim):
#         x = Xd - 1 
#         revenue = 0
#         for t in range(Td - 2):
#             action = policy[x, t]
#             if action != 0:
#                 act_index = A.index(action)
#                 prob = P[act_index]
#                 if uniform() < prob:
#                     x -= 1
#                     revenue += action
#         maxrewards.append(revenue)
#     plt.hist(maxrewards, bins=30)
#     plt.ylabel('Frequency')
#     plt.xlabel('Revenue obtained using optimal policy')
#     plt.tight_layout()
#     plt.show()
#     meanrev = mean(maxrewards)
#     print('Mean expected revenue', meanrev)
#     return meanrev

# if __name__ == '__main__':
#     # Define constants
#     T_dim, X_dim = 501, 101
#     A = [50, 100, 200]
#     P = [0.8, 0.5, 0.1]

#     # Run dynamic programming for inventory
#     MAXREV, rewards, rDF, policy, pDF = basic_inventory_dp(T_dim, X_dim, A, P)
#     plot_policy(T_dim, X_dim, policy)
#     avg_reward = simulate(T_dim, X_dim, A, P, policy)

import numpy as np
import matplotlib.pyplot as plt
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
        V = np.zeros((self.T + 1, self.initial_inventory + 1))  # initialize value function at time t with inventory i --> V[t, i]
        policy = np.zeros((self.T, self.initial_inventory + 1), dtype=int)  # policy[t, i] stores the optimal action

        for t in reversed(range(self.T)): # backward induction approach (reverse iterate through T)
            for i in range(self.initial_inventory + 1):
                best_value = float('-inf')
                best_action = 0
                for a, (price, prob) in enumerate(zip(self.prices, self.probabilities)):  # Iterate over actions (prices) to calc value (bellman expectation)
                    if i > 0:
                        reward = price * prob
                        next_inventory = i - 1
                        expected_value = reward + prob * V[t + 1, next_inventory] + (1 - prob) * V[t + 1, i]
                    else:
                        expected_value = 0
                    
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = a
                
                V[t, i] = best_value
                policy[t, i] = best_action
        
        self.V = V
        self.policy = policy
        return V, policy

    def run_simulations(self, n_simulations=1000):
        rewards = []
        for _ in range(n_simulations):
            inventory = self.initial_inventory
            total_reward = 0
            for t in range(self.T):
                if inventory == 0:
                    break
                action = self.policy[t, inventory]
                price = self.prices[action]
                sale = np.random.rand() < self.probabilities[action]
                if sale:
                    total_reward += price
                    inventory -= 1
            rewards.append(total_reward)
        return rewards

    def plot_optimal_policy(self):
        plt.figure(figsize=(12, 6))
        for i in range(0, self.initial_inventory + 1, 1):  # Plot for different inventory levels
            plt.plot(range(self.T), self.policy[:, i], label=f'Inventory {i}')
        plt.xlabel('Time Period')
        plt.ylabel('Optimal Action (Price Index)')
        plt.title('Optimal Policy as a Function of Time and Inventory Level')
        plt.legend()
        plt.show()

    def plot_cumulative_reward_over_time(self):
        inventory = self.initial_inventory
        cumulative_reward = 0
        cumulative_rewards = [cumulative_reward]
        
        for t in range(self.T):
            if inventory == 0:
                break
            action = self.policy[t, inventory]
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
        plt.imshow(self.V[:self.T, :self.initial_inventory + 1], aspect='auto', cmap='viridis', origin='lower')
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
            action = self.policy[t, inventory]
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

    def plot_optimal_policy(self):
        plt.figure(figsize=(12, 8))
        cmap = mcolors.ListedColormap(['blue', 'green', 'red'])  # Define a colormap with 3 colors representing actions
        bounds = [-0.5, 0.5, 1.5, 2.5]  # Define bounds for the colors
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        plt.imshow(self.policy.T, aspect='auto', cmap=cmap, norm=norm, origin='lower')
        colorbar = plt.colorbar(ticks=[0, 1, 2])
        colorbar.set_ticklabels([f'Price {self.prices[0]}', f'Price {self.prices[1]}', f'Price {self.prices[2]}'])
        colorbar.set_label('Optimal Action (Price)')
        plt.xlabel('Time Period')
        plt.ylabel('Inventory Level')
        plt.title('Optimal Policy as a Function of Time and Inventory Level')
        plt.show()
        
        

    def main(self):
        self.calculate_value_function()
        rewards = self.run_simulations()
        
        self.plot_optimal_policy()
        self.plot_reward_histogram(rewards)
        self.plot_inventory_over_time()
        self.plot_cumulative_reward_over_time()

    def calculate_value_function(self):
        V = np.zeros((self.T + 1, self.initial_inventory + 1))  # initialize value function at time t with inventory i --> V[t, i]
        policy = np.zeros((self.T, self.initial_inventory + 1), dtype=int)  # policy[t, i] stores the optimal action

        for t in reversed(range(self.T)): # backward induction approach (reverse iterate through T)
            for i in range(self.initial_inventory + 1):
                best_value = float('-inf')
                best_action = 0
                for a, (price, prob) in enumerate(zip(self.prices, self.probabilities)):  # Iterate over actions (prices) to calc value (bellman expectation)
                    if i > 0:
                        reward = price * prob
                        next_inventory = i - 1
                        expected_value = reward + prob * V[t + 1, next_inventory] + (1 - prob) * V[t + 1, i]
                    else:
                        expected_value = 0
                    
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = a
                
                V[t, i] = best_value
                policy[t, i] = best_action
        
        self.V = V
        self.policy = policy
    
    # Print the maximal value in the value function
        print("Maximal Value in Value Function:", np.max(self.V))
    
        return V, policy


T = 500
initial_inventory = 100
prices = [200, 100, 50]
probabilities = [0.1, 0.5, 0.8]

dp = SeasonalProductDP(T, initial_inventory, prices, probabilities)
dp.main()