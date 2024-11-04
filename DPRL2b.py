import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy import zeros, argmax, meshgrid, mean
from numpy.random import uniform
from pandas import DataFrame

# Define constants
MAX_TIME = 500
MAX_INVENTORY = 100
PRICES = [50, 100, 200]
SALE_PROB = {50: 0.8, 100: 0.5, 200: 0.1}

# Define reward function
def reward(a):
    return a * SALE_PROB[a]

# Define allowed actions based on previous price
def get_action_set(last_price):
    if last_price == 50:
        return PRICES  # All prices are allowed
    elif last_price == 100:
        return [50, 100]  # Only 50 or 100 allowed
    elif last_price == 200:
        return [50, 200]  # Only 50 or 200 allowed

# Define and solve the problem as a finite-horizon DP with constrained state space
def constrained_inventory_dp(T_dim, X_dim, A, P):
    V = {}  # Value function dictionary with (time, inventory, last_price) as keys
    POLICY = {}  # Policy dictionary to store the optimal action at each state

    # Terminal condition for all (inventory, last_price) combinations at MAX_TIME
    for X in range(X_dim + 1):
        for last_price in A:
            V[MAX_TIME, X, last_price] = 0  # No value at the end of the horizon
            POLICY[MAX_TIME, X, last_price] = 0  # No action at terminal time

    # Backward induction
    for t in range(MAX_TIME - 1, -1, -1):
        for X in range(X_dim + 1):
            for last_price in A:
                if X == 0:
                    V[t, X, last_price] = 0  # No inventory, value is zero
                    POLICY[t, X, last_price] = 0  # No action can be taken
                else:
                    action_values = [
                        reward(a) + SALE_PROB[a] * V.get((t + 1, X - 1, a), 0) +
                        (1 - SALE_PROB[a]) * V.get((t + 1, X, a), 0)
                        for a in get_action_set(last_price)
                    ]
                    best_action_index = argmax(action_values)
                    best_action = get_action_set(last_price)[best_action_index]
                    V[t, X, last_price] = action_values[best_action_index]
                    POLICY[t, X, last_price] = best_action

    # Check for any missing actions in the policy dictionary
    missing_actions = sum(1 for key in POLICY if POLICY[key] == 0)
    print(f"Number of states with no valid action (0): {missing_actions}")

    return V, POLICY

# Plot the optimal policy for different inventory levels and last prices
def plot_policy_with_subplots(Td, Xd, policy):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    last_prices = [50, 100, 200]
    
    # Define a discrete colormap for the three price levels and zero as white
    cmap = mcolors.ListedColormap(['white', 'purple', 'blue', 'green'])
    bounds = [0, 50, 100, 200, 250]  # Boundaries for color mapping; 250 is a dummy upper bound
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    for i, last_price in enumerate(last_prices):
        ax = axes[i]
        X, Y = meshgrid(range(Td), range(Xd + 1))
        
        # Get optimal prices from the policy, mapped to the actual price values
        Z = [[policy.get((t, x, last_price), 0) for t in range(Td)] for x in range(Xd + 1)]
        
        # Plot with discrete colormap and set color boundaries to exact prices
        c = ax.contourf(X, Y, Z, levels=bounds, cmap=cmap, norm=norm)
        fig.colorbar(c, ax=ax, ticks=[0, 50, 100, 200], label='Optimal Price')  # Set colorbar ticks to exact prices
        
        ax.set_title(f'Optimal Policy for Last Price = {last_price}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Inventory')
    
    plt.tight_layout()
    plt.show()

# Main function to run DP and plot
if __name__ == '__main__':
    # Define dimensions
    T_dim = MAX_TIME + 1
    X_dim = MAX_INVENTORY

    # Solve DP with constraints
    V_constrained, policy_constrained = constrained_inventory_dp(T_dim, X_dim, PRICES, SALE_PROB)
    
    # Plot optimal policy for each last price
    plot_policy_with_subplots(T_dim, X_dim, policy_constrained)
