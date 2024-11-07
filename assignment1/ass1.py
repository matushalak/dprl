<<<<<<< Updated upstream
# First part of the first assignment
=======
from numpy import zeros, argmax, ndarray, meshgrid, mean, dtype
from numpy.random import uniform
from pandas import DataFrame
import matplotlib.pyplot as plt

def basic_inventory_dp(T_dim:int, X_dim:int, A:list, P:list,
                       partDE:bool = False
                       )->tuple[float,
                                ndarray,DataFrame,
                                ndarray, DataFrame]:
    
    if partDE:
        # use structured array
        DP = zeros((X_dim, T_dim),
                   dtype = dtype([('revenue', 'f8'), ('set_price', 'i4')])) # add extra dimension to State space
    else:
        DP = zeros((X_dim, T_dim)) # initialize with zeros 
    
    POLICY = zeros((X_dim,T_dim-1))

    # Expected immediate reward = prob of action * value of action
    # in ABC reward is independent of state
    ER = [p*a for p,a in zip(P,A)]

    # Value function
    # Bellman equation:
        # V(x,t) = max(a E {50,100,200}) (a * P(a) + P(a) * V(x-1,t+1) + (1-P(a)) * V(x, t+1))
    # go backwards filling the table from the ...<-498th<-499th<-500th column
    # 501st column is all just zeros (no value after time horizon)
    for t in range(T_dim-2, -1, -1):
        if not partDE:
            assert all(DP[0,:] == 0) # make sure that when inventory 0, value always 0
        # 0th row - inventory == 0 is untouched, that is the boundary condition from which
        # the divergence in values across inventory levels emerges
        for x in range(1,X_dim):
            possible_actions = []
            for ai, a in enumerate(A):
                if partDE:
                    # can only decrease or keep price constant over time, NOT increase! = a(t) >= a(t+1)
                    # cover both options of sale & no sale
                    # if cover only no sale option - same result
                    if a >= DP[x,t+1]['set_price'] and a >= DP[x-1,t+1]['set_price']:
                        immediate_reward = ER[ai]
                        taken_action = P[ai] * DP[x - 1, t + 1]['revenue']
                        untaken_action = (1 - P[ai]) * DP[x, t + 1]['revenue']
                        possible_actions.append(immediate_reward + taken_action + untaken_action)
                    else:
                        # key piece that was missing
                        possible_actions.append(-1) # so that not chosen but to preserve indexing
                else:
                    immediate_reward = ER[ai]
                    taken_action = P[ai] * DP[x - 1, t + 1]
                    untaken_action = (1 - P[ai]) * DP[x, t + 1]
                    possible_actions.append(immediate_reward + taken_action + untaken_action)
            
            # best_action
            best_action = argmax(possible_actions)
            if partDE:
                DP[x,t]['revenue'] = possible_actions[best_action]
                DP[x,t]['set_price'] = A[best_action] # add taken action to state
                POLICY[x,t] = A[best_action]
            else:
                DP[x,t] = possible_actions[best_action]
            POLICY[x,t] = A[best_action]

    # Maximum expected revenue
    print('Optimal expected revenue:', maxrev := DP[100,0])

    if not partDE:
        DP_frame = DataFrame(DP, columns = [f't={t}' for t in range(T_dim)]).to_csv('Part1matrix.csv')
        POLICY_frame = DataFrame(POLICY, columns = [f't={t}' for t in range(T_dim-1)]).to_csv('Part1policy.csv')

    return (maxrev,
            DP,
            POLICY)

def plot_policy(Td:int, Xd:int, pol:ndarray,show:bool = False,
                name:str = 'optimal_policy_partB.png')->None:
    X, Y = meshgrid(list(range(Td-1)),
                    list(range(Xd)))
    
    # policy_plot = plt.contour(X,Y,pol, levels = [0,50,100,200])
    cmap = plt.get_cmap('viridis', 3)  # Define a colormap with N-1 colors
    
    pol[0,0] = 200 # dirty trick to allow same colorbar for both plots
    
    policy_plot = plt.pcolormesh(X,Y,pol,
                                 cmap = cmap)
    plt.colorbar(policy_plot, label = 'Optimal policy (t, x)', 
                 values = [50,100,200], boundaries = [0,50,100,200],
                 drawedges = True, cmap = cmap)
    plt.xlabel('Time period (t)')
    plt.ylabel('Remaining inventory (x)')
    # start from one
    plt.yticks([1]+[tick for tick in range(0,101,20)][1:])
    plt.xticks([0,100,200,300,400,499])
    plt.ylim(bottom = 1)
    plt.tight_layout()
    plt.savefig(name)
    if show == True:
        plt.show()
    plt.close()
    
def simulate(Td:int, Xd:int, A:list, P:list, policy:ndarray, 
             n_sim:int = 1000, show:bool = False) -> float:
    maxrewards = []
    for _ in range(n_sim):
        x = Xd - 1 
        revenue = 0
        for t in range(Td-2):
            # follow optimal policy
            action = policy[x,t]
            if action != 0:
                act_index = A.index(action)
                prob = P[act_index]

                # probabilistic
                if uniform() < prob:
                    x -= 1
                    revenue += action

        maxrewards.append(revenue)

    print('Mean expected revenue', meanrev := mean(maxrewards))

    plt.hist(maxrewards, bins = 50)
    plt.axvline(x= meanrev, color = 'r', linestyle= "--") # mean expected revenue
    plt.ylabel('Frequency')
    plt.xlabel('Revenue obtained using optimal policy')
    plt.tight_layout()
    plt.savefig('simulation.png')
    if show == True:
        plt.show()
    plt.close()
    return meanrev

# if __name__ == '__main__':
#     # GOAL: MAX SALES (max expected cumulative rewards)
#     # Time (T): 500:0 decisions left in finite time horizon
#     # State (Xt): Invenstory left at time t - 0:100
#     #Dimensions of DP matrix
#     T_dim, X_dim = 501, 101

#     # Actions (A): Change price to 200 | 100 | 50
#     # and corresponding immediate reward
#     A = [50,100,200]

#     # probability of sale with given price
#     P = [0.8, 0.5, 0.1]

#     # AB parts of the assignment
#     MAXREV, rewards, policy = basic_inventory_dp(T_dim, X_dim, A, P)
#     plot_policy(T_dim, X_dim, policy)
#     avg_reward = simulate(T_dim, X_dim, A, P, policy)
#     print("Average reward from 1000 simulations:", avg_reward)


#     # Part D, E
#     # State (Xt): (Inventory left at time t, PRICE at time t) TUPLE
    

#     # Actions (A(Price t+1)): Change price to a E {50, 100, 200} & a >= Price t+1 
#     # at every time point will include only actions with higher price than price at X(t+1)
#     mr, rew, pol = basic_inventory_dp(T_dim, X_dim, A, P, partDE=True)
#     plot_policy(T_dim, X_dim, pol, name = 'optimal_policy_partD.png')

if __name__ == '__main__':
    # Initial setup for DP
    T_dim, X_dim = 501, 101
    A = [50, 100, 200]
    P = [0.8, 0.5, 0.1]

    # Solve for Part D with price constraint
    max_reward_D, rewards_D, policy_D = basic_inventory_dp(T_dim, X_dim, A, P, partDE=True)
    
    # Print the expected maximal reward for Part D
    print("Expected maximal reward with no price increase constraint (Part D):", max_reward_D)
    
    # Plot the optimal policy for Part D
    plot_policy(T_dim, X_dim, policy_D, name='optimal_policy_partD.png')

    
    
# import sys
# print("Python version:", sys.version)
>>>>>>> Stashed changes
