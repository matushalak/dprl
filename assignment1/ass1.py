from numpy import zeros, argmax, ndarray, meshgrid, mean, dtype
from numpy.random import uniform
from pandas import DataFrame
import matplotlib.pyplot as plt
import argparse

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



def plot_policy(Td:int, Xd:int, pol:ndarray, Actions:list,
                name:str, show:bool = False)->None:
    X, Y = meshgrid(list(range(Td-1)),
                    list(range(Xd)))

    cmap = plt.get_cmap('viridis', len(Actions))  # Define a colormap with N-1 colors
    
    pol[0,:len(Actions)] = Actions # dirty trick to allow same colorbar for both plots
    
    policy_plot = plt.pcolormesh(X,Y,pol,
                                 cmap = cmap)
    plt.colorbar(policy_plot, label = 'Optimal policy (t, x)', 
                 boundaries = [0] + Actions if 0 not in Actions else Actions,
                 drawedges = True, cmap = cmap)
    plt.xlabel('Time period (t)')
    plt.ylabel('Remaining inventory (x)')
    # start from one
    plt.yticks([1]+[tick for tick in range(0,Xd,20)][1:])
    plt.xticks([tick for tick in range(0,Td,(Td-1)//5)][:-1] + [Td -2])

    plt.ylim(bottom = 1)
    plt.tight_layout()
    plt.savefig(name)
    if show == True:
        plt.show()
    plt.close()
    


def simulate(Td:int, Xd:int, A:list, P:list, policy:ndarray, 
             n_sim:int = 1000, show:bool = False) -> float:
    '''
    Produces histogram of revenue earned in T time over n simulations
    Also produces an example plot earnings in a given simulation
    '''
    maxrewards = []
    for sim in range(n_sim):
        x = Xd - 1 
        revenue = 0

        # example revenue plot
        if sim == 0:
            inventory_levels = [x]
            revenue_levels = [0]

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
                
                if sim == 0:
                    inventory_levels.append(x)
                    revenue_levels.append(revenue)

        maxrewards.append(revenue)

    print('Mean expected revenue', meanrev := mean(maxrewards))

    plt.hist(maxrewards, bins = 'auto')
    # plt.title(f'Histogram of revenue obtained over {n_sim} simulations')
    plt.axvline(x= meanrev, color = 'r', linestyle= "--", 
                label = f'Average Revenue: {meanrev}') # mean expected revenue
    plt.ylabel('Frequency')
    plt.xlabel('Revenue obtained using optimal policy')

    plt.legend()
    plt.tight_layout()
    plt.savefig('simulation.png')
    if show == True:
        plt.show()
    plt.close()

    # example revenue plot
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel(r'$Inventory\ Level\ (X_t)$', color=color)
    ax1.plot(xax := range(len(inventory_levels)), inventory_levels, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(r'$Revenue\ \$\ (V_t)$', color=color) # we already handled the x-label with ax1
    ax2.plot(xax, revenue_levels, color=color,
             label = f'Earned revenue = {int(revenue_levels[-1])}$')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.legend(loc = 8)
    # plt.title(f'Example simulation run with {revenue_levels[-1]}$ revenue')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('example_run.png')
    if show == True:
        plt.show()
    plt.close()

    return meanrev



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t','--time', type = int, default = 500)
    parser.add_argument('-i','--inventory', type = int, default = 100)
    parser.add_argument('-a', '--actions', nargs='+', type = float, default = [50, 100, 200])
    parser.add_argument('-p', '--probabilities', nargs='+', type = float, default = [0.8, 0.5, 0.1])
    parser.add_argument('-show_plots', type = bool, default = False)
    
    return parser.parse_args()
    


if __name__ == '__main__':
    # GOAL: MAX SALES (max expected cumulative rewards)

    # generalizable version to any T, X and any number / type of actions & probabilities
    args = parse_args()    

    # Time (T): 500:0 decisions left in finite time horizon
    # State (Xt): Invenstory left at time t - 0:100
    #Dimensions of DP matrix
    # T_dim, X_dim = 501, 101
    T_dim = args.time + 1
    X_dim = args.inventory + 1
    
    # Actions (A): Change price to 200 | 100 | 50
    # and corresponding immediate reward
    # A = [50,100,200]
    A = args.actions
    
    # probability of sale with given price
    # P = [0.8, 0.5, 0.1]
    P = args.probabilities
    show_plots = args.show_plots

    # AB parts of the assignment
    MAXREV, rewards, policy = basic_inventory_dp(T_dim, X_dim, A, P)
    plot_policy(T_dim, X_dim, policy, A,
                name = f'optimal_policy_{T_dim-1}_{X_dim-1}_{A}_partB.png', 
                show = show_plots)
    avg_reward = simulate(T_dim, X_dim, A, P, policy, show = show_plots)

    # Part D, E
    # State (Xt): (Inventory left at time t, PRICE at time t) TUPLE

    # Actions (A(Price t+1)): Change price to a E {50, 100, 200} & a >= Price t+1 
    # at every time point will include only actions with higher price than price at X(t+1)
    mr, rew, pol = basic_inventory_dp(T_dim, X_dim, A, P, partDE=True)
    plot_policy(T_dim, X_dim, pol, A, 
                show = show_plots, 
                name = f'optimal_policy_{T_dim-1}_{X_dim-1}_{A}_partD.png')

