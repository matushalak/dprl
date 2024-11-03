from numpy import zeros, argmax, ndarray, meshgrid
from pandas import DataFrame
import matplotlib.pyplot as plt

def basic_inventory_dp(T_dim:int, X_dim:int, A:list, P:list
                       )->dict[str:tuple[ndarray, DataFrame], 
                               str:tuple[ndarray, DataFrame]]:
    
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
        assert all(DP[0,:] == 0) # make sure that when inventory 0, value always 0
        # 0th row - inventory == 0 is untouched, that is the boundary condition from which
        # the divergence in values across inventory levels emerges
        for x in range(1,X_dim):
            possible_actions = []
            for ai, a in enumerate(A):
                immediate_reward = ER[ai]
                taken_action = P[ai] * DP[x - 1, t + 1]
                untaken_action = (1 - P[ai]) * DP[x, t + 1]
                possible_actions.append(immediate_reward + taken_action + untaken_action)
            
            # best_action
            best_action = argmax(possible_actions)
            DP[x,t] = possible_actions[best_action]
            POLICY[x,t] = A[best_action]

    # Maximum expected revenue
    print(DP[100,0])

    DP_frame = DataFrame(DP, columns = [f't={t}' for t in range(T_dim)]).to_csv('Part1matrix.csv')
    POLICY_frame = DataFrame(POLICY, columns = [f't={t}' for t in range(T_dim-1)]).to_csv('Part1policy.csv')

    return {'dp_matrix':(DP, DP_frame),
            'policy':(POLICY, POLICY_frame)}

def plot_policy(Td:int, Xd:int, pol:ndarray)->None:
    X, Y = meshgrid(list(range(Td-1)),
                    list(range(Xd)))
    
    policy_plot = plt.contourf(X,Y,pol, levels = 1)
    plt.colorbar(policy_plot, label = 'Optimal policy (t, x)')
    plt.xlabel('Time period (t)')
    plt.ylabel('Remaining inventory (x)')
    plt.tight_layout()
    plt.savefig('optimal_policy_partB.png')
    plt.show()

if __name__ == '__main__':
    # GOAL: MAX SALES (max expected cumulative rewards)
    # Time (T): 500:0 decisions left in finite time horizon
    # State (Xt): Invenstory left at time t - 0:100
    #Dimensions of DP matrix
    T_dim, X_dim = 501, 101

    # Actions (A): Change price to 200 | 100 | 50
    # and corresponding immediate reward
    A = [50,100,200]

    # probability of sale with given price
    P = [0.8, 0.5, 0.1]

    # AB parts of the assignment
    ab_results = basic_inventory_dp(T_dim, X_dim, A, P)
    plot_policy(T_dim, X_dim, ab_results['policy'][0])
