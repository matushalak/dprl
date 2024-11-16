# @matushalak
# Markov Decision Process assignment 2 DPRL
# infinite horizon problem
from numpy import ndarray, zeros, arange, savetxt, where, array, full, argmin
from numpy.random import randint, uniform, seed
from numpy.linalg import eig
from itertools import product

def mdp(capacity:tuple[int, int] = [20, 20], o:int = 5):
    # a)
    # State space (X)
    c1, c2 = capacity
    # can never start in 0 and thus actions can't be taken based on 0,0
    # and can never END in 0 (because orders arrive by end of day and FORCED to order once inventory 1)
    I1 = arange(1,c1+1)
    I2 = arange(1,c2+1)
    # all possible states X in a list, need corresponding transition probability matrix
    # while each states lies somewhere in 2D space here it is a flattened vector because we care about
    # transitions from any 2D state -> another 2D state
    STATES = [(i, j) for i, j in product(I1, I2)] # X

    # b)
    # Transition probability matrix (P)
    P = zeros((len(STATES), 
                len(STATES)))

    # for a, (o1, o2) in enumerate([(0,0)]+all_actions):
        # Row = FROM, Column = TO
    for FROM, (i1, i2) in enumerate(STATES):
        # if a == 0:
            # straight away include fixed policy condition at Depth 0
        if 1 in (i1, i2):
            # order up to 5
            # XXX change !!!!
            # possibility to sell at I = 1 after order
            TO = [STATES.index((i1+j1 if i1 > 5 else 5+j1,
                                i2+j2 if i2 > 5 else 5+j2))
                                for j1 in range(-1, 1) for j2 in range(-1, 1) ]
        
        else:
            # otherwise no order
            TO = [STATES.index((i1+j1, i2+j2)) 
                for j1 in range(-1, 1) for j2 in range(-1, 1) 
                if 0 not in (i1+j1, i2+j2)]
    
        # dictionary lookup: faster over if/else or match/case
        assert len(TO) != 3
        p = {1 : 1.0, 2 : 0.5, 4 : 0.25}
        # for that FROM state, give corresponding TO states and corresponding probabilities
        P[FROM, TO] = p[len(TO)] # this gives the filled transition probability matrix
        assert P[FROM,:].sum() == 1

    # save P matrix to inspect
    savetxt('2d.txt', P, fmt = '%.2f')
    # savez_compressed('full_mat', P)
    return P, STATES


# a)
# Action space (A) - possible orders, 2 independent actions
# theoretically, actions are order UP to levels {1, ..., 20} | I + a <= 20
# or orders of (0,0), (0,1), (1,0), ..., (19, 19) 
#   => 361 options, I guess policy iteration is choosing between those options?
# EVERY ACTION = NEW TRANSITION PROBABILITY MATRIX
def order(state:tuple[int, int], max_invL:tuple[int, int] = [20, 20], fixed:bool = False):
    if fixed: # fixed order policy
        # FIXED POLICY: if either Inventory level = 1: order so that both up to 5
        return (5-state[0], 5-state[1]) if 1 in state else [0,0]
    
    else: # all possible orders
        max_order = [max_invL[0]-state[0],
                    max_invL[1]-state[1]]
        min_order = [0 if state[0] > 1 else 1,
                    0 if state[1] > 1 else 1]
        
        # 3rd dimension of P array - 1 depth / action applied in all states
        all_orders = [(o1, o2) for o1, o2 in product(arange(min_order[0], max_order[0]+1),
                                                    arange(min_order[1], max_order[1]+1))]
        return all_orders
    
# c)
# start with random x0, fixed actions
def simulate(h:tuple[float, float] = (1,2), o:float = 5, 
             d:tuple[float,float] = (.5,.5),
             T_sim:int = 1e5) -> float:
    # average of all initial states
    long_run_costs = 0
    x = randint(1, 21, 2)
    # x = array((1,1))
    print(x)
    for t in range(round(T_sim)):
        long_run_costs += x[0] * h[0] + x[1] * h[1] + (5 if 1 in x else 0)

        ord = array([0,0])
        if 1 in x:
            # if t < 300:
            #     print(t, x)
            ord[x <= o] = o - x[x <= o]

        demand = uniform(size = 2)
        x[demand < d] -= 1

        # orders arrive end of day
        x += ord

    return long_run_costs / T_sim


# d)
# start with random x0
def stationary_distribution(Xs:list, P:ndarray,
                          h:list[float, float] = [1, 2], o:float = 5,
                          iteration:bool = False, T_sim:int = 1e5, d:tuple[float,float] = (.5,.5)) -> float:

    if iteration:
        x = randint(1, 21, 2)
        # x = array((1,1))

        π = zeros(len(Xs))
        for t in range(round(T_sim)):
            π[Xs.index(tuple(x))] += 1

            ord = array([0,0])
            if 1 in x:
                ord[x <= o] = o - x[x <= o]
            
            demand = uniform(size = 2)
            x[demand < d] -= 1

            # orders arrive end of day
            x += ord

        π = π / round(T_sim)

    # exact π calculation
    else:
        # stationary dist == left eigenvector of transition matrix
        eigvals, eigvects = eig(P.T) # get left eigenvectors by transposing transition matrix)
        π =  eigvects[:,argmin(abs(eigvals - 1))].real
        π = abs(π / sum(π)) # probabilities
        print("Residual:", max(abs((π @ P) - π)))
        assert all((π @ P).round(10) == π.round(10)) # stationary dist = left eigenvector of P
        
    Xs = array(Xs)
    # elementwise multiplication
    holdingCs = full(Xs.shape, h) * Xs
    holdingCs = holdingCs.sum(axis = 1) # now a 1D vector
    
    # order costs = boolean mask multiplied by order costs
    orders = ((Xs[:,0] == 1) | (Xs[:,1] == 1)) * o

    # COSTS = holding costs + order costs
    C = holdingCs + orders
    return π @ C, π # same as sum(π * C)


# e)
# initiate with random V0
def poisson_value_iteration(X:ndarray, P:ndarray,
                            epsilon:float):
    long_run_average = 0
    return long_run_average




#%% running the script
def main():
    P, X  = mdp()
    # c)
    long_term1 = simulate()
    print(f'Simulation: {long_term1}')
    # d)
    long_term2, πsim = stationary_distribution(X, P, iteration=True)
    print(f'Stationary SIM: {long_term2}')
    long_term2m, πexact = stationary_distribution(X, P, iteration=False)
    print(f'Stationary MATH: {long_term2m}')
    assert πsim.sum().round(8) == 1 and πexact.sum().round(8) == 1
    
    print(abs(πexact - πsim).max())
    print(πexact - πsim)

if __name__ == '__main__':
    main()