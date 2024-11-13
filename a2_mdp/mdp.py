# @matushalak
# Markov Decision Process assignment 2 DPRL
# infinite horizon problem
from numpy import ndarray, zeros, arange, savetxt
from itertools import product

def mdp(capacity:list[int, int] = [20, 20], holding_costs:list[float, float] = [1, 2], order_costs: float = 5):
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

    # Row = FROM, Column = TO
    for FROM, (i1, i2) in enumerate(STATES):
            TO = [STATES.index(TOTO:= (i1+j1, i2+j2)) for j1 in range(-1, 1) for j2 in range(-1, 1) 
                if 0 not in (i1+j1, i2+j2)]

            # my interpretation Right or wrong?
            # dictionary lookup: faster over if/else or match/case
            assert len(TO) != 3
            p = {1 : 1.0, 2 : 0.5, 4 : 0.25}
            # for that FROM state, give corresponding TO states and corresponding probabilities
            P[FROM, TO] = p[len(TO)] # this gives the filled transition probability matrix

    # save P matrix to inspect
    savetxt('2d.txt', P, fmt = '%.2f')


# a)
# Action space (A) - possible orders, 2 independent actions
# theoretically, actions are order UP to levels {1, ..., 20} | I + a <= 20
# or orders of (0,0), (0,1), (1,0), ..., (19, 19) 
#   => 361 options, I guess policy iteration is choosing between those options?
# XXX QUESTION: in b-e can we just incorporate actions into transition probability matrix, since deterministic?
# probably the function doesn't make sense for the fixed case
def order(state:tuple[int, int], max_invL:tuple[int, int] = [20, 20], fixed:bool = False):
    if fixed: # fixed order policy
        # FIXED POLICY: if either Inventory level = 1: order so that both up to 5
        return (5-state[0], 5-state[1]) if 1 in state else [0,0]
    
    else: # all possible orders
        max_order = [max_invL[0]-state[0],
                    max_invL[1]-state[1]]
        min_order = [0 if state[0] > 1 else 1,
                    0 if state[1] > 1 else 1]
        
        all_orders = [(o1, o2) for o1, o2 in product(arange(min_order[0], max_order[0]+1),
                                                    arange(min_order[1], max_order[1]+1))]
        return all_orders
    
# TODO
def costs(X:list[tuple[int, int]], h = list[float, float], o = float):
    pass


# c)
# start with random x0
def simulate(X:ndarray, A:list, P:ndarray,
             h:list[float, float], o:float,
             T_sim:int = 1e3):
    long_run_average = 0
    return long_run_average


# d)
# start with random x0
def limiting_distribution(X:ndarray, A:list, P:ndarray,
                          h:list[float, float], o:float,
                          T_sim:1e3):
    long_run_average = 0
    return long_run_average


# e)
# initiate with random V0
def poisson_value_iteration(X:ndarray, P:ndarray,
                            epsilon:float):
    long_run_average = 0
    return long_run_average




#%% running the script
def main():
    mdp()

if __name__ == '__main__':
    main()