# @matushalak
# Markov Decision Process assignment 2 DPRL
# infinite horizon problem
from numpy import ndarray, zeros, arange, savetxt, einsum, unique, savez_compressed, float16
from itertools import product

def mdp(capacity:tuple[int, int] = [20, 20], holding_costs:tuple[float, float] = [1, 2], order_costs: float = 5):
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
    # make all other transition probability matrices
    # all possible actions 3rd dimension of P matrix
    all_actions = order(state = (1,1))
    # Transition probability matrix (P) without actions, action (0,0)
    P = zeros((len(STATES), 
                len(STATES),
                len(all_actions)+1),
                dtype= float16)


    for a, (o1, o2) in enumerate([(0,0)]+all_actions):
        # Row = FROM, Column = TO
        for FROM, (i1, i2) in enumerate(STATES):
            if a == 0:
                # straight away include fixed policy condition at Depth 0
                if 1 in (i1, i2):
                    # order up to 5
                    TO = [STATES.index((i1 if i1 >= 5 else 5,
                                        i2 if i2 >= 5 else 5))]
                
                else:
                    # otherwise no order
                    TO = [STATES.index((i1+j1, i2+j2)) 
                        for j1 in range(-1, 1) for j2 in range(-1, 1) 
                        if 0 not in (i1+j1, i2+j2)]
            
            # General case for all possible actions
            else:
                # breakpoint()
                # still take into account sales BUT also add orders to that
                TO = [STATES.index((i1+j1+o1 if i1+j1+o1 <= I1[-1] else i1+j1, # if that action cannot be performed on this inventory level, only take into account sales 
                                    i2+j2+o2 if i2+j2+o2 <= I2[-1] else i2+j2)) # if that action cannot be performed on this inventory level, only take into account sales
                      for j1 in range(-1, 1) for j2 in range(-1, 1) ]
                    #   if 0 not in (i1+j1, i2+j2)]
            


            # my interpretation Right or wrong?
            # dictionary lookup: faster over if/else or match/case
            assert len(TO) != 3
            p = {1 : 1.0, 2 : 0.5, 4 : 0.25}
            # for that FROM state, give corresponding TO states and corresponding probabilities
            P[FROM, TO, a] = p[len(TO)] # this gives the filled transition probability matrix

    # save P matrix to inspect
    savetxt('2d.txt', P[:,:,0], fmt = '%.2f')
    savez_compressed('full_mat', P)
    return STATES


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
    
# TODO
# need to take into account holding costs & order costs -> both immediate costs
# can do inside loop
# def costs(x:tuple[int, int], cap: tuple[int,int],
#           h = tuple[float, float], o = float,
#           fixed:bool = False):
#     C = zeros(cap)
#     # use fixed order policy, only 
#     if fixed:
#         pass
#     else:
#         pass


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