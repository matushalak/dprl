# @matushalak
# Markov Decision Process assignment 2 DPRL
'''
Assignment 2, default settings output:
--------------------------------------
Simulation Long-term: 10.44767
Stationary dist. ITER: 10.44283
Stationary dist. MATH: 10.445065045248867
Poisson Value ITER: 10.44506504564523
Minimal long term cost following optimal policy: 7.987500000470679
TIME 2.681994915008545
π_math @ P vs π_math: 6.938893903907228e-17
π_math vs π_sim 0.0015373303167421082
'''
# infinite horizon problem
from numpy import ndarray, zeros, arange, savetxt, array, full, argmin, einsum, empty, unique
from numpy.random import randint, uniform, seed
from numpy.linalg import eig
from itertools import product
from seaborn import heatmap
import matplotlib.pyplot as plt
import time
import argparse

def mdp(capacity:tuple[int, int], upto:int = 5) -> tuple[ndarray, list]:
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
    # Transition probability matrix (P) under FIXED policy
    P = zeros((len(STATES), 
                len(STATES)))

    # Row = FROM, Column = TO
    for FROM, (i1, i2) in enumerate(STATES):
        # if a == 0:
            # straight away include fixed policy condition at Depth 0
        if 1 in (i1, i2):
            # order up to 5
            # XXX change !!!!
            # possibility to sell at I = 1 after order
            TO = [STATES.index((i1+j1 if i1 > upto else upto+j1,
                                i2+j2 if i2 > upto else upto+j2))
                                for j1, j2 in product(range(-1, 1), repeat = 2)]
        
        else:
            # otherwise no order
            TO = [STATES.index((i1+j1, i2+j2))
                  for j1, j2 in product(range(-1, 1), repeat = 2)
                  if 0 not in (i1+j1, i2+j2)]
    
        # dictionary lookup: faster over if/else or match/case
        assert len(TO) != 3
        p = {1 : 1.0, 2 : 0.5, 4 : 0.25}
        # for that FROM state, give corresponding TO states and corresponding probabilities
        P[FROM, TO] = p[len(TO)] # this gives the filled transition probability matrix
        assert len(set(TO)) not in (1,2)
        assert P[FROM,:].sum() == 1

    # save P matrix to inspect
    savetxt('2d.txt', P, fmt = '%.2f')
    # savez_compressed('full_mat', P)
    return P, STATES



# a)
# Action space (A) - possible orders, 2 independent actions
# theoretically, actions are order UP to levels {1, ..., 20} | I + a <= 20
# or orders of (0,0), (0,1), (1,0), ..., (19, 19) 
# EVERY ACTION = NEW TRANSITION PROBABILITY MATRIX
def order(state:tuple[int, int], max_invL:tuple[int, int]
          ) -> list:
    # all possible orders for given state
    max_order = [max_invL[0]-state[0],
                max_invL[1]-state[1]]
    
    all_orders = [(o1, o2) for o1, o2 in product(range(max_order[0]+1),
                                                 range(max_order[1]+1))]
    return all_orders



# c)
# start with random x0, fixed actions
def simulate(h:tuple[float, float], o:float, cap:tuple[int, int],
             upto:int = 5, 
             d:tuple[float,float] = (.5,.5),
             T_sim:int = 1e5) -> float:
    # average of all initial states
    long_run_costs = 0
    x = array([randint(1, cap[0]), randint(1, cap[1])])
    # print(x)
    for t in range(round(T_sim)):
        long_run_costs += x[0] * h[0] + x[1] * h[1] + (o if 1 in x else 0)

        ord = array([0,0])
        if 1 in x:
            # if t < 300:
            #     print(t, x)
            ord[x <= upto] = upto - x[x <= upto]

        demand = uniform(size = 2)
        x[demand < d] -= 1

        # orders arrive end of day
        x += ord

    return long_run_costs / T_sim



# d)
# start with random x0
def stationary_distribution(Xs:list, P:ndarray,
                            h:list[float, float], o:float, cap:tuple[int, int],
                            upto:int = 5,
                            iteration:bool = False, T_sim:int = 1e5, d:tuple[float,float] = (.5,.5)
                            ) -> tuple[float, ndarray, ndarray]:
    # simulated π calculation
    if iteration:
        x = array([randint(1, cap[0]), randint(1, cap[1])])
        # x = array((1,1))

        π = zeros(len(Xs))
        for t in range(round(T_sim)):
            π[Xs.index(tuple(x))] += 1

            ord = array([0,0])
            if 1 in x:
                ord[x <= upto] = upto - x[x <= upto]
            
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
        assert all((π @ P).round(10) == π.round(10)) # stationary dist = left eigenvector of P
    
    Xs = array(Xs)
    # elementwise multiplication
    holdingCs = full(Xs.shape, h) * Xs
    holdingCs = holdingCs.sum(axis = 1) # now a 1D vector
    
    # order costs = boolean mask multiplied by order costs
    orders = ((Xs[:,0] == 1) | (Xs[:,1] == 1)) * o

    # COSTS = holding costs + order costs
    C = holdingCs + orders
    return π @ C, π, C # π @ C dot product same as sum(π * C)



# e)
# initiate with random V0
def poisson_value_iteration(C:ndarray, P:ndarray,
                            epsilon:float = 1e-8) -> float:
    # Vt = C
    # Vt = zeros(C.shape)
    Vt = randint(1,100,C.shape)
    delta = 1 # arbitrary
    
    while delta > epsilon:
        Vt1 = C + P @ Vt
        delta = max(Vt1 - Vt) - min(Vt1 - Vt)
        if delta > epsilon:
            Vt = Vt1
    
    return (Vt1 - Vt).mean()



def big_transition(STATES:list, capacity:tuple[int,int]
                   ) -> tuple[ndarray, list]:
    all_actions = order((1,1), max_invL=capacity)
    c1, c2 = capacity
    I1 = arange(1,c1+1)
    I2 = arange(1,c2+1)
    P = zeros((len(STATES), 
                len(STATES),
                len(all_actions)))
    
    # Create lookup dictionary for O(1) state to index mapping
    state_to_idx = {tuple(state): idx for idx, state in enumerate(STATES)}

    for a, (o1, o2) in enumerate(all_actions):
        # Row = FROM, Column = TO
        for FROM, (i1, i2) in enumerate(STATES):
            # General case for all possible actions
            # still take into account sales BUT also add orders to that
            # if that action cannot be performed on this inventory level, just take the demand without that action
            TO = [state_to_idx[(i1+j1+o1 if i1+j1+o1 <= c1 else i1+j1, 
                                i2+j2+o2 if i2+j2+o2 <= c2 else i2+j2 )]
                                for j1, j2 in product(range(-1, 1), repeat = 2)
                                if 0 not in (i1+j1+o1, i2+j2+o2)]

            p = {1 : 1.0, 2 : 0.5, 4 : 0.25}
            # for that FROM state, give corresponding TO states and corresponding probabilities
            P[FROM, TO, a] = p[len(set(TO))] # this gives the filled transition probability matrix
            # if len(set(TO)) in (1,2):
            #     print(f'act: {int(o1),int(o2)} state: {int(i1),int(i2)}, TO: {set(TO)}')
            assert P[FROM,:, a].sum() == 1
       
    return P, all_actions



# f)
# Bellman equation value iteration for optimal policy
def bellman(Xs:list, h:list[float, float], o:float, cap:tuple[int, int],
            epsilon:float = 1e-8
            ) -> tuple[float, ndarray, ndarray]:
    '''
    Vt+1 = min_a {C_a + P_a @ Vt}

    g)
    Seems to be optimal to only order when stock runs out -> 1 and even then, 
    to order little of the item (2) to minimize having to pay holding costs for a long time
    '''
    Pa, actions = big_transition(Xs,capacity=cap)
    Xs = array(Xs)
    actions = array(actions)
    # doesn't change
    holding = full(Xs.shape, h) * Xs
    holding = holding.sum(axis = 1) # now a 1D vector
    
    # calculate order costs for all actions where something ordered
    # always order cost except when nothing ordered a[0]
    Cxa = full((Xs.shape[0],actions.shape[0]), holding + o).T
    # no order costs for action (0,0) unless one of inventories = 5 (in that case HAVE to order)
    Cxa[((Xs[:,0] != 1) & (Xs[:,1] != 1)), 0] = holding[((Xs[:,0] != 1) & (Xs[:,1] != 1))]
    
    # start as immediate costs
    # Vt = holding.copy()
    Vt = zeros(holding.shape)
    Vt1 = zeros(Vt.shape)
    Policies = full(Xs.shape, [0,0]) # initialize without ordering anything

    delta = 10 # arbitrary
    loop = 0
    while delta > epsilon:
        loop += 1
        # Immediate = Cxa
        # Past costs
        past_a = einsum('ijk,j->ik', Pa, Vt)
        combined_C_a = Cxa + past_a

        Vt1[:] = combined_C_a.min(axis = 1)
        Policies[:] = actions[argmin(combined_C_a, axis = 1)]

        delta = (Vt1 - Vt).max() - (Vt1 - Vt).min()
        # delta = abs(Vt1-Vt).max()
        # print(loop, delta, res := (Vt1 - Vt).mean())
        if delta > epsilon:
            Vt[:] = Vt1

    return (Vt1 - Vt).mean(), Policies, actions



def visualize_policy(Pol:ndarray, A:ndarray, X:ndarray, 
                     h:tuple[float, float], o:float, cap:tuple[int,int],
                     show_plots:bool = False,) -> None:
    indextostate = {i:tuple(state-1) for i, state in enumerate(X)}
    
    pZ = zeros(cap)
    vals = empty(cap, dtype=object)
    for i, p in enumerate(Pol):
        # p[p>0] += 2 # just for fixed policy visualization
        pZ[indextostate[i]] = sum(p)
        vals[indextostate[i]] = f"({p[0]}, {p[1]})" if not all(p == 0) else '0'
    
    
    # Create a figure with a custom gridspec layout
    fig = plt.figure(figsize=(11, 6))

    # Calculate font size dynamically
    grid_cell_height = 11 / cap[0]  # Figure height divided by rows
    grid_cell_width = 6 / cap[1]   # Figure width divided by cols
    font_size = max(min(grid_cell_height*15, grid_cell_width*15), 10)  # Scale and set a minimum size

    hmp = heatmap(pZ,
                  annot = vals, fmt = 's',
                  linewidth=.8,
                  annot_kws = {'size':font_size},
                  cbar_kws={'boundaries': arange(0, pZ.max()+1),
                            'label':'Total number of ordered products'})
    hmp.invert_yaxis()
    hmp.set_yticklabels(arange(1, cap[0]+1))
    hmp.set_xticklabels(arange(1, cap[1]+1))
    plt.ylabel('Product 1 inventory level')
    plt.xlabel('Product 2 inventory level')
    plt.tight_layout()
    plt.savefig(f'optimal_policy-c:{cap}_h:{h}_o:{o}.png', dpi = 500)
    if show_plots:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cap', nargs='+', type = int, default = [20,20])
    parser.add_argument('-holdc', nargs='+', type = float, default = [1,2])
    parser.add_argument('-ordc', type = float, default = 5)
    parser.add_argument('-upto', type = float, default = 5)
    parser.add_argument('-showp', type = bool, default = False)
    
    return parser.parse_args()


#%% running the script
def main():
    seed(420)
    args = parse_args()

    cap = args.cap
    h = args.holdc
    o = args.ordc
    upto = max(cap) if args.upto > max(cap) else args.upto
    plots = args.showp

    start = time.time()
    # a), b)
    P, X  = mdp(capacity=cap, upto=upto)
    
    # c)
    long_term1 = simulate(h = h, o = o, cap = cap, upto=upto)
    print(f'Simulation Long-term: {long_term1}')
    
    # d)
    long_term2, πsim, COSTS = stationary_distribution(X, P, iteration=True,
                                                      h = h, o = o, cap = cap, upto=upto)
    print(f'Stationary dist. ITER: {long_term2}')
    long_term2m, πexact, COSTS = stationary_distribution(X, P, 
                                                         iteration=False,
                                                         h = h, o = o, cap = cap, upto=upto)
    print(f'Stationary dist. MATH: {long_term2m}')
    assert πsim.sum().round(8) == 1 and πexact.sum().round(8) == 1

    # e)
    ltc3 = poisson_value_iteration(COSTS, P)
    print(f'Poisson Value ITER: {ltc3}')

    # f)
    ltc4, POL, A = bellman(X, h = h, o = o, cap = cap)
    print('Minimal long term cost following optimal policy:', ltc4)

    end = time.time()
    print('TIME', end - start)
    # g)
    visualize_policy(POL, A, array(X), h = h, o = o, cap = cap, show_plots= plots)

    # checks
    print("π_math @ P vs π_math:", max(abs((πexact @ P) - πexact)))
    print('π_math vs π_sim', abs(πexact - πsim).max()) # smaller w more simulations
    
if __name__ == '__main__':
    main()