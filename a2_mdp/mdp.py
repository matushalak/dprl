# @matushalak
# Markov Decision Process assignment 2 DPRL
# infinite horizon problem
from numpy import ndarray, zeros, arange, savetxt, where, array, full, argmin, isin
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
def order(state:tuple[int, int], max_invL:tuple[int, int] = [20, 20]):
    # all possible orders for given state
    max_order = [max_invL[0]-state[0],
                max_invL[1]-state[1]]
    min_order = [0 if state[0] > 1 else 1,
                0 if state[1] > 1 else 1]
    
    all_orders = [(o1, o2) for o1, o2 in product(arange(min_order[0], max_order[0]+1),
                                                arange(min_order[1], max_order[1]+1))]
    return array(all_orders)
    
# c)
# start with random x0, fixed actions
def simulate(h:tuple[float, float] = (1,2), o:float = 5, 
             d:tuple[float,float] = (.5,.5),
             T_sim:int = 1e5) -> float:
    # average of all initial states
    long_run_costs = 0
    x = randint(1, 21, 2)
    # x = array((1,1))
    # print(x)
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
    # simulated π calculation
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
                            epsilon:float = 1e-8):
    Vt = C
    delta = Vt.max() - Vt.min()
    
    while delta > epsilon:
        Vt1 = C + P @ Vt
        delta = max(Vt1 - Vt) - min(Vt1 - Vt)
        if delta > epsilon:
            Vt = Vt1
    
    return (Vt1 - Vt).mean()

# f)
# Bellman equation value iteration for optimal policy
def bellman(Xs:list, epsilon:float = 1e-8,
            h:list[float, float] = [1, 2], o:float = 5,):
    '''
    Vt+1 = max_a {C_a + P_a @ Vt}
    Vt+1(x) = max_a {C_a(x) + ∑_y p(y|x,a) * Vt(y)} 
    Vt+1(x) = max_a {C_a(x) + p_y @ Vt_y} # p_y probabilities where can go TO, costs where can go TO
    '''
    Xs = array(Xs)
    # doesn't change
    holding = full(Xs.shape, h) * Xs
    holding = holding.sum(axis = 1) # now a 1D vector
    # probabilities, always 4 options, every state
    probs = full(4, 0.25)
    
    # start as immediate costs
    Vt = holding.copy()
    Vt1 = zeros(Vt.shape)
    Policies = full(Xs.shape, [0,0]) # initialize without ordering anything

    delta = Vt.max() - Vt.min()
    loop = 0
    while delta > epsilon:
        # loop += 1
        # print(delta)
        for ix, x in enumerate(Xs):
            # all_relevant_actions
            actions = order(x)
            # holding cost already taken into account, independent of action
            costs = full(actions.shape[0], holding[ix], dtype=float)
            for ia, act in enumerate(actions):
                # add immediate order cost unless both orders 0
                cost_ord = (o if not all(act == 0) else 0) # elementwise

                possible_sales = array(list(product(range(-1,1), repeat = 2)))
                # -/0 possible sales + x[I1, I2] + orders[o1, o2]
                past_states = possible_sales + x + act
                # all possible states
                breakpoint()
                cost_past = probs @ Vt[(isin(Xs[:, 0], past_states[:, 0]) & 
                                        isin(Xs[:, 1], past_states[:, 1]))]

                costs[ia] += cost_ord + cost_past
            # breakpoint()    
            Vt1[ix] = min(costs)
            Policies[ix] = actions[argmin(costs)]

        # breakpoint()    
        delta = max(Vt1 - Vt) - min(Vt1 - Vt)
        if delta > epsilon:
            Vt[:] = Vt1
    breakpoint()
    return (Vt1 - Vt).mean(), Policies


#%% running the script
def main():
    # a), b)
    P, X  = mdp()
    
    # c)
    long_term1 = simulate()
    print(f'Simulation: {long_term1}')
    
    # d)
    long_term2, πsim, COSTS = stationary_distribution(X, P, iteration=True)
    print(f'Stationary ITER: {long_term2}')
    long_term2m, πexact, COSTS = stationary_distribution(X, P, iteration=False)
    print(f'Stationary MATH: {long_term2m}')
    assert πsim.sum().round(8) == 1 and πexact.sum().round(8) == 1

    # e)
    ltc3 = poisson_value_iteration(COSTS, P)
    print(f'Poisson: {ltc3}')

    # f)
    ltc4, POL = bellman(X)
    print('Minimal long term cost following optimal policy:', ltc4)
    # g)

    # checks
    print("π_math @ P vs π_math:", max(abs((πexact @ P) - πexact)))
    print('π_math vs π_sim', abs(πexact - πsim).max()) # smaller w more simulations
    
if __name__ == '__main__':
    main()