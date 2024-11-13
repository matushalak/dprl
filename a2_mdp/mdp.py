# @matushalak
# Markov Decision Process assignment 2 DPRL
# infinite horizon problem
from numpy import ndarray

def mdp(products:int = 2, demands:list[float, float] = [0.5, 0.5], capacity:list[int, int] = [20, 20],
        holding_costs:list[float, float] = [1, 2], order_costs:list[float, float] | int = 5):
    # a)
    # State space (X)
    # TODO ... 


    # Action space (A)
    # TODO ...


    # EASY VERSION b-e
    # FIXED POLICY: if either Inventory level = 1: order so that both up to 5

    # b)
    # Transitions (transition probabilities) (P)
    pass


# c)
# start with random x0
def simulate(X:ndarray, A:list, P:ndarray,
             h:dict[int:float], o:dict[int:float],
             T_sim:int = 1e3):
    long_run_average = 0
    return long_run_average


# d)
# start with random x0
def limiting_distribution(X:ndarray, A:list, P:ndarray,
                          h:dict[int:float], o:dict[int:float],
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
    # TODO
    # argparse to make it general IF time
    mdp()

if __name__ == '__main__':
    main()