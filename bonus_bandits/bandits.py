from numpy import argmax, zeros, array
from numpy.random import uniform, choice, beta, binomial
from multiprocessing import Pool, cpu_count
import time
import argparse

'''
2 uniformly distributed Bernouli Bandits (0-1)

Number of Simulations: 10,000
len(simulation): 1,000

discount BETA = 0.99 (very little discounting), .99**1000 = 5e-5

Objective: Maximum discounted # of successes (when following a given policy)
6 Policies:
-  0 Full information (we know which arm has higher discounted reward and always choose that one - cheating)
-  1 Bayesian (update beliefs & Gittins Indices based on observed sucesses & failures)
   At each Time, play arm with highest gittins index
-  2 Thompson sampling ...
-  3 Greedy policy
-  4 Optimistic Q-learning (initial values 0)
-  5 Epsilon-greedy Q-learning (eps = 0.1)
-  6 Epsilon-first Q-learning (first 200 times random, then greedy)
'''
def run_simulation(args):
    duration, discount = args
    # Each simulation draw the "true" P(success) for each arm
    p1, p2 = uniform(size = 2)
    # Policies
    full = 0
    bayes = 0
    def gittins_approx(s, f):
        return s / (f + 1)
    thrompson = 0
    greedy = 0

    optim_greedy = 0
    lr = 0.1 # learning rate, introduce delay to explore more initially

    eps_greedy = 0
    epsilon = 0.1
    
    eps_first = 0
    first_eps = 200

    # full information, GOAL: TO approximate!
    full_cheat = max(p1, p2)

    # bayesian & thompson sampling: prior
    # parameters for corresponding beta distribution
    # [successes, failures]
    prior1 = [1,1]
    prior2 = [1,1]
    outcome_to_beta = {1:0, 0:1}

    # to track Q-values for each action
    optQs = array([1,1])
    Qs = array([0,0])

    for t in range(duration):
        # print(optQs, optim_greedy)
        # print(Qs ,greedy)
        # if t > 100:
        #     breakpoint()
        # full information
        full += full_cheat * discount**t

        # BANDIT OUTCOME
        # outcome of random draw of bandit (0 | 1) with p1 & p2 probabilities
        bandit_results = array([binomial(n=1, p = p1), binomial(n=1, p = p2)])

        # DECISION (based on previous knowledge)
        # first 200 times random, then greedy
        fa = choice([0,1]) if t < first_eps else argmax(Qs)

        # epsilon % times random, else greedy action
        eps_act = choice([0,1]) if uniform() < epsilon else argmax(Qs)

        # thompson sampling
        thompson_choice = argmax([beta(*prior1), beta(*prior2)])

        # gittins approximation choice
        gittins_choice = argmax([gittins_approx(*prior1), gittins_approx(*prior2)])

        # UPDATE REWARDS based on success of decision
        greedy += bandit_results[argmax(Qs)] * discount**t
        optim_greedy += bandit_results[argmax(optQs)] * discount**t

        eps_first += bandit_results[fa] * discount**t
        eps_greedy += bandit_results[eps_act] * discount**t

        thrompson += bandit_results[thompson_choice] * discount**t

        bayes += bandit_results[gittins_choice] * discount**t

        # UPDATE KNOWLEDGE
        # updated Qs only influence next round
        # update Qs for both bandits
        Qs = bandit_results * discount**t + (1-discount**t) * Qs
        # learning rate (0.1) plays a big role in how successful optimistic Q-learning is
        # slower lr = better (more exploration)
        optQs = optQs + lr * (discount** t * bandit_results - optQs)

        # update priors - adds to successes / fails based on outcome
        prior1[outcome_to_beta[bandit_results[0]]] += 1
        prior2[outcome_to_beta[bandit_results[1]]] += 1
    return array([full, bayes, thrompson,greedy ,optim_greedy,eps_greedy, eps_first])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nsim', type = int, default=10000)
    parser.add_argument('-discounting', type = float,default=0.99)
    parser.add_argument('-duration', type = int,default=1000)
    # TODO: add parameters for individual policy methods: epsilon, learning rate, ...
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    # Params
    simulations = args.nsim
    discount = args.discounting
    duration = args.duration
    # Different policies
    policies = zeros((7, simulations)).T

    # run_simulation((duration, discount)) # debug
    with Pool(processes = cpu_count()) as workers:
        policies[:,:] = workers.map(run_simulation, [(duration, discount)]*simulations)

    policies = policies.T
    print(f'Maximum discounted rewards (when following a given policy) after {simulations} simulations\n',
          '--------------------------------------------------------------------')
    print('Full Info', policies[0,:].mean())
    print('Bayesian Gittins', policies[1,:].mean()) # performs best!
    print('Thompson Samping', policies[2,:].mean())
    print('Greedy', policies[3,:].mean())
    print('Optimistic Greedy', policies[4,:].mean())
    print('Epsilon Greedy', policies[-2,:].mean())
    print('Epsilon-first Then Greedy', policies[-1,:].mean())
    print(f'Duration: {time.time()-start}')