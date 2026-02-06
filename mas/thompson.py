import numpy as np
import numpy.random as rand
from math import factorial as fact
from matplotlib import pyplot as plt

def single_arm():
    # Experiment setup
    n_trials = int(2e2)+1
    npdfs = 4
    pdfplot = n_trials // npdfs
    n_samples = int(1e3)
    p_bandit = 0.75 # target to estimate

    # Data collection
    mean_s = np.zeros(n_trials)
    var_s = np.zeros(n_trials)
    beta_pdfs = np.zeros((npdfs+1, n_samples))

    # Beta distribution essentials
    beta_prior = dict(a = 1, b = 1) # uniform prior
    beta_pdf = lambda x, a, b: (fact(a+b-1)/(fact(a-1)*fact(b-1))) * (x**(a-1)) * ((1-x)**(b-1))
    update_beta = {1:'a', 0:'b'}

    for t in range(n_trials):
        if t % pdfplot == 0:
            # Beta distribution density measured regularly
            beta_pdfs[t//pdfplot, :] = beta_pdf(np.linspace(0,1, n_samples), **beta_prior)
        beta_dist = rand.beta(**beta_prior, size = n_samples)
        mean_s[t] = np.mean(beta_dist) # empirically measure mean
        var_s[t] = np.var(beta_dist) # empirically measure variance
        # pull bandit arm
        reward = rand.binomial(n = 1, p = p_bandit, size = 1)[0]
        beta_prior[update_beta[reward]] += 1 # thompson update

    print(pdfplot)
    f, ax = plt.subplots(ncols=2)
    ax[0].plot(mean_s, label = r'E[B($\alpha, \beta$)]')
    ax[0].plot(var_s, label = r'Var[B($\alpha, \beta$)]')
    ax[0].legend(); ax[0].set_xlabel('Sampling iteration'); ax[0].set_ylabel('Beta distribution mean and variance')
    ax[1].plot(np.linspace(0,1, n_samples), beta_pdfs.T)
    ax[1].set_ylabel('Probability Density'); ax[1].set_xlabel('x')
    plt.show()

from typing import Literal
def bandit(n_iter:int, probs:list[float], mode:Literal['thompson', 'ucb'], c_param:float = None):
    K = len(probs)
    # expected reward of best arm = maximum success probability across bandits
    pstar = max(probs)
    # regrets
    total_regret = 0
    cum_regret = np.zeros(n_iter)
    if mode == 'ucb': 
        assert c_param is not None
        # Q-values for different arms
        qvals = np.zeros(K)
        # N times each arm pulled
        pulls = np.zeros(K)
        # for comparison with thompson 0:failure, 1:success
        fail_succes = np.zeros((K, 2))
        
        for t in range(n_iter):
            # Start by pulling each arm at least once
            if np.any(pulls < 1):
                chosen_arm = np.argmin(pulls)
            else: # then follow UCB rule
                ucb = c_param * np.sqrt(np.log(t) / pulls)
                ucb += qvals / pulls
                chosen_arm = np.argmax(ucb)
            reward = rand.binomial(n=1, p = probs[chosen_arm])
            qvals[chosen_arm] += reward
            pulls[chosen_arm] += 1
            # for comparison with thompson
            fail_succes[chosen_arm, reward] += 1
            # update total regret
            total_regret += pstar - probs[chosen_arm]
            cum_regret[t] = total_regret
        
        # Est. Posterior probabilities of success per arm (#success / (#pulls))
        print(fail_succes[:, 1] / fail_succes.sum(axis=1))
    
    elif mode == 'thompson':
        # col1 = a, col2 = b
        # initialized to 1,1
        beta_params = np.ones((K, 2))
        # 1: success (a), 0: failure (b)
        beta_update = {1:0, 0:1}
    
        for iter in range(n_iter):
            simulated_outcomes = rand.beta(a = beta_params[:, 0], b = beta_params[:, 1])
            best_arm = np.argmax(simulated_outcomes)
            
            reward = rand.binomial(n=1, p = probs[best_arm])
            beta_params[best_arm, beta_update[reward]] += 1

            # update total regret
            total_regret += pstar - probs[best_arm]
            cum_regret[iter] = total_regret
        
        # Est. Posterior probabilities of success per arm (#success / (#pulls))
        print(beta_params[:, 0] / beta_params.sum(axis=1))
    
    print(f'Total regret: {total_regret}')
    return cum_regret
        
if __name__ == '__main__':
    NITER = 1000
    regucb0 = bandit(n_iter=NITER, probs=(0.1, 0.6, 0.85), mode='ucb', c_param=0.25)
    regucb1 = bandit(n_iter=NITER, probs=(0.1, 0.6, 0.85), mode='ucb', c_param=1)
    regucb2 = bandit(n_iter=NITER, probs=(0.1, 0.6, 0.85), mode='ucb', c_param=2)
    regucb3 = bandit(n_iter=NITER, probs=(0.1, 0.6, 0.85), mode='ucb', c_param=20)
    regthom = bandit(n_iter=NITER, probs=(0.1, 0.6, 0.85), mode='thompson')

    f, ax = plt.subplots()
    ax.plot(regthom, color = 'k', label = 'Thompson sampling')
    ax.plot(regucb0, color = 'r', label = 'UCB sampling (c = 0.25)')
    ax.plot(regucb1, color = 'b', label = 'UCB sampling (c = 1)')
    ax.plot(regucb2, color = 'g', label = 'UCB sampling (c = 2)')
    ax.plot(regucb3, color = 'purple', label = 'UCB sampling (c = 20)')
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('Log(Sampling iteration)')
    ax.set_ylabel('Log(Total regret)')
    ax.legend(loc = 2)
    f.tight_layout()
    plt.show()