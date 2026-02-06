import numpy as np
mu, sig2 = 3.1, 0.75
v, tau2 = -2.4, 2
N = int(1e7)
# Sample from f ~ N(mu, sig2)
xf = np.random.normal(loc = mu, scale=np.sqrt(sig2), size = N)
# Sample from g ~ N(v, tau2)
xg = np.random.normal(loc = v, scale=np.sqrt(tau2), size = N)
univar_gauss_pdf = lambda x, mean, var: np.exp(-((x - mean)**2)/(2*var)
                                                )/np.sqrt(2*np.pi*var)
# Closed-form analytical solution
KLfg = 0.5*(np.log(tau2/sig2) + ((sig2-tau2+((mu - v)**2))/(tau2)))
KLgf = 0.5*(np.log(sig2/tau2) + ((tau2-sig2+((v - mu)**2))/(sig2)))
# MC solution KL(f||g)
f = univar_gauss_pdf(xf, mu, sig2)
g = univar_gauss_pdf(xf, v, tau2)
MCKLfg = np.mean(np.log(f/g))
# MC solution KL(g||f)
f = univar_gauss_pdf(xg, mu, sig2)
g = univar_gauss_pdf(xg, v, tau2)
MCKLgf = np.mean(np.log(g/f))
print(f'Analytical KL(f||g):{KLfg}\n',
        f'Monte-Carlo KL(f||g):{MCKLfg}\n',
        f'Analytical KL(g||f):{KLgf}\n',
        f'Monte-Carlo KL(g||f):{MCKLgf}'
        )
'''
We obtain the output:
Analytical KL(f||g):7.7404146265058635
Monte-Carlo KL(f||g):7.741587131368327
Analytical KL(g||f):20.509585373494136
Monte-Carlo KL(g||f):20.510246849263076
'''