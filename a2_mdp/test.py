import numpy as np
from itertools import product

# 1D easy case
def oned():
    DP = np.zeros((20,20), dtype = float)

    for i in range(1, 21):
        DP[i-1:i+1, i-1:i+1] = 0.25

    DP[0,0] = 0

    np.savetxt('1d.txt', DP, fmt = '%.2f')
    print(DP)

d1, d2 = 20, 20
# can never start in 0 and thus actions can't be taken based on 0,0
# and can never END in 0 (because orders arrive by end of day and FORCED to order once inventory 1)
I1 = np.arange(1,d1+1)
I2 = np.arange(1,d2+1)
STATES = [(i, j) for i, j in product(I1, I2)]

DP = np.zeros((len(STATES), 
               len(STATES)))

for FROM, (i1, i2) in enumerate(STATES):
        TO = [STATES.index(TOTO:= (i1+j1, i2+j2)) for j1 in range(-1, 1) for j2 in range(-1, 1) 
              if 0 not in (i1+j1, i2+j2)]
        print(f'FROM: {(i1, i2)} TO: {TOTO}')

        # my interpretation Right or wrong?
        # dictionary lookup: faster over if/else or match/case
        assert len(TO) != 3
        p = {1 : 1.0, 2 : 0.5, 4 : 0.25}
        # for that FROM state, give corresponding TO states and corresponding probabilities
        DP[FROM, TO] = p[len(TO)]

np.savetxt('2d.txt', DP, fmt = '%.2f')