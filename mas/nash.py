import nashpy as nash
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Row = [[1,7,1],
       [1,4,3],
       [3,1,4]]

Col = [[6,8,12],
       [1,2,1],
       [5,1,1]]

R = np.array(Row)
C = np.array(Col)

Game = nash.Game(R, C)

NE = Game.support_enumeration()
print('True NE:', list(NE))

nplays = 100
NEfp = Game.fictitious_play(nplays)
res = list(NEfp)[-1]
print('FP NE:', res)
plt.plot(res[0])
plt.plot(res[1])
plt.show()