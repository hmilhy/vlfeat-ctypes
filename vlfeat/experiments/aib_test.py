import numpy as np
import vlfeat

import pdb


#Pcx = np.array([[.3, .3, 0, 0], [0, 0, .2, .2]])
Pcx = np.array([
    0.6813,    0.3028,    0.8216,
    0.3795,    0.5417,    0.6449,
    0.8318,    0.1509,    0.8180,
    0.5028,    0.6979,    0.6602,
    0.7095,    0.3784,    0.3420,
    0.4289,    0.8600,    0.2897,
    0.3046,    0.8537,    0.3412,
    0.1897,    0.5936,    0.5341,
    0.1934,    0.4966,    0.7271,
    0.6822,    0.8998,    0.3093])
Pcx=Pcx.reshape(3,10)
parents, cost = vlfeat.vl_aib(Pcx, verbosity=1)
for n in range(len(parents)):
    print('%d => %d '%(n, parents[n]))
pdb.set_trace()



parents_true=[5,5,6,6,7,7,1]
cost_true=[0.67301, 0.67301, 0.67301, 0.0000]

cut, map_, short = vlfeat.vl_aibcut(parents, 2)
cut_1 = [5,6]
map_1 = [1,1,2,2,1,2,0]
short_1 = [5,5,6,6,5,6,7]

