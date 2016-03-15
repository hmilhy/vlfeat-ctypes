import vlfeat
import numpy as np
import pdb

import scipy.io as scio

import matplotlib.pyplot as plt
K = 3
#data = np.random.rand(2,1000)*255
#data=scio.loadmat('/home/sh/tmp/kmeans_data.mat')['data']
data1 = scio.loadmat('/home/sh/tmp/kmeans_data.mat')['data1']
data2 = np.asarray(data1, dtype=np.float32)
#data2 = np.asarray(data, dtype=np.float32)
C1,A1 = vlfeat.vl_ikmeans(data1,K,verbosity=1)

#C2,A2, energy=vlfeat.vl_kmeans(data2,K,verbosity=1)

pdb.set_trace()
colors = 'rgb'
markers = ('D','o','x')
for i in range(K):
    data_i = data1[:, A1==i]
    pdb.set_trace()
    plt.scatter(data_i[0], data_i[1], color=colors[i])
plt.show()
pdb.set_trace()

AT = vlfeat.vl_ikmeanspush(data, C)

pdb.set_trace()

hist = vlfeat.vl_ikmeanshist(K, AT)


pdb.set_trace()
