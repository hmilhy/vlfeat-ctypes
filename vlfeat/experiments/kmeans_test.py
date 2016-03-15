import numpy as np


import vlfeat

import pdb

#import sklearn.kmeans

data = np.random.rand(2,5000)
#data = np.asarray(data, dtype=np.int)

#pdb.set_trace()
centers = vlfeat.vl_kmeans(data, 10,verbosity=1)

pdb.set_trace()
