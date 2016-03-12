import vlfeat
import numpy as np
import pdb



K = 3
data = np.random.rand(2,1000)*255
data = np.asarray(data, dtype=np.uint8)
data2 = np.asarray(data, dtype=np.float32)
C,A = vlfeat.vl_ikmeans(data,K,verbosity=1)



datat = np.random.rand(2,10000)*255
datat = np.asarray(datat, dtype=np.uint8)

pdb.set_trace()

AT = vlfeat.vl_ikmeanspush(datat, C)

hist = vlfeat.vl_ikmeanshist(K, AT)


pdb.set_trace()
