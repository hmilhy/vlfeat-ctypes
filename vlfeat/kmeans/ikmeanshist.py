

import numpy as np
from vlfeat.misc.vl_binsum import vl_binsum
def vl_ikmeanshist(K, asgn):
    h=np.zeros([K,1],dtype=np.int)
    return vl_binsum(h, 1, asgn)
